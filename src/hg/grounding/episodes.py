"""The grounding episode: the item-level artifact of an HG-v2 run.

One episode is one item passed through the whole loop: private AI state, a
route, at most one query, one external response with a measured transmission
cost, and a final action. It is the HG-v2 analogue of `hg.schema.ItemResult`
and follows the same rules for the same reason (plan sections 2.2 and 25):

* flat and JSON-only, because the file has to outlive the code;
* provenance travels with each row, not in a sidecar;
* loss is recorded *before* and *after* grounding, so the value of the external
  information and the cost of acquiring it can be divided by any later
  analysis without rerunning anything expensive.

`ItemResult` is not reused: its `validate()` requires one of the M2-LLM
ensemble conditions HS/HD/DS/DD, and its member/adapter/anchor fields have no
referent here. See experiments/hg_v2/REUSE_AUDIT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..io import RunDir, ensure_dir, read_jsonl, write_jsonl
from ..schema import SchemaError
from .channels import QuerySpec, Response, Route

# The contract with downstream analysis, kept explicit for the same reason
# `hg.schema.MINIMUM_COLUMNS` is: a hand-rolled or older producer must fail
# loudly rather than have a missing column read as a null.
MINIMUM_COLUMNS = (
    "item_id",
    "task",
    "route",
    "initial_action",
    "initial_loss",
    "final_action",
    "final_loss",
    "external_raw_bits",
    "external_cost",
)


@dataclass(slots=True)
class GroundingEpisode:
    """One item's grounding history (plan section 4.3)."""

    # --- identity -------------------------------------------------------
    item_id: str
    task: str
    policy: str  # the fixed policy or router that chose the route
    route: str  # a Route value

    # --- the two private states -----------------------------------------
    # Kept separately and explicitly. The whole experiment is about an
    # information asymmetry, and an artifact that merged them would make the
    # asymmetry unreconstructable after the fact.
    ai_private_state: dict[str, Any] = field(default_factory=dict)
    hidden_reference_state: dict[str, Any] = field(default_factory=dict)

    # --- decision state at route time -----------------------------------
    # Estimator outputs in general; analytically exact in the Phase 0
    # synthetic setting, where the joint distribution is constructed. Never to
    # be read as Shannon quantities elsewhere without a validated derivation
    # (plan sections 4.1 and 18).
    decision_state: dict[str, Any] = field(default_factory=dict)

    # --- acting without grounding ---------------------------------------
    initial_action: Any = None
    initial_loss: float = float("nan")

    # --- grounding ------------------------------------------------------
    query: dict[str, Any] | None = None
    external_response: dict[str, Any] | None = None
    # log2 |Y| of the response alphabet: what the source had to transmit.
    external_raw_bits: float = 0.0
    # Priced cost, external_raw_bits * lambda_bit. Separate from the raw bits
    # so a re-analysis can reprice a saved run at a different lambda.
    external_cost: float = 0.0

    # --- after grounding ------------------------------------------------
    final_action: Any = None
    final_loss: float = float("nan")

    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- api ---
    @property
    def distortion_reduction(self) -> float:
        """D_before - D_after. The numerator of information efficiency."""
        return float(self.initial_loss - self.final_loss)

    def validate(self) -> None:
        try:
            route = Route(self.route)
        except ValueError as exc:
            raise SchemaError(
                f"unknown route {self.route!r}; expected one of "
                f"{[r.value for r in Route]}"
            ) from exc
        if self.external_raw_bits < 0:
            raise SchemaError(f"external_raw_bits {self.external_raw_bits} < 0")
        if route is Route.ACT:
            if self.query is not None or self.external_response is not None:
                raise SchemaError("route ACT must not carry a query or a response")
            if self.external_raw_bits != 0.0:
                raise SchemaError(
                    f"route ACT paid {self.external_raw_bits} external bits; "
                    "acting now is by definition free of external communication"
                )
        else:
            if self.query is None:
                raise SchemaError(f"route {self.route} carries no query")
            if self.external_response is None:
                raise SchemaError(f"route {self.route} carries no response")
            got = float(self.external_response.get("raw_bits", float("nan")))
            if got != self.external_raw_bits:
                raise SchemaError(
                    f"response raw_bits {got} != episode external_raw_bits "
                    f"{self.external_raw_bits}; the cost accounting is the "
                    "result, so a mismatch is a hard error"
                )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "item_id": self.item_id,
            "task": self.task,
            "policy": self.policy,
            "route": self.route,
            "ai_private_state": self.ai_private_state,
            "hidden_reference_state": self.hidden_reference_state,
            "decision_state": self.decision_state,
            "initial_action": self.initial_action,
            "initial_loss": self.initial_loss,
            "query": self.query,
            "external_response": self.external_response,
            "external_raw_bits": self.external_raw_bits,
            "external_cost": self.external_cost,
            "final_action": self.final_action,
            "final_loss": self.final_loss,
            "metadata": self.metadata,
        }
        # Unlike ItemResult.to_dict, empty containers are kept: an episode with
        # no decision_state and one with an empty one are different claims, and
        # `query: null` on an ACT row is information, not noise.
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GroundingEpisode":
        known = {
            "item_id", "task", "policy", "route", "ai_private_state",
            "hidden_reference_state", "decision_state", "initial_action",
            "initial_loss", "query", "external_response", "external_raw_bits",
            "external_cost", "final_action", "final_loss", "metadata",
        }
        unknown = set(d) - known
        if unknown:
            raise SchemaError(f"unexpected fields in episode: {sorted(unknown)}")
        return cls(**d)  # type: ignore[arg-type]

    # Convenience for producers: build the grounded half from typed objects.
    def attach_grounding(
        self, query: QuerySpec, response: Response, *, lambda_bit: float = 1.0
    ) -> "GroundingEpisode":
        self.query = query.to_dict()
        self.external_response = response.to_dict()
        self.external_raw_bits = float(response.raw_bits)
        self.external_cost = float(response.raw_bits) * float(lambda_bit)
        return self


def check_minimum_columns(row: Mapping[str, Any]) -> None:
    missing = [c for c in MINIMUM_COLUMNS if c not in row]
    if missing:
        raise SchemaError(f"episode row is missing required columns: {missing}")


def write_episodes(path: str | Path, episodes: Sequence[GroundingEpisode]) -> int:
    for e in episodes:
        e.validate()
    return write_jsonl(path, [e.to_dict() for e in episodes])


def read_episodes(path: str | Path, strict: bool = True) -> list[GroundingEpisode]:
    out: list[GroundingEpisode] = []
    for row in read_jsonl(path):
        if strict:
            check_minimum_columns(row)
        e = GroundingEpisode.from_dict(row)
        if strict:
            e.validate()
        out.append(e)
    return out


def episodes_to_rows(episodes: Iterable[GroundingEpisode]) -> list[dict[str, Any]]:
    return [e.to_dict() for e in episodes]


class GroundingRunDir(RunDir):
    """`results/<run_id>/` for a grounding experiment.

    Subclasses rather than edits `RunDir`: the M2-LLM subdirectory names are a
    contract with existing result directories and with `hg.pipeline.reanalyze`.
    Everything shared -- config.resolved.yaml, figures/, report.md -- keeps the
    same location, so a results tree stays legible across both experiments.

        results/<run_id>/
            config.resolved.yaml   the resolved config that produced this run
            manifest.json          commit, timestamp, seed, task version, env
            episodes/<policy>.jsonl item-level episodes (the artifact that matters)
            figures/               plots, with CSV fallback if matplotlib is absent
            analysis.json          every computed quantity
            report.md              the Phase 0 table and the gate verdict
    """

    def create(self) -> "GroundingRunDir":
        for sub in ("episodes", "figures"):
            ensure_dir(self.root / sub)
        return self

    def episodes(self, policy: str) -> Path:
        return self.root / "episodes" / f"{policy}.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def analysis_path(self) -> Path:
        return self.root / "analysis.json"

    def available_policies(self) -> list[str]:
        d = self.root / "episodes"
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.jsonl"))
