"""The localization race (protocol sections 8, 15 Phase 4, and 17).

The practical question behind the whole programme: given a fixed budget of
human labels, which items should you spend it on? Five strategies, ranked by
how much they presuppose:

    random         audit a random sample -- the baseline any method must beat
    uncertainty    lowest ensemble confidence (M1 in section 17)
    disagreement   members disagree (M2) -- needs only internal diversity
    failure_memory items resembling previously-anchored failures (M3) -- needs
                   an external anchor history, i.e. the grounding architecture
    oracle         items the ensemble actually gets wrong -- the ceiling

`localization surplus = V_targeted - V_disagreement` is the decisive number.
Prediction P5 says disagreement sampling should match or beat anchored
targeting *unless* a clean shared core survives, and P6 says failure memory
wins when such a core exists and is clustered. The race is what distinguishes
them, and section 15 restricts Phase 4 to tasks where a clean core survived DD
-- running it elsewhere measures which method finds errors that internal
diversity would have surfaced for free.

Value definition
----------------
`V_strategy(B)` = fraction of the ensemble's *clean consensus failures* found
within a budget of B audited items. Consensus failures, not all errors: an item
one member gets wrong is already internally visible and does not need a human.
Clean, because contested items are an anchor problem and buying more labels for
them measures adjudication cost, not localization skill.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from ..metrics.matrix import ErrorMatrix

# A scorer ranks items: higher score = audit sooner.
Scorer = Callable[[ErrorMatrix], np.ndarray]


def random_score(matrix: ErrorMatrix, seed: int = 0) -> np.ndarray:
    """Deterministic pseudo-random ranking, keyed on item id.

    Keyed on the id rather than drawn from an RNG stream so the baseline does
    not shift when the item order changes.
    """
    return np.array(
        [
            int(
                hashlib.blake2b(
                    f"{seed}:{i}".encode("utf-8"), digest_size=8
                ).hexdigest(),
                16,
            )
            / 2**64
            for i in matrix.item_ids
        ]
    )


def uncertainty_score(matrix: ErrorMatrix) -> np.ndarray:
    """1 - mean ensemble confidence. Falls back to error count when absent.

    The fallback is not a silent substitution: a backend that reports no
    confidence cannot support an uncertainty strategy, and using error counts
    instead would leak the labels this strategy is not supposed to see. So the
    fallback is a *uniform* score, which degrades the strategy to random and
    makes the missing signal visible in the results rather than flattering it.
    """
    conf = matrix.confidences
    if np.all(np.isnan(conf)):
        return np.zeros(matrix.n_items)
    with np.errstate(invalid="ignore"):
        mean_conf = np.nanmean(conf, axis=1)
    mean_conf = np.where(np.isnan(mean_conf), 0.5, mean_conf)
    return 1.0 - mean_conf


def disagreement_score(matrix: ErrorMatrix) -> np.ndarray:
    """Vote entropy over member predictions.

    Uses only what the ensemble can see about itself -- no labels. This is the
    strategy that internal diversity makes available for free, and therefore
    the bar that any externally-anchored method has to clear.
    """
    n_i, n_m = matrix.predictions.shape
    scores = np.zeros(n_i)
    for i in range(n_i):
        counts: dict[object, int] = {}
        for j in range(n_m):
            key = matrix.predictions[i, j]
            counts[key] = counts.get(key, 0) + 1
        p = np.array(list(counts.values()), dtype=float) / n_m
        # Clamped at zero: the epsilon guard against log(0) makes a degenerate
        # distribution score -1e-12, and a negative disagreement score is not
        # only meaningless but can flip the tie-break ordering below.
        scores[i] = max(0.0, float(-(p * np.log(p + 1e-12)).sum()))
    return scores


def failure_memory_score(
    matrix: ErrorMatrix,
    anchored_history: Sequence[int],
    *,
    embeddings: np.ndarray | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Similarity to items already known (from anchors) to be consensus failures.

    This is the grounding architecture's own strategy: a memory of externally
    anchored failures, generalized to new items by representation similarity.

    `anchored_history` are item indices whose labels have already been bought.
    Without embeddings there is no way to generalize from them, so the score is
    uniform -- which is the honest answer, not a fallback to something that
    peeks at labels. A run that wants this strategy must set
    `embedding_dim > 0` on the backend.
    """
    n_i = matrix.n_items
    if embeddings is None or len(anchored_history) == 0:
        return np.zeros(n_i)

    hist = np.asarray(list(anchored_history), dtype=int)
    known_failures = hist[matrix.all_wrong()[hist]]
    if known_failures.size == 0:
        return np.zeros(n_i)

    emb = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.where(norms == 0, 1.0, norms)
    sims = unit @ unit[known_failures].T  # (n_items, n_known)
    # Max similarity to any known failure, temperature-smoothed. Max rather
    # than mean because a clustered core is the hypothesis being tested: mean
    # similarity would wash a tight cluster out against the bulk.
    return np.max(sims, axis=1) / max(temperature, 1e-6)


def oracle_score(matrix: ErrorMatrix) -> np.ndarray:
    """Clean consensus failures first. The ceiling, not a usable strategy."""
    return (matrix.all_wrong() & matrix.clean_mask).astype(float)


STRATEGIES = ("random", "uncertainty", "disagreement", "failure_memory", "oracle")


@dataclass(slots=True)
class RaceResult:
    strategy: str
    budgets: list[int]
    # Fraction of clean consensus failures found within each budget.
    recall: list[float]
    # Area under the recall-vs-budget curve, normalized to [0, 1].
    auc: float
    n_targets: int

    def value_at(self, budget: int) -> float:
        for b, r in zip(self.budgets, self.recall):
            if b >= budget:
                return r
        return self.recall[-1] if self.recall else float("nan")

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "budgets": self.budgets,
            "recall": self.recall,
            "auc": self.auc,
            "n_targets": self.n_targets,
        }


@dataclass(slots=True)
class LocalizationRace:
    results: dict[str, RaceResult] = field(default_factory=dict)
    n_items: int = 0
    n_targets: int = 0
    # Number of distinct labels the ensemble actually emitted. Load-bearing for
    # interpretation, not decoration: see `degenerate_for_disagreement`.
    n_labels: int = 0

    @property
    def degenerate_for_disagreement(self) -> bool:
        """True when the task's label space makes the race unfair by construction.

        With two labels, "every member is wrong" and "every member said the same
        thing" are the same event: a consensus failure has zero vote entropy, so
        the disagreement scorer ranks its own targets *last*. Any strategy then
        beats disagreement sampling on a binary task, and that says nothing
        about prediction P5 -- which is a claim about disagreement carrying
        localizing signal, not about binary arithmetic.

        Consequence for the experiment: P5 and P6 must be tested on task
        families with three or more labels (section 5.2's policy-application and
        agent tasks qualify; a binary harmfulness label does not). On a binary
        task the informative comparison is uncertainty vs. failure memory, both
        of which can still rank consensus failures.
        """
        return self.n_labels > 0 and self.n_labels < 3

    def surplus(self, budget: int, targeted: str = "failure_memory") -> float:
        """V_targeted - V_disagreement at a given budget (section 8)."""
        if targeted not in self.results or "disagreement" not in self.results:
            return float("nan")
        return self.results[targeted].value_at(budget) - self.results[
            "disagreement"
        ].value_at(budget)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "n_items": self.n_items,
            "n_targets": self.n_targets,
            "n_labels": self.n_labels,
            "degenerate_for_disagreement": self.degenerate_for_disagreement,
            "strategies": {k: v.to_dict() for k, v in self.results.items()},
        }
        if self.results:
            budgets = next(iter(self.results.values())).budgets
            out["localization_surplus"] = {
                str(b): self.surplus(b) for b in budgets
            }
            out["auc"] = {k: v.auc for k, v in self.results.items()}
        return out

    def interpret(self, budget: int | None = None) -> dict[str, object]:
        """Read the race against predictions P5 and P6."""
        if not self.results or self.n_targets == 0:
            return {
                "verdict": "not_applicable",
                "reason": (
                    "no clean consensus failures to localize; the race is only "
                    "informative where a clean residual exists (protocol "
                    "section 15, Phase 4)"
                ),
            }
        if self.degenerate_for_disagreement:
            budgets = next(iter(self.results.values())).budgets
            b = budget if budget is not None else budgets[len(budgets) // 2]
            return {
                "verdict": "not_applicable",
                "budget": b,
                "localization_surplus": self.surplus(b),
                "n_labels": self.n_labels,
                "reason": (
                    f"the ensemble emitted only {self.n_labels} distinct label(s), "
                    "so a consensus failure has zero disagreement by construction "
                    "and the disagreement strategy ranks its own targets last"
                ),
                "note": (
                    "Predictions P5 and P6 are not testable on a binary task: "
                    "'all wrong' and 'all agreed' are the same event, so any "
                    "strategy beats disagreement sampling for arithmetic reasons "
                    "rather than epistemic ones. Run the race on a task family "
                    "with three or more labels (protocol section 5.2, families B "
                    "and E). The uncertainty-vs-failure-memory comparison below "
                    "is still meaningful."
                ),
                "uncertainty_auc": (
                    self.results["uncertainty"].auc
                    if "uncertainty" in self.results else float("nan")
                ),
                "failure_memory_auc": (
                    self.results["failure_memory"].auc
                    if "failure_memory" in self.results else float("nan")
                ),
            }

        budgets = next(iter(self.results.values())).budgets
        b = budget if budget is not None else budgets[len(budgets) // 2]
        surplus = self.surplus(b)
        disagreement_auc = self.results["disagreement"].auc
        random_auc = self.results.get("random", RaceResult("random", [], [], float("nan"), 0)).auc

        verdict: str
        if np.isnan(surplus):
            verdict = "undetermined"
        elif surplus > 0.05:
            verdict = "targeting_wins"
        elif surplus < -0.05:
            verdict = "disagreement_wins"
        else:
            verdict = "tie"

        note = {
            "targeting_wins": (
                "Anchored failure memory localizes the clean residual better than "
                "diverse disagreement does. This is prediction P6's habitat: the "
                "clean shared core is large and clustered enough for external "
                "targeting to pay."
            ),
            "disagreement_wins": (
                "Diverse disagreement localizes at least as well as anchored "
                "targeting (prediction P5). Building an anchored failure-memory "
                "system on this task would buy labels for errors internal "
                "diversity already surfaces."
            ),
            "tie": (
                "No material difference at this budget. Not evidence for the "
                "grounding architecture: P5's default expectation is a tie or a "
                "disagreement win."
            ),
            "undetermined": "Insufficient data to compare strategies.",
        }[verdict]

        return {
            "verdict": verdict,
            "budget": b,
            "localization_surplus": surplus,
            "disagreement_auc": disagreement_auc,
            "random_auc": random_auc,
            "disagreement_beats_random": (
                bool(disagreement_auc > random_auc)
                if not (np.isnan(disagreement_auc) or np.isnan(random_auc))
                else None
            ),
            "note": note,
        }


def run_race(
    matrix: ErrorMatrix,
    *,
    budgets: Sequence[int] | None = None,
    embeddings: np.ndarray | None = None,
    history_size: int = 50,
    seed: int = 0,
    strategies: Sequence[str] = STRATEGIES,
) -> LocalizationRace:
    """Score every strategy on how fast it finds clean consensus failures.

    `history_size` items are set aside as the already-anchored history that
    failure memory learns from, and are excluded from every strategy's
    candidate pool so no strategy is credited for finding what it was told.
    """
    targets = matrix.all_wrong() & matrix.clean_mask
    n_i = matrix.n_items

    rng = np.random.default_rng(seed)
    history = rng.choice(n_i, size=min(history_size, n_i // 2), replace=False)
    candidate = np.ones(n_i, dtype=bool)
    candidate[history] = False
    cand_idx = np.flatnonzero(candidate)
    n_targets = int(targets[cand_idx].sum())

    if budgets is None:
        top = max(1, len(cand_idx))
        budgets = [
            b for b in (10, 25, 50, 100, 200, 400, 800, 1600) if b <= top
        ] or [top]

    scorers: dict[str, np.ndarray] = {}
    if "random" in strategies:
        scorers["random"] = random_score(matrix, seed)
    if "uncertainty" in strategies:
        scorers["uncertainty"] = uncertainty_score(matrix)
    if "disagreement" in strategies:
        scorers["disagreement"] = disagreement_score(matrix)
    if "failure_memory" in strategies:
        scorers["failure_memory"] = failure_memory_score(
            matrix, history, embeddings=embeddings
        )
    if "oracle" in strategies:
        scorers["oracle"] = oracle_score(matrix)

    n_labels = len({
        p for p in matrix.predictions.ravel().tolist()
        if not (isinstance(p, str) and p.startswith("<unparsed:"))
    })
    race = LocalizationRace(
        n_items=len(cand_idx), n_targets=n_targets, n_labels=n_labels
    )
    tie_break = random_score(matrix, seed + 977)

    for name, scores in scorers.items():
        sub = scores[cand_idx]
        # Sort by score, breaking ties with a fixed pseudo-random key so a
        # constant scorer degrades to random rather than to item order (which
        # would correlate with whatever order the dataset was built in).
        order = np.lexsort((tie_break[cand_idx], -sub))
        hit = targets[cand_idx][order]
        cum = np.cumsum(hit)
        recall = [
            float(cum[min(b, len(cum)) - 1] / n_targets) if n_targets else float("nan")
            for b in budgets
        ]
        auc = (
            float(np.trapezoid(recall, x=np.asarray(budgets, dtype=float)) / (budgets[-1] - budgets[0]))
            if len(budgets) > 1 and n_targets
            else float("nan")
        )
        race.results[name] = RaceResult(
            strategy=name,
            budgets=list(budgets),
            recall=recall,
            auc=auc,
            n_targets=n_targets,
        )
    return race
