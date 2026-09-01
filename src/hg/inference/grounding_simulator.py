"""A backend that answers from what the prompt actually reveals.

Registered as a backend so Phase 2 exercises the same code path a frozen
transformers model will use: swapping to a real model is a config change, not a
new pipeline. It is a test instrument and nothing it produces is evidence about
an LLM.

Why not reuse `hg.inference.simulator`: that one models an *ensemble's* error
structure -- family skill, lineage bias, a shared information-limited core --
which is the M2-LLM question. Phase 2 needs a different mechanism: whether the
answer is determined by the information in the prompt. So this backend reads
two fields off the item and nothing else:

    gs_sufficient       does the prompt in hand determine the correct action?
    gs_reference_action what that action is

When the prompt suffices it answers correctly (optionally with a configurable
error rate, to check that the analysis degrades sensibly for an imperfect
model). When it does not, it guesses -- and guesses with a *bias* toward
whichever route the visible numbers favour, because that is what a real model
does when a decision-relevant variable is missing. A bias rather than a coin
flip matters for one specific baseline: B1 in plan section 17 spends more
internal computation via self-consistency, and self-consistency cannot correct a
systematic bias. That the two come out the same is the point, and it would be
invisible against an unbiased guesser.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Mapping, Sequence

from .base import (
    GenerationSettings,
    InferenceItem,
    Prediction,
    register_backend,
)

META_REFERENCE_ACTION = "gs_reference_action"
META_SUFFICIENT = "gs_sufficient"
META_HINT = "gs_prior_hint"


def _uniform(seed: int, *keys: Any) -> float:
    payload = "\x1f".join(str(k) for k in keys).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, key=struct.pack("<q", seed)).digest()
    return int.from_bytes(digest, "little") / 2.0**64


@register_backend("grounding_simulator")
class GroundingSimulator:
    """Simulated primary LM for the HG-v2 controlled tasks."""

    name = "grounding_simulator"

    def __init__(
        self,
        *,
        seed: int = 0,
        # Error rate on items the prompt *does* determine. Zero models a model
        # that applies a stated rule reliably; raise it to check that the
        # analysis attributes the resulting distortion to capability rather
        # than to a missing variable.
        error_rate_when_sufficient: float = 0.0,
        # P(guess follows the visible numbers) when the prompt is insufficient.
        # 0.5 would be an unbiased coin; a real model is not unbiased.
        bias_strength: float = 0.8,
        **options: Any,
    ):
        if not 0.0 <= error_rate_when_sufficient <= 1.0:
            raise ValueError("error_rate_when_sufficient must be in [0, 1]")
        if not 0.0 <= bias_strength <= 1.0:
            raise ValueError("bias_strength must be in [0, 1]")
        self.seed = int(seed)
        self.error_rate_when_sufficient = float(error_rate_when_sufficient)
        self.bias_strength = float(bias_strength)
        self.options = dict(options)

    def load(self) -> None:  # no weights to acquire
        return None

    def unload(self) -> None:
        return None

    def predict(
        self, items: Sequence[InferenceItem], settings: GenerationSettings
    ) -> list[Prediction]:
        out: list[Prediction] = []
        for item in items:
            out.append(self._predict_one(item, settings))
        return out

    def _predict_one(
        self, item: InferenceItem, settings: GenerationSettings
    ) -> Prediction:
        labels = list(item.labels)
        meta: Mapping[str, Any] = item.meta or {}
        reference = str(meta.get(META_REFERENCE_ACTION, labels[0]))
        sufficient = bool(meta.get(META_SUFFICIENT, False))
        hint = str(meta.get(META_HINT, labels[0]))

        # Greedy decoding must be a deterministic function of the item alone.
        # Under sampling the draw also depends on the decoding seed, which is
        # what lets a caller take k independent samples by varying it -- the
        # only honest way to implement self-consistency through this interface.
        sample_key = () if settings.greedy else (settings.seed,)

        if sufficient:
            p_correct = 1.0 - self.error_rate_when_sufficient
            u = _uniform(self.seed, "suff", item.item_id, *sample_key)
            chosen = reference if u < p_correct else _other(labels, reference)
            confidence = p_correct
        else:
            u = _uniform(self.seed, "guess", item.item_id, *sample_key)
            chosen = hint if u < self.bias_strength else _other(labels, hint)
            confidence = self.bias_strength if chosen == hint else 1.0 - self.bias_strength

        probs = {label: 0.0 for label in labels}
        probs[chosen] = confidence
        remainder = 1.0 - confidence
        others = [label for label in labels if label != chosen]
        for label in others:
            probs[label] = remainder / len(others) if others else 0.0

        return Prediction(
            item_id=item.item_id,
            label=chosen,
            raw_output=chosen,
            parse_ok=True,
            confidence=confidence,
            probs=probs,
        )


def _other(labels: Sequence[str], label: str) -> str:
    for candidate in labels:
        if candidate != label:
            return candidate
    return label
