"""Backend interface.

Every backend takes the same items and returns the same `Prediction` objects,
so the dependence analysis never has to know whether a run came from the
simulator, HuggingFace transformers, or vLLM. Backends are registered by name
and selected from config, which is what keeps `configs/` the only input to a
run.

Determinism is part of the interface, not an afterthought: protocol section 13
requires "deterministic or tightly controlled inference for the primary
dependence analysis", because sampling noise is itself a source of ensemble
disagreement and would inflate D_E for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Decoding configuration, recorded on every row.

    Defaults are greedy. `temperature > 0` is allowed -- some conditions may
    want seed/order variation to come partly from sampling -- but then
    `seed` must be set so the run is reproducible.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 8
    seed: int | None = 0
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.temperature > 0 and self.seed is None:
            raise ValueError(
                "sampling with seed=None makes the run irreproducible; "
                "set a seed or use temperature=0"
            )

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "stop": list(self.stop),
        }


@dataclass(slots=True)
class InferenceItem:
    """One evaluation item as handed to a backend."""

    item_id: str
    prompt: str
    labels: Sequence[str]
    # Carried through untouched; the simulator reads latent fields from here.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    """What a backend returns for one item."""

    item_id: str
    label: str | None
    raw_output: str = ""
    parse_ok: bool = True
    confidence: float | None = None
    probs: dict[str, float] | None = None
    logprob: float | None = None
    embedding: list[float] | None = None


@runtime_checkable
class Backend(Protocol):
    """Minimal contract a backend must satisfy."""

    name: str

    def load(self) -> None:
        """Acquire weights / start a server. Must be idempotent."""

    def unload(self) -> None:
        """Release GPU memory. Called between families so a 5-family panel
        fits on one Spark without holding all checkpoints resident."""

    def predict(
        self, items: Sequence[InferenceItem], settings: GenerationSettings
    ) -> list[Prediction]:
        """Score items in order. Length and order of the result must match."""


_REGISTRY: dict[str, Callable[..., Backend]] = {}


def register_backend(name: str) -> Callable[[Callable[..., Backend]], Callable[..., Backend]]:
    def deco(factory: Callable[..., Backend]) -> Callable[..., Backend]:
        if name in _REGISTRY:
            raise ValueError(f"backend {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def get_backend(name: str, **kwargs: Any) -> Backend:
    """Instantiate a backend by name.

    GPU backends are imported lazily so the analysis half of the pipeline
    installs and runs without torch.
    """
    if name not in _REGISTRY:
        _import_optional(name)
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}. "
            "GPU backends need the 'gpu' extra installed."
        )
    return _REGISTRY[name](**kwargs)


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def _import_optional(name: str) -> None:
    module = {
        "transformers": ".hf",
        "hf": ".hf",
        "vllm": ".vllm_backend",
    }.get(name)
    if module is None:
        return
    try:
        __import__(f"{__package__}{module}", fromlist=["_"])
    except ImportError:
        # Leave the registry untouched; get_backend raises a useful KeyError.
        pass


def validate_predictions(
    items: Sequence[InferenceItem], preds: Sequence[Prediction]
) -> None:
    """Backends must preserve item order and count."""
    if len(preds) != len(items):
        raise ValueError(f"backend returned {len(preds)} predictions for {len(items)} items")
    for item, pred in zip(items, preds):
        if pred.item_id != item.item_id:
            raise ValueError(
                f"backend reordered items: expected {item.item_id!r}, got {pred.item_id!r}"
            )
