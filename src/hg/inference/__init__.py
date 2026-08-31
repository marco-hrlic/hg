"""Model inference: one interface, several backends, identical row schema."""

from .base import Backend, GenerationSettings, InferenceItem, Prediction, get_backend, register_backend
from .prompts import PROMPT_REGISTRY, render_prompt
from .canonicalize import CanonicalizeResult, canonicalize

# Registering the simulator on import keeps the backend registry non-empty on
# a machine with no GPU stack, so the analysis pipeline is always runnable.
from . import simulator as _simulator  # noqa: F401

__all__ = [
    "Backend",
    "GenerationSettings",
    "InferenceItem",
    "Prediction",
    "get_backend",
    "register_backend",
    "PROMPT_REGISTRY",
    "render_prompt",
    "CanonicalizeResult",
    "canonicalize",
]
