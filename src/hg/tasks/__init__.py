"""Dataset construction, kept decoupled from inference and analysis."""

from .synthetic import SyntheticSpec, generate_synthetic_panel

__all__ = ["SyntheticSpec", "generate_synthetic_panel"]
