"""Diversity as a measured variable, not a verbal label."""

from .overlap import jaccard, overlap_matrix, mean_overlap, overlap_report
from .splits import SplitPlan, build_splits, split_summary
from .survival import (
    ExcessSurvival,
    SurvivalPoint,
    excess_survival,
    residual_survival,
    survival_shape,
)

__all__ = [
    "jaccard",
    "overlap_matrix",
    "mean_overlap",
    "overlap_report",
    "SplitPlan",
    "build_splits",
    "split_summary",
    "ExcessSurvival",
    "SurvivalPoint",
    "excess_survival",
    "residual_survival",
    "survival_shape",
]
