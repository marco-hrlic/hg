"""Metrics, dependence modelling, and resampling."""

from .matrix import ErrorMatrix, build_matrix, load_matrix, save_matrix
from .core import (
    ConditionMetrics,
    clean_consensus_wrong,
    condition_metrics,
    consensus_wrong,
    contested_share_of_residual,
    disagreement_rate,
    diversity_gap,
    localization_surplus,
    member_accuracy,
)
from .dependence import (
    DependenceFit,
    excess_consensus,
    fit_item_difficulty,
    independence_null,
    pairwise_error_correlation,
)
from .bootstrap import (
    BootstrapResult,
    bootstrap_metric,
    odds_ratio,
    paired_bootstrap,
)

__all__ = [
    "ErrorMatrix",
    "build_matrix",
    "load_matrix",
    "save_matrix",
    "ConditionMetrics",
    "condition_metrics",
    "consensus_wrong",
    "clean_consensus_wrong",
    "disagreement_rate",
    "contested_share_of_residual",
    "diversity_gap",
    "localization_surplus",
    "member_accuracy",
    "DependenceFit",
    "fit_item_difficulty",
    "pairwise_error_correlation",
    "independence_null",
    "excess_consensus",
    "BootstrapResult",
    "bootstrap_metric",
    "paired_bootstrap",
    "odds_ratio",
]
