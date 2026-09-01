"""Grounding: residual decision state, targeted queries, and what they cost.

The HG-v2 side of the repository (plan sections 1-8). Phase 0 lives entirely
here and uses no model: it checks the abstractions against the index problem,
where H(J | M_A) = 1 bit exactly and the gap between a generic and a targeted
query is a theorem rather than a measurement.
"""

from .channels import (
    BitVectorSource,
    ExternalSource,
    QuerySpec,
    Response,
    Route,
    generic_bit_query,
    log2_alphabet,
    targeted_bit_query,
)
from .config import GroundingConfig, IndexSweep, Phase0Gate, load_grounding_config
from .costs import (
    ArmMetrics,
    aggregate_all,
    aggregate_arm,
    bootstrap_mean,
    coordination_overhead,
    empirical_response_entropy,
    group_by_arm,
    information_efficiency,
    overhead_status,
    zero_one_loss,
)
from .episodes import (
    GroundingEpisode,
    GroundingRunDir,
    read_episodes,
    write_episodes,
)
from .synthetic import (
    INDEX_TASK_VERSION,
    POLICIES,
    IndexItem,
    IndexTaskSpec,
    generate_index_items,
    run_arm,
    run_policy,
)

__all__ = [
    "BitVectorSource",
    "ExternalSource",
    "QuerySpec",
    "Response",
    "Route",
    "generic_bit_query",
    "log2_alphabet",
    "targeted_bit_query",
    "GroundingConfig",
    "IndexSweep",
    "Phase0Gate",
    "load_grounding_config",
    "ArmMetrics",
    "aggregate_all",
    "aggregate_arm",
    "bootstrap_mean",
    "coordination_overhead",
    "empirical_response_entropy",
    "group_by_arm",
    "information_efficiency",
    "overhead_status",
    "zero_one_loss",
    "GroundingEpisode",
    "GroundingRunDir",
    "read_episodes",
    "write_episodes",
    "INDEX_TASK_VERSION",
    "POLICIES",
    "IndexItem",
    "IndexTaskSpec",
    "generate_index_items",
    "run_arm",
    "run_policy",
]
