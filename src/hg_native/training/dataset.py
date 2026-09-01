"""Combine the four synthetic generators into one episode stream.

Deliberately excludes spec section 6's `ordinary_lm` component (80% of the
recommended mixture): that needs the tokenizer/corpus decision, which
[[hg-native-m0-project-state]] records as explicitly deferred. Training on
this mixture alone is enough for the spec section 8.1 smoke stage ("test
loss decrease, finite gradients, head gradients, checkpointing, evaluator
roundtrip") but not a real M0 run -- the model would only ever see the
synthetic mini-language, exactly the "pure symbolic classifier" failure mode
spec section 6 warns `ordinary_lm` exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

from hg_native.data import accessibility, capability_refinement, sequential_quotient, static_hidden
from hg_native.data.schema import Episode, rng_for

FAMILIES = ("static_hidden", "accessibility", "sequential_quotient", "capability_refinement")


@dataclass(frozen=True)
class SyntheticMix:
    """Relative weights among the 4 synthetic families (renormalized internally).

    Not the spec section 6 `data_mix` itself (that also prices in
    `ordinary_lm`) -- just how to split the synthetic remainder.
    """

    static_hidden: float = 0.4
    accessibility: float = 0.3
    sequential_quotient: float = 0.2
    capability_refinement: float = 0.1

    def __post_init__(self) -> None:
        if self.probs_sum <= 0:
            raise ValueError("SyntheticMix proportions must sum to > 0")

    @property
    def probs_sum(self) -> float:
        return self.static_hidden + self.accessibility + self.sequential_quotient + self.capability_refinement

    def weights(self) -> list[float]:
        total = self.probs_sum
        return [
            self.static_hidden / total,
            self.accessibility / total,
            self.sequential_quotient / total,
            self.capability_refinement / total,
        ]


def sample_transition_pairs(
    seed: int, n: int, n_steps: int = 4, start_index: int = 0
) -> list[tuple[Episode, Episode]]:
    """Sample `n` consecutive `(episode_t, episode_t+1)` pairs from Family C
    trajectories, for training the transition head (spec section 3.3).

    Note the actual generator's `Z_{t+1}` transition (`r_{t+1} = f_t`, see
    `sequential_quotient.py`) does not depend on the action taken at all --
    it is an exogenous process the model observes, not one it controls.
    The spec's `g_T(S_t, A_t)` signature is action-conditioned regardless
    (section 3.3), so training against this family is a valid (if slightly
    degenerate) test of whether the transition head learns to *ignore* an
    irrelevant action input, rather than of action-conditioned control.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2 to have a next step, got {n_steps}")
    rng = rng_for(seed, "dataset_transition", start_index)
    pairs: list[tuple[Episode, Episode]] = []
    for offset in range(n):
        index = start_index + offset
        traj = sequential_quotient.generate_trajectory(seed, index, n_steps)
        t = int(rng.integers(0, n_steps - 1))
        pairs.append((traj[t], traj[t + 1]))
    return pairs


def sample_episodes(seed: int, n: int, mix: SyntheticMix | None = None, start_index: int = 0, n_steps_c: int = 4) -> list[Episode]:
    """Deterministically sample `n` episodes from the synthetic mixture.

    `start_index` lets a trainer draw successive, non-overlapping batches
    (`start_index=step*batch_size`) from one seed without repeating the same
    episodes every step; the same `(seed, start_index, n)` always reproduces
    the same batch.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    mix = mix or SyntheticMix()
    weights = mix.weights()
    rng = rng_for(seed, "dataset", start_index)

    episodes: list[Episode] = []
    for offset in range(n):
        index = start_index + offset
        family = FAMILIES[int(rng.choice(len(FAMILIES), p=weights))]
        if family == "static_hidden":
            subclass = static_hidden.SUBCLASSES[int(rng.integers(0, len(static_hidden.SUBCLASSES)))]
            episodes.append(static_hidden.generate_episode(seed, subclass, index))
        elif family == "accessibility":
            subclass = accessibility.SUBCLASSES[int(rng.integers(0, len(accessibility.SUBCLASSES)))]
            episodes.append(accessibility.generate_episode(seed, subclass, index))
        elif family == "sequential_quotient":
            traj = sequential_quotient.generate_trajectory(seed, index, n_steps_c)
            episodes.append(traj[int(rng.integers(0, n_steps_c))])
        else:  # capability_refinement
            pair = capability_refinement.generate_pair(seed, index)
            members = list(pair)
            episodes.append(pair[members[int(rng.integers(0, len(members)))]])
    return episodes
