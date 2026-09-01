"""Phase 0: the exact synthetic complementarity experiment (plan section 5).

No LLM, by design. The point is to check the new abstractions against a result
whose answer is known analytically *before* any of it is pointed at a model
(plan section 2.4).

The index problem
-----------------
The source privately holds M_H = (X_0, ..., X_{n-1}), i.i.d. Bernoulli(1/2).
The AI privately holds an index K, uniform on {0, ..., n-1}. The decision
variable is

    J = X_K.

So H(J | M_A) = H(X_K | K) = 1 bit exactly: the AI's own information pins down
*which* bit matters but says nothing about its value. Jointly the two states
suffice, H(J | M_A, M_H) = 0. Both facts hold by construction, not by estimate.

Why this is the right first experiment: the two ask policies reach identical,
perfect decision quality, so they cannot be separated by accuracy at all. The
only thing that distinguishes them is what the source had to transmit --
n bits for a generic dump versus 1 bit for a targeted query -- which is exactly
the axis the HG-v2 architecture claims to move. If the infrastructure cannot
show that separation here, where it is a theorem, no LLM result on it would
mean anything.

The arms
--------
`act`                   guess without asking. Accuracy -> 1/2, 0 external bits.
`ask_generic`           the source transmits all n bits. Accuracy 1, n bits.
`ask_targeted`          the AI asks for X_K. Accuracy 1, 1 bit. Omega = 0.
`ask_targeted_ablated`  negative control (plan section 24): the source does not
                        hold X_K. The query and its cost are unchanged, so the
                        AI pays 1 bit for a well-formed but uninformative
                        answer and accuracy falls back to 1/2. This is what
                        makes joint sufficiency a measured property of the
                        artifact rather than a claim in a docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .channels import (
    BitVectorSource,
    Route,
    generic_bit_query,
    targeted_bit_query,
)
from .costs import zero_one_loss
from .episodes import GroundingEpisode

# Bumped whenever the generated task distribution changes, so a saved run
# records which task it measured (plan section 25).
INDEX_TASK_VERSION = "index-v1"
TASK_NAME = "index"

POLICIES = ("act", "ask_generic", "ask_targeted", "ask_targeted_ablated")

# Candidate actions: report the value of the decision bit.
CANDIDATE_ACTIONS = ("0", "1")


@dataclass(frozen=True, slots=True)
class IndexTaskSpec:
    n_bits: int
    n_items: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_bits < 2:
            raise ValueError(
                f"n_bits must be >= 2, got {self.n_bits}: with one bit the "
                "index carries no information and a targeted query is a "
                "generic one"
            )
        if self.n_items < 1:
            raise ValueError(f"n_items must be >= 1, got {self.n_items}")


@dataclass(frozen=True, slots=True)
class IndexItem:
    """One instance of the index problem, holding both private states."""

    item_id: str
    n_bits: int
    index: int  # K -- AI-private
    bits: tuple[int, ...]  # M_H -- source-private

    @property
    def target(self) -> str:
        """The reference decision J = X_K, as an action label."""
        return str(self.bits[self.index])

    @property
    def bit_string(self) -> str:
        """Compact lossless encoding of M_H for the saved artifact."""
        return "".join(str(b) for b in self.bits)


def generate_index_items(spec: IndexTaskSpec) -> list[IndexItem]:
    """Generate the task. Deterministic given (seed, n_bits, n_items).

    The RNG is seeded on (seed, n_bits) so each point of the n-sweep is an
    independent draw that nonetheless reproduces exactly on its own -- one
    sweep point can be re-run without regenerating the others.
    """
    rng = np.random.default_rng([spec.seed, spec.n_bits])
    bits = rng.integers(0, 2, size=(spec.n_items, spec.n_bits))
    indices = rng.integers(0, spec.n_bits, size=spec.n_items)
    return [
        IndexItem(
            item_id=f"idx-n{spec.n_bits}-{i:06d}",
            n_bits=spec.n_bits,
            index=int(indices[i]),
            bits=tuple(int(b) for b in bits[i]),
        )
        for i in range(spec.n_items)
    ]


def _act_now_guess(item: IndexItem, seed: int) -> str:
    """The action the AI would take with no external information.

    Keyed on (seed, item) rather than drawn from a walking RNG stream so every
    policy sees the *same* act-now guess for the same item. That makes
    D_before identical across arms and the arms paired, which is what lets
    delta-distortion be compared between them at all.
    """
    rng = np.random.default_rng([seed, item.n_bits, item.index, hash_item(item)])
    return CANDIDATE_ACTIONS[int(rng.integers(0, 2))]


def hash_item(item: IndexItem) -> int:
    """Stable small integer id for an item, for RNG keying."""
    return int(item.item_id.rsplit("-", 1)[1])


def decision_state(item: IndexItem) -> dict[str, float | bool | int]:
    """The decision state at route time.

    Exact for this task, and labelled as such. Every quantity below is a
    property of the constructed joint distribution, not an estimator output --
    which is the point of Phase 0 and is precisely what will *not* be true once
    a learned estimator produces these fields in Phase 3 (plan section 18).
    """
    return {
        "exact": True,
        "candidate_actions": list(CANDIDATE_ACTIONS),
        # H(J | M_A) = 1 bit: K identifies which bit matters, never its value.
        "residual_deficit_bits": 1.0,
        # Expected 0/1 loss from acting on the prior: the decision bit is a
        # fair coin given M_A.
        "expected_loss_now": 0.5,
        # No amount of further internal computation reduces the deficit; the
        # missing information is not in M_A at all. This is the case where the
        # THINK route provably cannot help.
        "internal_reducibility": 0.0,
        # Expected loss reduction available from a sufficient external answer.
        "external_value": 0.5,
        "human_value": 0.5,
        # The reference is a determinate fact, so it is never contested and
        # DELIBERATE is not the right route (plan section 20).
        "contestedness": 0.0,
        "consequence": 1.0,
    }


def run_policy(
    policy: str,
    item: IndexItem,
    *,
    seed: int = 0,
    lambda_bit: float = 1.0,
) -> GroundingEpisode:
    """Run one item under one fixed policy and return the episode."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")

    target = item.target
    initial_action = _act_now_guess(item, seed)
    episode = GroundingEpisode(
        item_id=item.item_id,
        task=TASK_NAME,
        policy=policy,
        route=Route.ACT.value,
        ai_private_state={"index": item.index, "n_bits": item.n_bits},
        hidden_reference_state={
            "bits": item.bit_string,
            "decision_variable": target,
        },
        decision_state=decision_state(item),
        initial_action=initial_action,
        initial_loss=zero_one_loss(initial_action, target),
        final_action=initial_action,
        metadata={"task_version": INDEX_TASK_VERSION, "seed": seed},
    )

    if policy == "act":
        episode.final_loss = episode.initial_loss
        episode.validate()
        return episode

    # An independent RNG for the source, used only when it has to fabricate an
    # answer it does not know (the ablated arm).
    source_rng = np.random.default_rng([seed, 101, item.n_bits, hash_item(item)])

    if policy == "ask_generic":
        route = Route.ASK_GENERIC
        query = generic_bit_query(item.n_bits)
        source = BitVectorSource(item.bits, rng=source_rng)
    else:
        route = Route.ASK_TARGETED
        query = targeted_bit_query(item.index, item.n_bits)
        known = [True] * item.n_bits
        if policy == "ask_targeted_ablated":
            # Remove exactly the variable the decision turns on. Joint
            # sufficiency should fail: H(J | M_A, M_H') = 1 bit still.
            known[item.index] = False
        source = BitVectorSource(item.bits, rng=source_rng, known=known)

    response = source.respond(query)
    episode.route = route.value
    episode.attach_grounding(query, response, lambda_bit=lambda_bit)
    episode.final_action = _decode(policy, item, str(response.content))
    episode.final_loss = zero_one_loss(episode.final_action, target)
    episode.validate()
    return episode


def _decode(policy: str, item: IndexItem, content: str) -> str:
    """Turn the source's answer into a final action.

    The generic answer requires the AI to locate its own bit inside a payload
    it did not need: the extra n-1 bits are pure coordination overhead, and the
    decode is where that shows up as work the targeted arm never does.
    """
    if policy == "ask_generic":
        if len(content) != item.n_bits:
            raise ValueError(
                f"generic response has {len(content)} bits, expected {item.n_bits}"
            )
        return content[item.index]
    if len(content) != 1:
        raise ValueError(f"targeted response has {len(content)} symbols, expected 1")
    return content


def run_arm(
    policy: str,
    items: Sequence[IndexItem],
    *,
    seed: int = 0,
    lambda_bit: float = 1.0,
) -> list[GroundingEpisode]:
    return [
        run_policy(policy, item, seed=seed, lambda_bit=lambda_bit) for item in items
    ]
