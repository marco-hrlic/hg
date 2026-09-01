"""Phase 1's task: sometimes the AI already knows enough (plan section 9).

Phase 0's index problem always needed the source, so routing was trivial. Phase
1 needs a task where the optimal route genuinely varies per item and is known:

    "With probability p, J is directly observable from M_A. Otherwise it
     depends on one hidden human/source variable."   (plan section 9)

Construction
------------
The source holds M_H = (X_0, ..., X_{n-1}) as before, the AI holds the index K,
and J = X_K. What is new is that the AI also holds `m` internal observations of
X_K, each independently flipped with a per-item error rate e_i:

    e_i = 0            with probability p       -> J is directly observable
                                                   from M_A; asking is waste
    e_i ~ error_rates  otherwise               -> the observations are noisy,
                                                   and how noisy varies

So the AI's act-now action is the majority vote of its own observations. It is
right for free on the self-sufficient items and unreliable elsewhere.

Why the noise is graded rather than binary: if the AI simply *knew* whether it
was self-sufficient, the routing feature would predict the label perfectly and
there would be no learning problem, only a lookup. With graded noise the router
sees the *agreement* among its own observations -- an internal signal that is
informative about whether acting is safe but never conclusive. That is the
realistic shape of the problem, and it is also the same signal the M2-LLM
experiment measures across ensemble members.

Counterfactuals
---------------
Every item records the loss under *both* routes, because plan section 9 asks for
counterfactual outcomes rather than only the realized one. That is what makes
the oracle route, and therefore routing regret, computable at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .channels import BitVectorSource, Route, targeted_bit_query
from .costs import zero_one_loss
from .episodes import GroundingEpisode

NOISY_INDEX_TASK_VERSION = "noisy-index-v1"
TASK_NAME = "noisy_index"

CANDIDATE_ACTIONS = ("0", "1")

# A targeted ask resolves the decision for one bit, exactly as in Phase 0.
ASK_BITS = 1.0
RESIDUAL_DEFICIT_BITS = 1.0


@dataclass(frozen=True, slots=True)
class NoisyIndexTaskSpec:
    n_bits: int = 8
    n_items: int = 4000
    # Internal observations of X_K available to the AI without asking.
    n_observations: int = 5
    # Share of items where the AI's observations are noiseless, i.e. J is
    # directly observable from M_A and the optimal route is ACT.
    p_self_sufficient: float = 0.4
    # Per-observation flip rates for the remaining items, drawn uniformly.
    # Values near 0.5 make the internal evidence almost worthless.
    error_rates: tuple[float, ...] = (0.25, 0.45)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_bits < 2:
            raise ValueError(f"n_bits must be >= 2, got {self.n_bits}")
        if self.n_items < 1:
            raise ValueError(f"n_items must be >= 1, got {self.n_items}")
        if self.n_observations < 1 or self.n_observations % 2 == 0:
            raise ValueError(
                f"n_observations must be odd and >= 1, got {self.n_observations}: "
                "an even panel can tie, and a tie-break rule would be a second, "
                "undocumented policy inside the task"
            )
        if not 0.0 <= self.p_self_sufficient <= 1.0:
            raise ValueError(f"p_self_sufficient must be in [0, 1], got {self.p_self_sufficient}")
        if not self.error_rates:
            raise ValueError("error_rates must not be empty")
        if any(not 0.0 <= e <= 0.5 for e in self.error_rates):
            raise ValueError(
                f"error_rates must be in [0, 0.5], got {self.error_rates}: above "
                "0.5 an observation is anti-correlated with the truth, which the "
                "AI could simply invert"
            )


@dataclass(frozen=True, slots=True)
class NoisyIndexItem:
    item_id: str
    n_bits: int
    index: int  # K -- AI-private
    bits: tuple[int, ...]  # M_H -- source-private
    # The AI's own noisy views of X_K. AI-private, and the only evidence the
    # router is allowed to see.
    observations: tuple[int, ...]
    # Generative truth, recorded for analysis and never shown to the router.
    error_rate: float

    @property
    def target(self) -> str:
        return str(self.bits[self.index])

    @property
    def bit_string(self) -> str:
        return "".join(str(b) for b in self.bits)

    @property
    def self_sufficient(self) -> bool:
        """True when M_A already determines J (error rate exactly zero)."""
        return self.error_rate == 0.0

    @property
    def act_now_action(self) -> str:
        """Majority vote of the AI's own observations."""
        ones = sum(self.observations)
        return "1" if ones * 2 > len(self.observations) else "0"

    @property
    def loss_act(self) -> float:
        """Counterfactual 0/1 loss if the AI acts on internal evidence alone."""
        return zero_one_loss(self.act_now_action, self.target)

    @property
    def loss_ask(self) -> float:
        """Counterfactual loss if the AI asks. The source is truthful and holds
        X_K, so a targeted ask always resolves the decision."""
        return 0.0


def generate_noisy_index_items(spec: NoisyIndexTaskSpec) -> list[NoisyIndexItem]:
    """Generate the task. Deterministic given the spec."""
    rng = np.random.default_rng([spec.seed, spec.n_bits, spec.n_observations])
    n, m = spec.n_items, spec.n_observations

    bits = rng.integers(0, 2, size=(n, spec.n_bits))
    indices = rng.integers(0, spec.n_bits, size=n)
    self_sufficient = rng.random(n) < spec.p_self_sufficient
    drawn = np.asarray(spec.error_rates, dtype=float)[
        rng.integers(0, len(spec.error_rates), size=n)
    ]
    error_rates = np.where(self_sufficient, 0.0, drawn)
    flips = rng.random((n, m)) < error_rates[:, None]

    items: list[NoisyIndexItem] = []
    for i in range(n):
        truth = int(bits[i, indices[i]])
        observations = tuple(int(truth ^ int(f)) for f in flips[i])
        items.append(
            NoisyIndexItem(
                item_id=f"nidx-n{spec.n_bits}-{i:06d}",
                n_bits=spec.n_bits,
                index=int(indices[i]),
                bits=tuple(int(b) for b in bits[i]),
                observations=observations,
                error_rate=float(error_rates[i]),
            )
        )
    return items


# --------------------------------------------------------- true error rates ---
def true_act_error_probability(item: NoisyIndexItem) -> float:
    """P(the majority vote is wrong) given the item's *generative* noise level.

    Conditions on e_i, which the AI never sees. Recorded for analysis only.

    This is deliberately **not** used as a reference policy. Conditioning on
    e_i is neither more nor less informative than conditioning on the observed
    votes -- a unanimous vote at e = 0.45 is safer than a 3-2 vote at the same
    e -- so a policy built on it is not an upper bound on a feature-based
    router, and either can win. The actual ceiling is
    `bayes_act_error_probability`.
    """
    m = len(item.observations)
    e = item.error_rate
    if e == 0.0:
        return 0.0
    need = m // 2 + 1
    return float(
        sum(math.comb(m, k) * e**k * (1 - e) ** (m - k) for k in range(need, m + 1))
    )


def noise_mixture(spec: NoisyIndexTaskSpec) -> tuple[tuple[float, float], ...]:
    """The prior over per-item error rates, as (weight, error_rate) pairs."""
    rest = 1.0 - spec.p_self_sufficient
    per = rest / len(spec.error_rates)
    return ((spec.p_self_sufficient, 0.0),) + tuple(
        (per, float(e)) for e in spec.error_rates
    )


def bayes_act_error_probability(
    observations: Sequence[int], spec: NoisyIndexTaskSpec
) -> float:
    """P(acting on the internal vote is wrong | the observations).

    The exact posterior, marginalizing the unknown per-item error rate over the
    generative mixture. This is the ceiling for any router that sees only the
    AI's own evidence: it uses the same features plus perfect knowledge of the
    noise distribution, which a learned router has to infer from data.

    A learned router materially *beating* this is a signal to check for a leak,
    not a result. Small differences either way are sampling noise, since this
    bounds expected cost and a run measures realized cost on finitely many
    items.

    Because the noise is symmetric and below 1/2, the posterior is monotone in
    the vote count, so the Bayes-optimal action is the majority vote -- the same
    action the task's `loss_act` scores.
    """
    m = len(observations)
    if m == 0:
        raise ValueError("need at least one observation")
    k = int(sum(observations))
    mixture = noise_mixture(spec)
    # Likelihood of the observed votes under each hypothesis for X_K. Under
    # X_K = 1 the (m - k) zeros are the flipped ones, and vice versa.
    like_1 = sum(w * e ** (m - k) * (1 - e) ** k for w, e in mixture)
    like_0 = sum(w * e**k * (1 - e) ** (m - k) for w, e in mixture)
    total = like_1 + like_0
    if total <= 0.0:
        return 0.5
    p1 = like_1 / total
    acts_one = k * 2 > m
    return float(1.0 - p1) if acts_one else float(p1)


# ------------------------------------------------------------------ episodes ---
def build_episode(
    item: NoisyIndexItem,
    route: Route,
    *,
    policy: str,
    decision_state: dict,
    lambda_bit: float,
    seed: int = 0,
) -> GroundingEpisode:
    """Run one item under one already-chosen route.

    Routing is the caller's job -- a fixed policy, an oracle or a learned
    router -- so that every route choice runs through exactly this code and the
    episodes stay comparable across policies.
    """
    initial_action = item.act_now_action
    episode = GroundingEpisode(
        item_id=item.item_id,
        task=TASK_NAME,
        policy=policy,
        route=route.value,
        ai_private_state={
            "index": item.index,
            "n_bits": item.n_bits,
            "observations": list(item.observations),
        },
        hidden_reference_state={
            "bits": item.bit_string,
            "decision_variable": item.target,
            # Generative truth, kept for analysis. Never an input to a router.
            "error_rate": item.error_rate,
            "self_sufficient": item.self_sufficient,
        },
        decision_state=decision_state,
        initial_action=initial_action,
        initial_loss=item.loss_act,
        final_action=initial_action,
        final_loss=item.loss_act,
        metadata={
            "task_version": NOISY_INDEX_TASK_VERSION,
            "seed": seed,
            # Counterfactuals for both routes (plan section 9), so oracle cost
            # and routing regret are recomputable from the artifact alone.
            "loss_act": item.loss_act,
            "loss_ask": item.loss_ask,
            "bits_act": 0.0,
            "bits_ask": ASK_BITS,
            "lambda_bit": lambda_bit,
            "true_act_error_probability": true_act_error_probability(item),
        },
    )

    if route is Route.ACT:
        episode.validate()
        return episode

    if route is not Route.ASK_TARGETED:
        raise ValueError(
            f"Phase 1 routes between ACT and ASK_TARGETED only, got {route}"
        )

    query = targeted_bit_query(item.index, item.n_bits)
    source = BitVectorSource(item.bits, rng=np.random.default_rng([seed, 101]))
    response = source.respond(query)
    episode.attach_grounding(query, response, lambda_bit=lambda_bit)
    episode.final_action = str(response.content)
    episode.final_loss = zero_one_loss(episode.final_action, item.target)
    episode.validate()
    return episode


def split_items(
    items: Sequence[NoisyIndexItem], *, train_fraction: float = 0.5, seed: int = 0
) -> tuple[list[NoisyIndexItem], list[NoisyIndexItem]]:
    """Split into router-training and evaluation halves.

    The router must be scored out of sample: a router evaluated on the items it
    was fitted to would report the fit, not the policy.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    rng = np.random.default_rng([seed, 5501])
    order = rng.permutation(len(items))
    cut = int(round(len(items) * train_fraction))
    train = [items[i] for i in order[:cut]]
    evaluate = [items[i] for i in order[cut:]]
    return train, evaluate
