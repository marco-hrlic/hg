"""E1: the full bounded-query frontier, and E1b its lossy relaxation.

The improved plan (sections 4 and 5) replaces Phase 0's two endpoints with the
whole frontier. Phase 0 measured the index problem at r = 0 (generic
escalation, n bits) and at r = log2 n (a fully targeted query, 1 bit). Between
those sits a curve, and the v0.8 zero-error theorem says exactly where:

    b*(n, r) = ceil(n / 2**r)

for deterministic, fixed-length, single-shot, zero-error protocols, where the
AI may spend r bits naming what it wants and the source answers with b bits.

Why the theorem holds, in the form this module checks
-----------------------------------------------------
The r query bits partition the n candidate indices into at most 2**r fibers.
Some fiber has size s >= ceil(n / 2**r). Conditional on that query message the
source's answer must determine X_K for *every* K in the fiber, so the response
map restricted to those s coordinates has to be injective on {0,1}**s -- two
assignments differing anywhere in the fiber must get different answers, or the
index that distinguishes them is decided wrongly. Hence 2**b >= 2**s, i.e.
b >= s. Achievability is the balanced partition: fibers of size ceil(n / 2**r)
or floor(n / 2**r), and the source sends the fiber's bits verbatim.

Both halves are *run* here rather than asserted. `zero_error` simulates the
protocol and must reach accuracy 1.0 at exactly b* bits; `truncated_converse`
runs the same protocol one bit poorer and must fail. An arithmetic identity
that no code path can violate is not a regression test.

The lossy frontier (E1b)
------------------------
With a per-decision error tolerance eps, the source may compress the fiber
lossily. Three distinct claims, which the plan (section 5) is emphatic must not
be merged:

    finite converse         b >= s * (1 - h2(eps))
    asymptotic rate         b*(n, r, eps) -> n * 2**-r * (1 - h2(eps))
    achievability           whatever a constructed code actually attains

Only the first is exact at finite s: it is the rate-distortion converse for a
Bernoulli(1/2) source under Hamming distortion, applied to one fiber. The
second is a limit, and at the blocklengths reachable here (s <= 10 by
enumeration) it is not a prediction about any single cell. So this module
reports all three side by side and gates only on the converse, which a bug in
the accounting *could* violate and a correct implementation never can.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..io import git_commit, run_id, write_json, write_yaml
from ..report.figures import plot_grounding_sweep
from .channels import QuerySpec, Response, Route, log2_alphabet
from .config import GroundingConfig
from .costs import zero_one_loss
from .episodes import GroundingEpisode, GroundingRunDir, read_episodes, write_episodes

log = logging.getLogger(__name__)

BOUNDED_QUERY_TASK_VERSION = "bounded-query-v1"
TASK_NAME = "bounded_query"

# One arm per claim, and they are separate arms because they answer different
# questions: the first two bracket b*(n, r) from both sides, the third measures
# a lossy code.
POLICY_ZERO_ERROR = "zero_error"
POLICY_TRUNCATED = "truncated_converse"
POLICY_LOSSY = "lossy"
POLICIES = (POLICY_ZERO_ERROR, POLICY_TRUNCATED, POLICY_LOSSY)

# Above this fiber size the 2**s enumeration that designs and scores a lossy
# codebook stops being tractable, and the module falls back to a structured
# scheme whose distortion is known in closed form. Reported per cell so a
# reader never has to guess which was used.
DEFAULT_MAX_ENUMERABLE_FIBER = 10

CODE_LLOYD = "lloyd_search"
CODE_PREFIX = "exact_prefix"

ACTIONS = ("0", "1")


# ---------------------------------------------------------------- the theory ---
def optimal_response_bits(n_bits: int, query_bits: int) -> int:
    """b*(n, r) = ceil(n / 2**r), the zero-error frontier (plan section 4)."""
    if n_bits < 1:
        raise ValueError(f"n_bits must be >= 1, got {n_bits}")
    if query_bits < 0:
        raise ValueError(f"query_bits must be >= 0, got {query_bits}")
    return math.ceil(n_bits / (2**query_bits))


def binary_entropy(p: float) -> float:
    """h2(p) in bits, with h2(0) = h2(1) = 0."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def lossy_converse_bits(fiber_size: int, epsilon: float) -> float:
    """b >= s * (1 - h2(eps)): the finite converse, exact at every s.

    Not an asymptotic statement. The source must describe s i.i.d. fair bits to
    within average Hamming distortion eps, and no code of any blocklength beats
    the rate-distortion function on its own block.
    """
    if epsilon >= 0.5:
        return 0.0
    return float(fiber_size) * (1.0 - binary_entropy(epsilon))


def lossy_asymptotic_bits(n_bits: int, query_bits: int, epsilon: float) -> float:
    """n * 2**-r * (1 - h2(eps)): the large-fiber rate prediction.

    Distinct from `lossy_converse_bits` in using the *un-rounded* fiber size, and
    distinct from anything measurable at a single finite s. Reported so the two
    can be compared, never used as an acceptance criterion.
    """
    if epsilon >= 0.5:
        return 0.0
    return float(n_bits) * (2.0**-query_bits) * (1.0 - binary_entropy(epsilon))


# ------------------------------------------------------------- the protocol ---
def balanced_fibers(n_bits: int, query_bits: int) -> tuple[tuple[int, ...], ...]:
    """Partition {0..n-1} into 2**r contiguous fibers, as evenly as possible.

    Balanced because the response length is fixed: an unbalanced partition pays
    for its largest fiber on every item, so it can only be worse. The largest
    fiber here has exactly ceil(n / 2**r) members, which is what makes the
    constructed protocol meet the bound rather than merely respect it.
    """
    n_fibers = min(2**query_bits, n_bits)
    sizes = [
        n_bits // n_fibers + (1 if i < n_bits % n_fibers else 0)
        for i in range(n_fibers)
    ]
    fibers: list[tuple[int, ...]] = []
    start = 0
    for size in sizes:
        fibers.append(tuple(range(start, start + size)))
        start += size
    return tuple(fibers)


@dataclass(frozen=True, slots=True)
class BoundedQueryProtocol:
    """A fixed-length (r, b) protocol for the index problem.

    `response_bits` is a parameter rather than a derived constant so the same
    machinery can run the protocol one bit short of the bound -- which is how
    the converse gets measured instead of asserted.
    """

    n_bits: int
    query_bits: int
    response_bits: int

    @property
    def fibers(self) -> tuple[tuple[int, ...], ...]:
        return balanced_fibers(self.n_bits, self.query_bits)

    @property
    def max_fiber_size(self) -> int:
        return max(len(f) for f in self.fibers)

    @property
    def meets_bound(self) -> bool:
        return self.response_bits == optimal_response_bits(self.n_bits, self.query_bits)

    def fiber_of(self, index: int) -> tuple[int, tuple[int, ...]]:
        """Which fiber the AI's index falls in. This is what the query names."""
        for i, fiber in enumerate(self.fibers):
            if index in fiber:
                return i, fiber
        raise ValueError(f"index {index} out of range for n_bits={self.n_bits}")

    def query(self, index: int) -> QuerySpec:
        """The r-bit query: name the fiber, not the index."""
        fiber_id, fiber = self.fiber_of(index)
        names = tuple(f"X{i}" for i in fiber)
        return QuerySpec(
            variable_names=names,
            response_bits=float(self.response_bits),
            response_alphabet_size=2**self.response_bits,
            candidate_values=None,
            candidate_actions_affected=ACTIONS,
            natural_language_query=(
                f"Report bits {fiber[0]}..{fiber[-1]} "
                f"(fiber {fiber_id} of {len(self.fibers)})."
            ),
            metadata={
                "kind": "bounded",
                "query_bits": self.query_bits,
                "fiber_id": fiber_id,
                "fiber_size": len(fiber),
                "n_bits": self.n_bits,
            },
        )

    def respond(self, bits: Sequence[int], index: int) -> Response:
        """The source's fixed-length answer.

        Sends the first `response_bits` coordinates of the fiber verbatim. At
        the bound that is the whole fiber; one bit short it is the whole fiber
        minus its last coordinate, and the AI has to guess when its own index
        landed there. Truncation *of the answer* rather than of the query is the
        right way to run the converse: the query is unchanged, so the only thing
        removed is transmitted information.
        """
        _fiber_id, fiber = self.fiber_of(index)
        sent = [int(bits[i]) for i in fiber[: self.response_bits]]
        return Response(
            content="".join(str(b) for b in sent),
            raw_bits=float(self.response_bits),
            informative=index in fiber[: self.response_bits],
            metadata={"n_variables": len(sent), "fiber_size": len(fiber)},
        )

    def decode(self, response: Response, index: int, *, guess: int) -> str:
        """The AI's final action.

        `guess` is used only for a coordinate the truncated protocol did not
        transmit. It is supplied by the caller rather than drawn here so the
        same item gets the same guess across arms.
        """
        _fiber_id, fiber = self.fiber_of(index)
        position = fiber.index(index)
        content = str(response.content)
        if position < len(content):
            return content[position]
        return str(guess)


def converse_witness(
    encoder: Sequence[int], fiber_size: int
) -> tuple[int, int, int] | None:
    """Find an input pair and an index on which `encoder` must decide wrongly.

    `encoder` maps each of the 2**s assignments of a fiber (indexed by its
    integer value, most significant bit first) to a message. Returns
    `(x, y, i)` such that the two assignments share a message but differ at
    coordinate `i`, so the AI holding K = i receives the same answer in two
    worlds that require different actions -- or None if no such pair exists.

    This is the converse argument itself rather than a property of one
    protocol. `truncated_converse` shows that *dropping a coordinate* fails,
    which leaves open the objection that some cleverer b* - 1 bit code might
    not. The pigeonhole here closes it: whenever the message alphabet is
    smaller than 2**s, two assignments collide, and two assignments that
    collide differ somewhere.
    """
    seen: dict[int, int] = {}
    for x in range(2**fiber_size):
        message = int(encoder[x])
        if message in seen:
            y = seen[message]
            differing = x ^ y
            if differing:
                # The lowest differing bit, counted from the fiber's first
                # coordinate, is as good a witness as any.
                position = fiber_size - 1 - (differing.bit_length() - 1)
                return y, x, position
        else:
            seen[message] = x
    return None


# -------------------------------------------------------------- lossy codes ---
@lru_cache(maxsize=None)
def design_lossy_code(
    fiber_size: int, code_bits: int, *, seed: int = 0, n_restarts: int = 4
) -> tuple[tuple[int, ...], ...]:
    """A 2**b codebook over {0,1}**s, from Lloyd search with seeded restarts.

    Best-found, not provably optimal, and that asymmetry is in the right
    direction: a suboptimal codebook makes the achievable rate look *worse*, so
    it can never manufacture a violation of the converse. The returned codebook
    is therefore an upper bound on b*(s, eps) and is labelled as one everywhere
    it is reported.

    Deterministic given (fiber_size, code_bits, seed): initial centres are drawn
    from a seeded generator and Lloyd iterations are exact.
    """
    if code_bits >= fiber_size:
        # Every source word is its own codeword: zero distortion, nothing to
        # search for.
        return tuple(
            tuple(int(b) for b in np.binary_repr(v, width=fiber_size))
            for v in range(2**fiber_size)
        )
    words = _all_words(fiber_size)
    n_centres = 2**code_bits
    if n_centres == 1:
        # One codeword; by symmetry any of them costs 1/2 per coordinate.
        return (tuple([0] * fiber_size),)

    best: np.ndarray | None = None
    best_cost = math.inf
    for restart in range(max(1, n_restarts)):
        rng = np.random.default_rng([seed, fiber_size, code_bits, restart])
        centres = words[
            rng.choice(words.shape[0], size=n_centres, replace=False)
        ].astype(np.int64)
        for _ in range(32):
            # Hamming distance from every word to every centre.
            dist = _hamming_matrix(words, centres)
            assignment = np.argmin(dist, axis=1)
            cost = float(dist[np.arange(words.shape[0]), assignment].sum())
            moved = np.zeros_like(centres)
            for c in range(n_centres):
                members = words[assignment == c]
                if members.size == 0:
                    # An empty cell wastes a codeword. Re-seed it on the word
                    # currently worst served, which is the standard fix and
                    # keeps the search deterministic.
                    worst = int(np.argmax(dist[np.arange(words.shape[0]), assignment]))
                    moved[c] = words[worst]
                else:
                    moved[c] = (members.mean(axis=0) >= 0.5).astype(np.int64)
            if np.array_equal(moved, centres):
                break
            centres = moved
        dist = _hamming_matrix(words, centres)
        cost = float(dist.min(axis=1).sum())
        if cost < best_cost:
            best_cost, best = cost, centres
    assert best is not None
    return tuple(tuple(int(v) for v in row) for row in best)


@lru_cache(maxsize=None)
def _all_words(fiber_size: int) -> np.ndarray:
    return np.array(
        [
            [int(b) for b in np.binary_repr(v, width=fiber_size)]
            for v in range(2**fiber_size)
        ],
        dtype=np.int64,
    )


def _hamming_matrix(words: np.ndarray, centres: np.ndarray) -> np.ndarray:
    # d(x, c) = |x| + |c| - 2 x.c for binary vectors.
    return (
        words.sum(axis=1, keepdims=True)
        + centres.sum(axis=1)[None, :]
        - 2 * (words @ centres.T)
    )


def code_distortion(fiber_size: int, code_bits: int, *, seed: int = 0) -> float:
    """Exact expected per-coordinate Hamming distortion of the designed code.

    Exact rather than sampled: the source is uniform on {0,1}**s, so averaging
    over all 2**s words *is* the expectation. This is the probability the AI
    decides wrongly, since its index is uniform on the fiber.
    """
    if code_bits >= fiber_size:
        return 0.0
    if code_bits == 0:
        return 0.5
    words = _all_words(fiber_size)
    centres = np.asarray(design_lossy_code(fiber_size, code_bits, seed=seed))
    dist = _hamming_matrix(words, centres).min(axis=1)
    return float(dist.mean() / fiber_size)


def prefix_distortion(fiber_size: int, code_bits: int) -> float:
    """Distortion of the structured fallback: send b coordinates, guess the rest.

    Used where 2**s enumeration is out of reach. Its distortion is closed-form,
    so a cell using it is still exact -- it is just a weaker code, which shows
    up as a larger gap to the converse rather than as a softer number.
    """
    if code_bits >= fiber_size:
        return 0.0
    return float(fiber_size - code_bits) / (2.0 * fiber_size)


def minimal_lossy_bits(
    fiber_size: int,
    epsilon: float,
    *,
    seed: int = 0,
    max_enumerable: int = DEFAULT_MAX_ENUMERABLE_FIBER,
) -> tuple[int, float, str]:
    """Smallest integer b whose code meets the tolerance. Returns (b, D, method).

    An upper bound on b*(s, eps) at fixed length, because the codebook is
    best-found rather than proven optimal.
    """
    method = CODE_LLOYD if fiber_size <= max_enumerable else CODE_PREFIX
    distort = (
        (lambda b: code_distortion(fiber_size, b, seed=seed))
        if method == CODE_LLOYD
        else (lambda b: prefix_distortion(fiber_size, b))
    )
    for b in range(0, fiber_size + 1):
        d = distort(b)
        if d <= epsilon + 1e-12:
            return b, d, method
    return fiber_size, 0.0, method


class LossySource:
    """A source that answers a bounded query with a lossy description.

    Same interface as the zero-error protocol's source, so the episode, the cost
    accounting and the decode are unchanged and the only difference between the
    arms is what was transmitted.
    """

    def __init__(
        self,
        protocol: BoundedQueryProtocol,
        *,
        method: str,
        seed: int = 0,
    ):
        self.protocol = protocol
        self.method = method
        self.seed = int(seed)

    def reconstruct(self, bits: Sequence[int], index: int) -> tuple[str, int]:
        """Returns (the AI's decoded action, the true value) for this item."""
        _fiber_id, fiber = self.protocol.fiber_of(index)
        s = len(fiber)
        b = self.protocol.response_bits
        truth = int(bits[index])
        position = fiber.index(index)
        if b >= s:
            return str(truth), truth
        if self.method == CODE_PREFIX:
            if position < b:
                return str(truth), truth
            # Not transmitted: guess, deterministically keyed on the item.
            rng = np.random.default_rng([self.seed, index, *bits])
            return str(int(rng.integers(0, 2))), truth
        word = np.asarray([int(bits[i]) for i in fiber], dtype=np.int64)
        centres = np.asarray(design_lossy_code(s, b, seed=self.seed))
        chosen = centres[int(np.argmin(np.abs(centres - word).sum(axis=1)))]
        return str(int(chosen[position])), truth


# ------------------------------------------------------------------ running ---
@dataclass(frozen=True, slots=True)
class FrontierCell:
    """One point of the frontier: a task size, a query budget, a tolerance."""

    n_bits: int
    query_bits: int
    epsilon: float | None = None

    @property
    def optimal_bits(self) -> int:
        return optimal_response_bits(self.n_bits, self.query_bits)


def query_budgets(n_bits: int) -> list[int]:
    """r = 0, 1, ..., ceil(log2 n) -- generic escalation through to targeted."""
    return list(range(0, math.ceil(math.log2(n_bits)) + 1))


def _items(n_bits: int, n_items: int, seed: int) -> list[tuple[str, int, tuple[int, ...]]]:
    """(item_id, K, M_H) triples. Seeded on n so one cell reruns independently."""
    rng = np.random.default_rng([seed, 4211, n_bits])
    bits = rng.integers(0, 2, size=(n_items, n_bits))
    indices = rng.integers(0, n_bits, size=n_items)
    return [
        (f"bq-n{n_bits}-{i:06d}", int(indices[i]), tuple(int(b) for b in bits[i]))
        for i in range(n_items)
    ]


def _episode(
    *,
    item_id: str,
    policy: str,
    protocol: BoundedQueryProtocol,
    index: int,
    bits: Sequence[int],
    action_before: str,
    action_after: str,
    query: QuerySpec,
    response: Response,
    lambda_bit: float,
    epsilon: float | None,
    method: str | None,
) -> GroundingEpisode:
    target = str(int(bits[index]))
    episode = GroundingEpisode(
        item_id=item_id,
        task=TASK_NAME,
        policy=policy,
        route=Route.ASK_TARGETED.value,
        ai_private_state={"index": index, "n_bits": protocol.n_bits},
        hidden_reference_state={
            "bits": "".join(str(b) for b in bits),
            "decision_variable": target,
        },
        decision_state={
            "exact": True,
            "candidate_actions": list(ACTIONS),
            # H(J | M_A) = 1 bit: the index says which bit decides, never which
            # value it takes. Unchanged by the query budget.
            "residual_deficit_bits": 1.0,
            "expected_loss_now": 0.5,
            "internal_reducibility": 0.0,
            "external_value": 0.5,
        },
        initial_action=action_before,
        initial_loss=zero_one_loss(action_before, target),
        final_action=action_after,
        final_loss=zero_one_loss(action_after, target),
        metadata={
            "task_version": BOUNDED_QUERY_TASK_VERSION,
            "n_bits": protocol.n_bits,
            "query_bits": protocol.query_bits,
            "response_bits": protocol.response_bits,
            "optimal_response_bits": optimal_response_bits(
                protocol.n_bits, protocol.query_bits
            ),
            "fiber_size": protocol.max_fiber_size,
            "epsilon": epsilon,
            "code_method": method,
            "lambda_bit": lambda_bit,
        },
    )
    episode.attach_grounding(query, response, lambda_bit=lambda_bit)
    episode.validate()
    return episode


def run_cell(
    cell: FrontierCell,
    policy: str,
    *,
    n_items: int,
    seed: int,
    lambda_bit: float,
    max_enumerable: int = DEFAULT_MAX_ENUMERABLE_FIBER,
) -> tuple[list[GroundingEpisode], dict[str, Any]]:
    """Run one frontier cell under one arm."""
    n, r = cell.n_bits, cell.query_bits
    optimal = cell.optimal_bits
    info: dict[str, Any] = {}

    if policy == POLICY_ZERO_ERROR:
        response_bits, method = optimal, None
    elif policy == POLICY_TRUNCATED:
        response_bits, method = max(optimal - 1, 0), None
    else:
        fiber = optimal  # the largest fiber is what a fixed-length code covers
        assert cell.epsilon is not None
        response_bits, distortion, method = minimal_lossy_bits(
            fiber, cell.epsilon, seed=seed, max_enumerable=max_enumerable
        )
        info = {
            "designed_distortion": distortion,
            "converse_bits": lossy_converse_bits(fiber, cell.epsilon),
            "asymptotic_bits": lossy_asymptotic_bits(n, r, cell.epsilon),
            "code_method": method,
        }

    protocol = BoundedQueryProtocol(n_bits=n, query_bits=r, response_bits=response_bits)
    source = LossySource(protocol, method=method or CODE_PREFIX, seed=seed)
    episodes: list[GroundingEpisode] = []
    for item_id, index, bits in _items(n, n_items, seed):
        # The act-now guess is keyed on the item so every arm sees the same one
        # and D_before is identical across the frontier.
        guess_rng = np.random.default_rng([seed, 99, n, index, int(item_id[-6:])])
        guess = int(guess_rng.integers(0, 2))
        query = protocol.query(index)
        if policy == POLICY_LOSSY:
            action_after, _truth = source.reconstruct(bits, index)
            response = Response(
                content=action_after,
                raw_bits=float(response_bits),
                informative=response_bits > 0,
                metadata={"n_variables": protocol.max_fiber_size},
            )
        else:
            response = protocol.respond(bits, index)
            action_after = protocol.decode(response, index, guess=guess)
        episodes.append(
            _episode(
                item_id=item_id,
                policy=policy,
                protocol=protocol,
                index=index,
                bits=bits,
                action_before=str(guess),
                action_after=action_after,
                query=query,
                response=response,
                lambda_bit=lambda_bit,
                epsilon=cell.epsilon,
                method=method,
            )
        )
    return episodes, info


def truncated_failure_probability(fiber_size: int, response_bits: int) -> float:
    """P(the truncated protocol decides wrongly), analytically.

    The dropped coordinates are decided by a coin flip, and the AI's index is
    uniform on the fiber, so the failure rate is (s - b) / 2s. It is small for a
    large fiber -- at n = 64, r = 0 one bit short of the bound fails once in
    128 items -- which is exactly why the converse check needs its own item
    count rather than the sweep's.
    """
    if response_bits >= fiber_size:
        return 0.0
    return float(fiber_size - response_bits) / (2.0 * fiber_size)


def items_for_converse(
    fiber_size: int, response_bits: int, *, min_expected_failures: int
) -> int:
    """How many items make the truncated arm's failure visible.

    Sized on the analytic failure rate so every cell of the frontier carries the
    same statistical power. Without this the converse check silently weakens as
    n grows: a run at 400 items has a 4% chance of seeing no failure at all at
    n = 64, r = 0, and would then report the bound as not tight.
    """
    p = truncated_failure_probability(fiber_size, response_bits)
    if p <= 0.0:
        return 1
    return int(math.ceil(min_expected_failures / p))


def run_sweep(cfg: GroundingConfig) -> tuple[list[GroundingEpisode], dict[str, Any]]:
    """Every (n, r) cell zero-error and truncated; every (n, r, eps) cell lossy."""
    spec = cfg.frontier_task
    if spec is None:
        raise ValueError("E1 requires a `task` section with kind: bounded_query")

    episodes: list[GroundingEpisode] = []
    design: list[dict[str, Any]] = []
    for n in spec.n_bits:
        for r in query_budgets(n):
            cell = FrontierCell(n_bits=n, query_bits=r)
            optimal = cell.optimal_bits
            # The zero-error arm is deterministic, so its item count is only
            # about exercising every fiber. The truncated arm's failure is
            # probabilistic and rare for a large fiber, so it gets whatever
            # count makes that failure visible.
            n_truncated = max(
                spec.n_items,
                items_for_converse(
                    optimal, optimal - 1,
                    min_expected_failures=cfg.gate.min_expected_failures,
                ),
            )
            for policy, n_items in (
                (POLICY_ZERO_ERROR, spec.n_items),
                (POLICY_TRUNCATED, n_truncated),
            ):
                if policy not in cfg.policies:
                    continue
                eps_, _info = run_cell(
                    cell, policy,
                    n_items=n_items, seed=cfg.seed, lambda_bit=cfg.lambda_bit,
                    max_enumerable=spec.max_enumerable_fiber,
                )
                episodes.extend(eps_)
            if POLICY_LOSSY not in cfg.policies:
                continue
            for epsilon in spec.epsilons:
                if epsilon <= 0.0:
                    # eps = 0 is the zero-error arm; running it again as a
                    # "lossy" cell would report the same protocol twice under
                    # two names.
                    continue
                lossy_cell = FrontierCell(n_bits=n, query_bits=r, epsilon=epsilon)
                eps_, info = run_cell(
                    lossy_cell, POLICY_LOSSY,
                    n_items=spec.n_items_lossy, seed=cfg.seed,
                    lambda_bit=cfg.lambda_bit,
                    max_enumerable=spec.max_enumerable_fiber,
                )
                episodes.extend(eps_)
                design.append({"n_bits": n, "query_bits": r, "epsilon": epsilon, **info})
        log.info("n=%d: %d query budgets", n, len(query_budgets(n)))

    diagnostics = {
        "n_bits_sweep": list(spec.n_bits),
        "epsilons": list(spec.epsilons),
        "n_items": spec.n_items,
        "n_items_lossy": spec.n_items_lossy,
        "max_enumerable_fiber": spec.max_enumerable_fiber,
        "lossy_code_design": design,
    }
    return episodes, diagnostics


# ----------------------------------------------------------------- analysis ---
def _cells(
    episodes: Sequence[GroundingEpisode], policy: str
) -> dict[tuple[int, int, float | None], list[GroundingEpisode]]:
    out: dict[tuple[int, int, float | None], list[GroundingEpisode]] = {}
    for e in episodes:
        if e.policy != policy:
            continue
        md = e.metadata
        key = (int(md["n_bits"]), int(md["query_bits"]), md.get("epsilon"))
        out.setdefault(key, []).append(e)
    return out


def _summarize(episodes: Sequence[GroundingEpisode]) -> dict[str, Any]:
    accuracy = float(np.mean([1.0 - e.final_loss for e in episodes]))
    bits = float(np.mean([e.external_raw_bits for e in episodes]))
    return {"n_items": len(episodes), "accuracy": accuracy, "mean_response_bits": bits}


def frontier_table(episodes: Sequence[GroundingEpisode]) -> list[dict[str, Any]]:
    """Predicted vs observed b*(n, r), with the one-bit-poorer control beside it.

    The two arms are reported on one row because they are one claim: b* bits
    suffice *and* b* - 1 do not. Either half alone is compatible with the bound
    being wrong in the other direction.
    """
    zero = _cells(episodes, POLICY_ZERO_ERROR)
    trunc = _cells(episodes, POLICY_TRUNCATED)
    rows: list[dict[str, Any]] = []
    for (n, r, _eps), eps_list in sorted(zero.items()):
        predicted = optimal_response_bits(n, r)
        summary = _summarize(eps_list)
        control = trunc.get((n, r, None))
        control_summary = _summarize(control) if control else None
        rows.append(
            {
                "n_bits": n,
                "query_bits": r,
                "fiber_size": int(eps_list[0].metadata["fiber_size"]),
                "predicted_response_bits": predicted,
                "observed_response_bits": summary["mean_response_bits"],
                "accuracy": summary["accuracy"],
                "matches_theorem": (
                    summary["mean_response_bits"] == float(predicted)
                    and summary["accuracy"] == 1.0
                ),
                "truncated_response_bits": (
                    control_summary["mean_response_bits"] if control_summary else None
                ),
                "truncated_accuracy": (
                    control_summary["accuracy"] if control_summary else None
                ),
                # A protocol one bit short of the bound must fail somewhere. If
                # it does not, the bound is not tight here and the theorem
                # regression has caught something real.
                "converse_is_tight": (
                    bool(control_summary["accuracy"] < 1.0)
                    if control_summary
                    else None
                ),
                # What the truncated arm's failure rate should be, and how many
                # failures its item count buys. Reported so a "tight: yes" can
                # be read as evidence rather than as a coin flip that landed.
                "truncated_failure_probability": (
                    truncated_failure_probability(predicted, predicted - 1)
                ),
                "expected_truncated_failures": (
                    control_summary["n_items"]
                    * truncated_failure_probability(predicted, predicted - 1)
                    if control_summary
                    else None
                ),
                "n_items": summary["n_items"],
            }
        )
    return rows


def lossy_table(
    episodes: Sequence[GroundingEpisode], design: Sequence[Mapping[str, Any]] = ()
) -> list[dict[str, Any]]:
    """The three lossy claims side by side, per (n, r, eps) cell."""
    by_key = {
        (int(d["n_bits"]), int(d["query_bits"]), float(d["epsilon"])): d
        for d in design
    }
    rows: list[dict[str, Any]] = []
    for (n, r, eps), group in sorted(
        _cells(episodes, POLICY_LOSSY).items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0.0)
    ):
        assert eps is not None
        summary = _summarize(group)
        fiber = int(group[0].metadata["fiber_size"])
        bits = summary["mean_response_bits"]
        converse = lossy_converse_bits(fiber, eps)
        rows.append(
            {
                "n_bits": n,
                "query_bits": r,
                "epsilon": eps,
                "fiber_size": fiber,
                "achievable_response_bits": bits,
                "converse_bits": converse,
                "asymptotic_bits": lossy_asymptotic_bits(n, r, eps),
                "zero_error_bits": float(optimal_response_bits(n, r)),
                "realized_error": 1.0 - summary["accuracy"],
                "designed_distortion": by_key.get((n, r, eps), {}).get(
                    "designed_distortion"
                ),
                "code_method": group[0].metadata.get("code_method"),
                # The only hard lossy criterion. An achievable rate below the
                # converse is a bug in the accounting, not a discovery.
                "respects_converse": bool(bits >= converse - 1e-9),
                # How far the best-found finite-length code sits above the
                # converse, per source symbol. Reported, never gated: at these
                # blocklengths the asymptotic rate is not a prediction about a
                # single cell.
                "redundancy_per_symbol": (bits - converse) / fiber if fiber else None,
                "n_items": summary["n_items"],
            }
        )
    return rows


def build_analysis(
    cfg: GroundingConfig,
    episodes: Sequence[GroundingEpisode],
    design: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    frontier = frontier_table(episodes)
    lossy = lossy_table(episodes, design)
    return {
        "experiment": cfg.name,
        "phase": "e1",
        "task": cfg.task_kind,
        "task_version": BOUNDED_QUERY_TASK_VERSION,
        "n_episodes": len(episodes),
        "frontier": frontier,
        "lossy": lossy,
        "gate": evaluate_gate(cfg, frontier, lossy),
    }


def evaluate_gate(
    cfg: GroundingConfig,
    frontier: Sequence[Mapping[str, Any]],
    lossy: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Gate H0 (plan section 18): the theorem-frontier regression.

    Three checks, and the third is deliberately weaker than the first two. The
    zero-error frontier is an exact identity a run either reproduces or does
    not. The lossy side has no finite-length equality to check, so what is
    gated there is the converse -- the one lossy statement that is exact at
    every blocklength.
    """
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})
        if not passed:
            failures.append(f"{name}: {detail}")

    for row in frontier:
        label = f"n={row['n_bits']},r={row['query_bits']}"
        record(
            f"zero_error_frontier[{label}]",
            bool(row["matches_theorem"]),
            f"observed {row['observed_response_bits']} bits at accuracy "
            f"{row['accuracy']:.4f}; b*(n, r) = {row['predicted_response_bits']} "
            "at accuracy 1.0",
        )
        if row["converse_is_tight"] is not None:
            record(
                f"converse_is_tight[{label}]",
                bool(row["converse_is_tight"]),
                f"the same protocol with {row['truncated_response_bits']} bits "
                f"scored {row['truncated_accuracy']:.4f} over "
                f"{row['expected_truncated_failures']:.1f} expected failures; "
                "anything but a drop below 1.0 means b* is not the minimum here",
            )

    tol = cfg.gate.epsilon_tolerance
    for row in lossy:
        label = f"n={row['n_bits']},r={row['query_bits']},eps={row['epsilon']}"
        record(
            f"lossy_respects_converse[{label}]",
            bool(row["respects_converse"]),
            f"achievable {row['achievable_response_bits']:.2f} bits vs converse "
            f"{row['converse_bits']:.2f}",
        )
        designed = row["designed_distortion"]
        if designed is not None:
            # The exact expected distortion of the designed code, computed over
            # the whole source alphabet or in closed form. This is the criterion
            # that has no sampling error in it.
            record(
                f"lossy_design_meets_tolerance[{label}]",
                designed <= row["epsilon"] + 1e-12,
                f"designed distortion {designed:.4f} against tolerance "
                f"{row['epsilon']:.2f}",
            )
        record(
            f"lossy_meets_tolerance[{label}]",
            row["realized_error"] <= row["epsilon"] + tol,
            f"realized error {row['realized_error']:.4f} against tolerance "
            f"{row['epsilon']:.2f} (+{tol} for the sampling error of "
            f"{row['n_items']} items)",
        )

    passed = not failures
    return {
        "gate": "H0",
        "title": "Theoretical frontier regression: b*(n, r) = ceil(n / 2**r)",
        "passed": passed,
        "n_checks": len(checks),
        "failures": failures,
        "checks": checks,
        "verdict": (
            "The full zero-error bounded-query frontier reproduces "
            "b*(n, r) = ceil(n / 2**r) at every (n, r), and one bit less fails "
            "at every one of them. The lossy cells respect the finite converse "
            "b >= s(1 - h2(eps)) and meet their tolerances. Note what this is "
            "not: no model is involved, and the lossy cells demonstrate "
            "achievability of a constructed code, not finite-length optimality."
            if passed
            else "Gate H0 FAILED. Plan section 18: this is a stop. The "
            "measurement infrastructure disagrees with a theorem, so nothing "
            "measured on top of it can be interpreted."
        ),
    }


# ------------------------------------------------------------------- writing ---
def write_run(
    cfg: GroundingConfig,
    episodes: Sequence[GroundingEpisode],
    diagnostics: Mapping[str, Any],
    *,
    output_root: Path | None = None,
    tag: str | None = None,
) -> GroundingRunDir:
    root = Path(output_root or cfg.output_root) / run_id(tag or cfg.name)
    if root.exists():
        raise FileExistsError(f"run directory already exists: {root}")
    run = GroundingRunDir(root).create()
    write_yaml(run.config_path, cfg.resolved_dict())

    by_policy: dict[str, list[GroundingEpisode]] = {}
    for e in episodes:
        by_policy.setdefault(e.policy, []).append(e)
    counts = {
        policy: write_episodes(run.episodes(policy), eps)
        for policy, eps in sorted(by_policy.items())
    }
    write_json(run.root / "diagnostics.json", dict(diagnostics))
    write_json(
        run.manifest_path,
        {
            "run_id": root.name,
            "experiment": cfg.name,
            "phase": "e1",
            "task": cfg.task_kind,
            "task_version": BOUNDED_QUERY_TASK_VERSION,
            "seed": cfg.seed,
            "policies": list(cfg.policies),
            "episode_counts": counts,
            "config_path": str(cfg.source_path) if cfg.source_path else None,
            "model_checkpoint": None,
            "backend": "none (analytic protocol simulator)",
            "git_commit": git_commit(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": _environment(),
        },
    )
    return run


def _environment() -> dict[str, Any]:
    from .phase0 import environment

    return environment()


def read_run_episodes(run: GroundingRunDir) -> list[GroundingEpisode]:
    policies = run.available_policies()
    if not policies:
        raise FileNotFoundError(f"no episode files under {run.root}/episodes")
    episodes: list[GroundingEpisode] = []
    for policy in policies:
        episodes.extend(read_episodes(run.episodes(policy), strict=True))
    return episodes


# ------------------------------------------------------------------ figures ---
def write_figures(run: GroundingRunDir, analysis: Mapping[str, Any]) -> dict[str, bool]:
    frontier = analysis["frontier"]
    series: dict[str, list[tuple[float, float]]] = {}
    for row in frontier:
        series.setdefault(f"n={row['n_bits']}", []).append(
            (float(row["query_bits"]), float(row["observed_response_bits"]))
        )
    wrote = {
        "frontier": plot_grounding_sweep(
            run.figure("e1_zero_error_frontier.png"),
            {k: sorted(v) for k, v in series.items()},
            xlabel="r = query bits the AI may spend",
            ylabel="b* = response bits the source must send",
            title="E1: the zero-error bounded-query frontier, b*(n, r) = ceil(n / 2^r)",
            # Log on y so the halving at each extra query bit is a straight
            # line and a departure from the theorem is visible as a kink.
            logy=True,
        )
    }
    lossy = analysis.get("lossy") or []
    if lossy:
        by_eps: dict[str, list[tuple[float, float]]] = {}
        biggest = max(row["n_bits"] for row in lossy)
        for row in lossy:
            if row["n_bits"] != biggest:
                continue
            by_eps.setdefault(f"eps={row['epsilon']:.2f}", []).append(
                (float(row["query_bits"]), float(row["achievable_response_bits"]))
            )
        wrote["lossy"] = plot_grounding_sweep(
            run.figure("e1b_lossy_frontier.png"),
            {k: sorted(v) for k, v in by_eps.items()},
            xlabel="r = query bits the AI may spend",
            ylabel="achievable response bits",
            title=f"E1b: lossy frontier at n = {biggest} (best-found codes)",
            logy=True,
        )
    return wrote


# ------------------------------------------------------------------- report ---
def render_report(
    cfg: GroundingConfig, analysis: Mapping[str, Any], run: GroundingRunDir
) -> str:
    gate = analysis["gate"]
    lines: list[str] = [
        f"# {analysis['experiment']} — E1 (gate H0)",
        "",
        f"Task `{analysis['task']}` version `{analysis['task_version']}`, "
        f"{analysis['n_episodes']} episodes, seed {cfg.seed}.",
        "",
        "**No model was involved.** This is the theorem-regression experiment: "
        "the improved plan's section 4 replaces Phase 0's two endpoints "
        "(r = 0 and r = log2 n) with the whole frontier between them.",
        "",
        "## The zero-error frontier",
        "",
        "For deterministic fixed-length single-shot zero-error protocols, "
        "`b*(n, r) = ceil(n / 2^r)`. Each row runs the protocol at `b*` bits "
        "and again at `b* - 1`: the bound is only established if the first "
        "reaches accuracy 1 and the second does not.",
        "",
        "| n | r | fiber | b* predicted | b observed | accuracy | at b*−1 | accuracy | tight |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    def _cell(value: Any, spec: str = ".2f") -> str:
        return "—" if value is None else format(value, spec)

    for row in analysis["frontier"]:
        tight = row["converse_is_tight"]
        lines.append(
            f"| {row['n_bits']} | {row['query_bits']} | {row['fiber_size']} | "
            f"{row['predicted_response_bits']} | "
            f"{row['observed_response_bits']:.2f} | {row['accuracy']:.4f} | "
            f"{_cell(row['truncated_response_bits'])} | "
            f"{_cell(row['truncated_accuracy'], '.4f')} | "
            + ("—" if tight is None else ("yes" if tight else "**no**"))
            + " |"
        )

    lossy = analysis.get("lossy") or []
    if lossy:
        lines += [
            "",
            "## The lossy frontier (E1b)",
            "",
            "Three claims kept apart, per plan section 5. `converse` is exact at "
            "every blocklength; `asymptotic` is a limit and is **not** a "
            "prediction about any single cell here; `achievable` is what a "
            "constructed code attained, and is an upper bound on `b*` because "
            "the codebook is best-found rather than proven optimal.",
            "",
            "| n | r | ε | fiber | achievable | converse | asymptotic | zero-error | realized err | code | ≥ converse |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|:---:|",
        ]
        for row in lossy:
            lines.append(
                f"| {row['n_bits']} | {row['query_bits']} | {row['epsilon']:.2f} | "
                f"{row['fiber_size']} | {row['achievable_response_bits']:.2f} | "
                f"{row['converse_bits']:.2f} | {row['asymptotic_bits']:.2f} | "
                f"{row['zero_error_bits']:.0f} | {row['realized_error']:.4f} | "
                f"`{row['code_method']}` | "
                f"{'yes' if row['respects_converse'] else '**NO**'} |"
            )

    lines += [
        "",
        "## Gate H0",
        "",
        f"**{'PASSED' if gate['passed'] else 'FAILED'}** — {gate['title']} "
        f"({gate['n_checks']} checks).",
        "",
        gate["verdict"],
    ]
    if gate["failures"]:
        lines += ["", "Failures:", ""] + [f"- {f}" for f in gate["failures"]]
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python experiments/hg_v2/run_e1.py --config "
        + (str(cfg.source_path) if cfg.source_path else "<config>"),
        f"uv run python experiments/hg_v2/analyze.py --run-dir {run.root}",
        "```",
        "",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------- entry points ---
def analyze_run_dir(
    run_root: str | Path, config_path: str | Path | None = None
) -> dict[str, Any]:
    from ..io import read_json
    from .config import load_grounding_config

    run = GroundingRunDir(run_root)
    cfg = load_grounding_config(config_path or run.config_path)
    episodes = read_run_episodes(run)
    design_path = run.root / "diagnostics.json"
    design = []
    if design_path.exists():
        design = (read_json(design_path) or {}).get("lossy_code_design") or []
    analysis = build_analysis(cfg, episodes, design)
    write_json(run.analysis_path, analysis)
    write_figures(run, analysis)
    run.report_path.write_text(render_report(cfg, analysis, run), encoding="utf-8")
    return analysis


def run_e1(
    cfg: GroundingConfig,
    *,
    output_root: Path | None = None,
    tag: str | None = None,
) -> tuple[GroundingRunDir, dict[str, Any], list[str]]:
    """Run the frontier, save it, and analyse it from disk."""
    episodes, diagnostics = run_sweep(cfg)
    run = write_run(cfg, episodes, diagnostics, output_root=output_root, tag=tag)
    live = build_analysis(cfg, episodes, diagnostics["lossy_code_design"])
    from_disk = analyze_run_dir(run.root)
    return run, from_disk, compare_analyses(live, from_disk)


def compare_analyses(
    live: Mapping[str, Any], from_disk: Mapping[str, Any]
) -> list[str]:
    problems: list[str] = []
    for table in ("frontier", "lossy"):
        a = {(r["n_bits"], r["query_bits"], r.get("epsilon")): r for r in live[table]}
        b = {(r["n_bits"], r["query_bits"], r.get("epsilon")): r for r in from_disk[table]}
        if set(a) != set(b):
            problems.append(f"{table}: cells differ across the round-trip")
            continue
        for key, row in a.items():
            for field, value in row.items():
                if b[key][field] != value:
                    problems.append(
                        f"{table}{key}: {field} = {value!r} live but "
                        f"{b[key][field]!r} recomputed from saved episodes"
                    )
    if live["gate"]["passed"] != from_disk["gate"]["passed"]:
        problems.append("the gate verdict changed across the round-trip")
    return problems
