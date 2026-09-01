"""E1: the bounded-query frontier as a theorem-regression suite.

These tests are the reason the improved plan calls E1 a regression rather than
an experiment. `b*(n, r) = ceil(n / 2**r)` is an identity, and an identity is
only worth re-checking if the *protocol* is checked against it -- so the
zero-error cells are verified exhaustively over every source state and every
index, not by re-deriving the arithmetic.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from hg.grounding.bounded_query import (
    CODE_LLOYD,
    CODE_PREFIX,
    POLICY_TRUNCATED,
    POLICY_ZERO_ERROR,
    BoundedQueryProtocol,
    FrontierCell,
    balanced_fibers,
    binary_entropy,
    build_analysis,
    code_distortion,
    frontier_table,
    lossy_asymptotic_bits,
    lossy_converse_bits,
    minimal_lossy_bits,
    optimal_response_bits,
    prefix_distortion,
    converse_witness,
    query_budgets,
    run_cell,
    truncated_failure_probability,
)
from hg.grounding.config import load_grounding_config
from hg.grounding.episodes import read_episodes, write_episodes


# --------------------------------------------------------------- the theorem ---
def test_frontier_matches_the_plan_table_at_n_64():
    """The seven rows of improved plan section 4, verbatim."""
    assert [optimal_response_bits(64, r) for r in range(7)] == [64, 32, 16, 8, 4, 2, 1]


@pytest.mark.parametrize("n", [8, 16, 32, 64])
def test_frontier_endpoints_reproduce_phase_0(n):
    """r = 0 is generic escalation (n bits); r = log2 n is targeted (1 bit).

    Phase 0 measured exactly these two points, so the new frontier has to
    contain the old result rather than replace it.
    """
    assert optimal_response_bits(n, 0) == n
    assert optimal_response_bits(n, int(math.log2(n))) == 1


@pytest.mark.parametrize("n,r", [(8, 1), (10, 2), (12, 3), (64, 4), (7, 1)])
def test_fibers_partition_the_index_set_and_meet_the_bound(n, r):
    fibers = balanced_fibers(n, r)
    covered = sorted(itertools.chain.from_iterable(fibers))
    assert covered == list(range(n)), "fibers must partition {0..n-1}"
    assert len({i for f in fibers for i in f}) == n, "fibers must be disjoint"
    assert len(fibers) <= 2**r
    # Achievability: the largest fiber is exactly b*, so sending it verbatim
    # meets the bound rather than merely respecting it.
    assert max(len(f) for f in fibers) == optimal_response_bits(n, r)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_zero_error_protocol_is_exhaustively_correct_at_the_bound(n):
    """Every source state, every index: b* bits decide correctly."""
    for r in query_budgets(n):
        protocol = BoundedQueryProtocol(n, r, optimal_response_bits(n, r))
        for bits in itertools.product((0, 1), repeat=n):
            for index in range(n):
                response = protocol.respond(bits, index)
                # `guess` must never be consulted at the bound; a wrong guess
                # is passed deliberately so a decode that fell back to it would
                # show up as an error rather than as a coin flip.
                decoded = protocol.decode(
                    response, index, guess=1 - bits[index]
                )
                assert decoded == str(bits[index])
                assert response.raw_bits == float(optimal_response_bits(n, r))


@pytest.mark.parametrize("n", [4, 6, 8])
def test_one_bit_short_of_the_bound_always_has_a_failing_input(n):
    """The converse, exhaustively: b* - 1 bits cannot be zero-error.

    This is the half that makes the frontier a bound rather than a construction
    that happens to work.
    """
    for r in query_budgets(n):
        optimal = optimal_response_bits(n, r)
        if optimal == 0:
            continue
        protocol = BoundedQueryProtocol(n, r, optimal - 1)
        failures = [
            (bits, index)
            for bits in itertools.product((0, 1), repeat=n)
            for index in range(n)
            if protocol.decode(
                protocol.respond(bits, index), index, guess=1 - bits[index]
            )
            != str(bits[index])
        ]
        assert failures, f"b* - 1 = {optimal - 1} bits decided everything at n={n}, r={r}"


def test_truncated_failure_probability_matches_a_simulation():
    """The analytic rate the converse arm's item count is sized from."""
    n, r = 16, 1
    optimal = optimal_response_bits(n, r)
    protocol = BoundedQueryProtocol(n, r, optimal - 1)
    rng = np.random.default_rng(0)
    misses = 0
    trials = 20000
    for _ in range(trials):
        bits = rng.integers(0, 2, size=n)
        index = int(rng.integers(0, n))
        guess = int(rng.integers(0, 2))
        decoded = protocol.decode(protocol.respond(bits, index), index, guess=guess)
        misses += decoded != str(int(bits[index]))
    expected = truncated_failure_probability(optimal, optimal - 1)
    assert abs(misses / trials - expected) < 0.01


# ------------------------------------------------------------------- lossy ---
def test_binary_entropy_endpoints():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.11) == pytest.approx(0.4999, abs=1e-3)


def test_converse_and_asymptotic_agree_when_the_fiber_divides_evenly():
    """The two lossy quantities differ only in rounding the fiber size.

    Worth pinning: they are separate claims, and the difference between them is
    exactly `ceil` -- so when n / 2**r is an integer they coincide, and when it
    is not, the converse is the one that applies to the protocol actually run.
    """
    assert lossy_converse_bits(8, 0.1) == pytest.approx(lossy_asymptotic_bits(64, 3, 0.1))
    # n = 10, r = 2 gives fibers of size 3, not 2.5.
    assert lossy_converse_bits(optimal_response_bits(10, 2), 0.1) > lossy_asymptotic_bits(
        10, 2, 0.1
    )


@pytest.mark.parametrize("s", [2, 4, 6, 8])
def test_code_distortion_is_monotone_and_hits_its_endpoints(s):
    distortions = [code_distortion(s, b) for b in range(s + 1)]
    assert distortions[0] == 0.5, "one codeword decides a fair bit at chance"
    assert distortions[-1] == 0.0, "a codeword per source word is lossless"
    assert all(
        later <= earlier + 1e-12
        for earlier, later in zip(distortions, distortions[1:])
    ), f"distortion must not increase with more bits: {distortions}"


@pytest.mark.parametrize("s", [4, 6, 8])
def test_lloyd_search_is_at_least_as_good_as_the_structured_fallback(s):
    """The designed codebook must beat 'send some coordinates, guess the rest'.

    Not a formality: the fallback is what large fibers fall back to, and if the
    search could not beat it, the enumerable cells would be reporting a worse
    achievable rate under a more impressive name.
    """
    for b in range(1, s):
        assert code_distortion(s, b) <= prefix_distortion(s, b) + 1e-12


@pytest.mark.parametrize("s", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2, 0.3, 0.4])
def test_achievable_bits_never_beat_the_finite_converse(s, eps):
    """b >= s(1 - h2(eps)) is exact at every blocklength.

    A violation here is an accounting bug, never a result -- which is why this
    is the one lossy statement gate H0 gates on.
    """
    bits, distortion, method = minimal_lossy_bits(s, eps)
    assert method == CODE_LLOYD
    assert distortion <= eps + 1e-12
    assert bits >= lossy_converse_bits(s, eps) - 1e-9
    if bits > 0:
        # Minimality of the search result: one bit fewer must miss the target.
        assert code_distortion(s, bits - 1) > eps


def test_large_fibers_fall_back_to_the_structured_scheme():
    bits, distortion, method = minimal_lossy_bits(16, 0.2, max_enumerable=10)
    assert method == CODE_PREFIX
    assert distortion == pytest.approx(prefix_distortion(16, bits))
    assert distortion <= 0.2
    assert bits >= lossy_converse_bits(16, 0.2)


# ------------------------------------------------------------ the run itself ---
def test_run_cell_episodes_round_trip_and_carry_their_cell(tmp_path):
    cell = FrontierCell(n_bits=8, query_bits=2)
    episodes, _info = run_cell(
        cell, POLICY_ZERO_ERROR, n_items=25, seed=0, lambda_bit=1.0
    )
    assert len(episodes) == 25
    path = tmp_path / "zero_error.jsonl"
    write_episodes(path, episodes)
    back = read_episodes(path, strict=True)
    assert [e.to_dict() for e in back] == [e.to_dict() for e in episodes]
    for e in back:
        assert e.metadata["n_bits"] == 8
        assert e.metadata["query_bits"] == 2
        assert e.external_raw_bits == 2.0
        assert e.final_loss == 0.0


def test_gate_fails_when_an_arm_pays_the_wrong_number_of_bits(tmp_path):
    """The regression has to be able to fail, so break it on purpose."""
    cfg = load_grounding_config("configs/experiments/hg_v2/e1_frontier.yaml")
    cell = FrontierCell(n_bits=8, query_bits=1)
    good, _ = run_cell(cell, POLICY_ZERO_ERROR, n_items=20, seed=0, lambda_bit=1.0)
    control, _ = run_cell(cell, POLICY_TRUNCATED, n_items=200, seed=0, lambda_bit=1.0)
    assert build_analysis(cfg, [*good, *control])["gate"]["passed"]

    # One arm charged as though a smaller query had been enough.
    for e in good:
        e.metadata["fiber_size"] = 3
        e.external_raw_bits = 3.0
        e.external_response["raw_bits"] = 3.0
    analysis = build_analysis(cfg, [*good, *control])
    assert not analysis["gate"]["passed"]
    assert any("zero_error_frontier" in f for f in analysis["gate"]["failures"])


def test_frontier_table_reports_the_power_of_its_converse_check():
    cell = FrontierCell(n_bits=8, query_bits=0)
    good, _ = run_cell(cell, POLICY_ZERO_ERROR, n_items=10, seed=0, lambda_bit=1.0)
    control, _ = run_cell(cell, POLICY_TRUNCATED, n_items=320, seed=0, lambda_bit=1.0)
    row = frontier_table([*good, *control])[0]
    assert row["truncated_failure_probability"] == pytest.approx(1 / 16)
    assert row["expected_truncated_failures"] == pytest.approx(20.0)
    assert row["converse_is_tight"] is True


# ------------------------------------------------- the converse, for any code ---
@pytest.mark.parametrize("s,b", [(2, 1), (3, 2), (4, 3), (4, 2), (5, 4)])
def test_no_encoder_below_the_bound_is_zero_error(s, b):
    """Not just the truncated protocol: *no* b < s bit code can be zero-error.

    `truncated_converse` runs one particular b* - 1 protocol -- send every
    coordinate but the last -- which leaves the obvious objection that a
    cleverer code might do better. Random encoders answer it: sampled maps from
    2**s assignments into 2**b messages all have a colliding pair, and a
    colliding pair always differs at some coordinate, which is an index the AI
    can be holding.
    """
    rng = np.random.default_rng([s, b])
    for _ in range(200):
        encoder = rng.integers(0, 2**b, size=2**s)
        witness = converse_witness(encoder, s)
        assert witness is not None, "an encoder below the bound decided everything"
        x, y, i = witness
        assert encoder[x] == encoder[y]
        # The witness must be a genuine disagreement at that coordinate.
        bits_x = np.binary_repr(x, width=s)
        bits_y = np.binary_repr(y, width=s)
        assert bits_x[i] != bits_y[i]


@pytest.mark.parametrize("s", [1, 2, 3, 4])
def test_an_encoder_at_the_bound_can_be_zero_error(s):
    """And the bound is achievable: the identity code has no witness."""
    assert converse_witness(list(range(2**s)), s) is None
    # One message short is not, however many ways it is arranged.
    collapsed = list(range(2**s - 1)) + [0]
    assert converse_witness(collapsed, s) is not None
