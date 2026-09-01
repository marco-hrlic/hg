"""Query efficiency eval -- torch-free (synthetic distortion_hat rows)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hg_native.data.static_hidden import generate_episode
from hg_native.eval.query_efficiency import chosen_action, query_efficiency
from hg_native.training.actions import ACTIONS, action_id


def _row(favored_action: str) -> np.ndarray:
    """A distortion_hat row that strongly prefers `favored_action`."""
    row = np.zeros(len(ACTIONS))
    row[action_id(favored_action)] = -1.0
    return row


# --- chosen_action ---------------------------------------------------


def test_chosen_action_picks_the_favored_action():
    ep = generate_episode(0, "one_bit_deficit", 0)
    assert chosen_action(ep, _row("ASK_h1")) == "ASK_h1"
    assert chosen_action(ep, _row("ACT_A")) == "ACT_A"


def test_chosen_action_only_considers_available_actions():
    # no_deficit has no ASK actions at all; favoring one that isn't
    # available must not be selectable.
    ep = generate_episode(0, "no_deficit", 0)
    row = np.zeros(len(ACTIONS))
    row[action_id("ASK_h1")] = -100.0  # would dominate if it were considered
    row[action_id("ACT_A")] = -1.0
    assert chosen_action(ep, row) == "ACT_A"


# --- query_efficiency ---------------------------------------------------


def test_always_sound_single_ask_on_one_bit_deficit():
    episodes = [generate_episode(0, "one_bit_deficit", i) for i in range(20)]
    distortion_hat = [_row("ASK_h1") for _ in episodes]  # always resolves in this subclass
    result = query_efficiency(episodes, distortion_hat)

    assert result["soundness"] == pytest.approx(1.0)
    assert result["under_asking_rate"] == pytest.approx(0.0)
    assert math.isnan(result["unnecessary_ask_rate"])  # no zero-deficit episodes in this set
    assert result["b_system_mean"] == pytest.approx(1.0)
    assert result["bits_per_resolved_decision"] == pytest.approx(1.0)
    assert result["b_star_mean"] == pytest.approx(1.0)
    # b_generic == b_star_mean here (every deficit item needs exactly 1 bit),
    # so eta_Q's denominator is 0 -- degenerate, correctly nan rather than a
    # divide-by-zero crash or a fabricated "perfect" 1.0.
    assert math.isnan(result["eta_q"])


def test_never_asking_on_deficit_episodes_is_all_under_asking():
    episodes = [generate_episode(0, "one_bit_deficit", i) for i in range(20)]
    distortion_hat = [_row("ACT_A") for _ in episodes]
    result = query_efficiency(episodes, distortion_hat)

    assert result["under_asking_rate"] == pytest.approx(1.0)
    assert result["b_system_mean"] == pytest.approx(0.0)
    assert math.isnan(result["soundness"])  # nothing was ever asked, so nothing to score


def test_unnecessary_ask_on_irrelevant_hidden():
    episodes = [generate_episode(0, "irrelevant_hidden", i) for i in range(20)]
    distortion_hat = [_row("ASK_h1") for _ in episodes]  # asks despite zero deficit
    result = query_efficiency(episodes, distortion_hat)

    assert result["unnecessary_ask_rate"] == pytest.approx(1.0)
    assert math.isnan(result["under_asking_rate"])  # no deficit episodes in this set
    assert math.isnan(result["eta_q"])  # eta_Q undefined with no deficit episodes


def test_never_asking_on_zero_deficit_is_no_unnecessary_asks():
    episodes = [generate_episode(0, "irrelevant_hidden", i) for i in range(20)]
    distortion_hat = [_row("ACT_A") for _ in episodes]
    result = query_efficiency(episodes, distortion_hat)
    assert result["unnecessary_ask_rate"] == pytest.approx(0.0)


def test_eta_q_partial_credit_for_multi_bit_deficit():
    # A policy that always asks h1 (the first of the minimal_query_set):
    # fully resolves one_bit_deficit (B*=1) but only sometimes resolves
    # multi_bit_deficit (B*=2), while spending the same single bit either way.
    episodes = [generate_episode(0, "one_bit_deficit", i) for i in range(20)] + [
        generate_episode(0, "multi_bit_deficit", i) for i in range(20)
    ]
    distortion_hat = [_row("ASK_h1") for _ in episodes]
    result = query_efficiency(episodes, distortion_hat)

    assert result["b_star_mean"] == pytest.approx(1.5)  # mean of 20x1.0 and 20x2.0
    assert result["b_generic"] == pytest.approx(2.0)  # the hardest case in this set
    # b_system is 1 bit for every deficit episode regardless of subclass.
    assert result["eta_q"] == pytest.approx((2.0 - 1.0) / (2.0 - 1.5))


def test_query_efficiency_rejects_length_mismatch():
    episodes = [generate_episode(0, "one_bit_deficit", i) for i in range(5)]
    with pytest.raises(ValueError):
        query_efficiency(episodes, np.zeros((4, len(ACTIONS))))
