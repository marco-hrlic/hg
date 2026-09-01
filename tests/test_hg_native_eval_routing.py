"""Router value evaluation -- torch-free."""

from __future__ import annotations

import math

import pytest

from hg_native.data.accessibility import generate_episode as gen_accessibility
from hg_native.data.static_hidden import generate_episode as gen_static_hidden
from hg_native.eval.routing import expected_cost, route_accuracy, route_value_report
from hg_native.model.config import DEFAULT_ROUTES
from hg_native.training.targets import route_target


def _one_bit_episodes(n=30):
    return [gen_static_hidden(0, "one_bit_deficit", i) for i in range(n)]


def test_route_accuracy_perfect_when_using_the_oracle_itself():
    episodes = _one_bit_episodes(20)
    oracle_pred = [route_target(ep)[0] for ep in episodes]
    assert route_accuracy(episodes, oracle_pred) == pytest.approx(1.0)


def test_route_accuracy_zero_when_always_wrong():
    episodes = _one_bit_episodes(20)
    ask_idx = DEFAULT_ROUTES.index("ASK")
    act_idx = DEFAULT_ROUTES.index("ACT")
    always_act = [act_idx] * len(episodes)
    # one_bit_deficit's oracle route at default lambda is ASK (see
    # test_hg_native_targets.py); always acting should score 0.
    assert route_accuracy(episodes, always_act) == pytest.approx(0.0)
    always_ask = [ask_idx] * len(episodes)
    assert route_accuracy(episodes, always_ask) == pytest.approx(1.0)


def test_expected_cost_skips_unavailable_routes():
    episodes = _one_bit_episodes(5)  # no THINK action in this family
    cost = expected_cost(episodes, ["THINK"] * len(episodes), lambda_b=1.0, lambda_k=1.0)
    assert math.isnan(cost)  # every episode skipped -> no valid entries


def test_expected_cost_matches_route_costs_directly():
    from hg_native.training.targets import route_costs

    episodes = _one_bit_episodes(10)
    manual_mean = sum(route_costs(ep, 1.0, 1.0)["ASK"] for ep in episodes) / len(episodes)
    assert expected_cost(episodes, ["ASK"] * len(episodes), 1.0, 1.0) == pytest.approx(manual_mean)


def test_route_value_report_has_every_baseline_at_every_grid_point():
    episodes = _one_bit_episodes(10)
    ask_idx = DEFAULT_ROUTES.index("ASK")
    report = route_value_report(episodes, [ask_idx] * len(episodes), lambda_grid=[(0.1, 1.0), (1.0, 1.0), (10.0, 1.0)])
    assert len(report["grid"]) == 3
    for point in report["grid"]:
        assert {"lambda_b", "lambda_k", "model", "oracle", "always_act", "always_think", "always_ask"} <= set(point)


def test_route_value_report_model_matches_oracle_when_predicting_the_oracle():
    episodes = _one_bit_episodes(10)
    oracle_pred = [route_target(ep, 0.1, 1.0)[0] for ep in episodes]
    report = route_value_report(episodes, oracle_pred, lambda_grid=[(0.1, 1.0)])
    point = report["grid"][0]
    assert point["model"] == pytest.approx(point["oracle"])


def test_route_value_report_ask_route_becomes_expensive_at_high_lambda_b():
    episodes = _one_bit_episodes(10)
    report = route_value_report(episodes, [DEFAULT_ROUTES.index("ASK")] * len(episodes), lambda_grid=[(0.1, 1.0), (10.0, 1.0)])
    cheap, expensive = report["grid"]
    assert report["grid"][0]["always_ask"] < report["grid"][1]["always_ask"]
