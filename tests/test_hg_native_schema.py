"""Round-trip and field-presence tests for `hg_native.data.schema.Episode`."""

from __future__ import annotations

import numpy as np
import pytest

from hg_native.data.schema import Episode, rng_for, stable_int_key


def _make_episode(**overrides) -> Episode:
    fields = dict(
        episode_id="ep-1",
        family="static_hidden",
        visible_state={"A.cost": 37, "B.cost": 29},
        hidden_state={"h1": "UNKNOWN"},
        full_state={"A.cost": 37, "B.cost": 29, "h1": "YES"},
        available_actions=["ACT_A", "ACT_B", "ASK_h1"],
        correct_action="ASK_h1",
        distortion_profile={"ACT_A": 1.0, "ACT_B": 0.0, "ASK_h1": 0.0},
        needs_grounding=True,
        min_grounding_bits=1.0,
        minimal_query_set=["h1"],
    )
    fields.update(overrides)
    return Episode(**fields)


def test_round_trip_dict():
    ep = _make_episode()
    restored = Episode.from_dict(ep.to_dict())
    assert restored == ep


def test_round_trip_json():
    ep = _make_episode()
    restored = Episode.from_json(ep.to_json())
    assert restored == ep


def test_optional_fields_have_defaults():
    ep = _make_episode()
    assert ep.scaffold_variants == []
    assert ep.sequential_state_id is None
    assert ep.true_quotient_state is None
    assert ep.metadata == {}


def test_from_dict_rejects_missing_required_field():
    data = _make_episode().to_dict()
    del data["correct_action"]
    with pytest.raises(ValueError, match="correct_action"):
        Episode.from_dict(data)


def test_from_dict_tolerates_missing_optional_field():
    data = _make_episode().to_dict()
    del data["metadata"]
    restored = Episode.from_dict(data)
    assert restored.metadata == {}


class TestStableIntKey:
    def test_deterministic_across_calls(self):
        assert stable_int_key("h3") == stable_int_key("h3")

    def test_differs_for_different_strings(self):
        assert stable_int_key("h1") != stable_int_key("h2")

    def test_unaffected_by_python_hash_randomization(self):
        # stable_int_key must not delegate to Python's hash(); a real
        # regression here would only be visible across separate processes
        # with different PYTHONHASHSEED, so instead assert it disagrees with
        # whatever this process's hash() happens to produce, catching an
        # accidental `hash(value)` implementation directly.
        assert stable_int_key("h3") != (hash("h3") & 0xFFFFFFFF)


class TestRngFor:
    def test_deterministic(self):
        a = rng_for(0, "static_hidden", "one_bit_deficit", 5).integers(0, 1_000_000, size=10)
        b = rng_for(0, "static_hidden", "one_bit_deficit", 5).integers(0, 1_000_000, size=10)
        assert np.array_equal(a, b)

    def test_sensitive_to_every_key(self):
        base = rng_for(0, "static_hidden", "one_bit_deficit", 5).integers(0, 1_000_000, size=10)
        variants = [
            rng_for(1, "static_hidden", "one_bit_deficit", 5),
            rng_for(0, "static_hidden", "multi_bit_deficit", 5),
            rng_for(0, "static_hidden", "one_bit_deficit", 6),
        ]
        for variant in variants:
            assert not np.array_equal(base, variant.integers(0, 1_000_000, size=10))

    def test_rejects_non_str_int_key(self):
        with pytest.raises(TypeError):
            rng_for(0, 1.5)

    def test_returns_a_generator(self):
        assert isinstance(rng_for(0, "x"), np.random.Generator)
