"""The item-level record contract."""

from __future__ import annotations

import pytest

from hg.schema import (
    MINIMUM_COLUMNS,
    ItemResult,
    SchemaError,
    check_minimum_columns,
    check_rectangular,
)


def row(item="i0", member="m0", correct=True, **kw) -> ItemResult:
    base = dict(
        item_id=item, task="t", model_family="f", model_version="ckpt",
        member_id=member, condition="DD", prediction="a", correct=correct,
        anchor_label="a", anchor_status="unanimous", score=1.0,
        adapter_seed=0, train_split="s", confidence=0.9,
    )
    base.update(kw)
    return ItemResult(**base)


def test_minimum_columns_present_in_serialized_row():
    d = row().to_dict()
    check_minimum_columns(d)
    assert set(MINIMUM_COLUMNS) <= set(d)


def test_unknown_condition_rejected():
    with pytest.raises(SchemaError, match="unknown condition"):
        row(condition="XX").validate()


def test_unknown_anchor_status_rejected():
    with pytest.raises(SchemaError, match="unknown anchor_status"):
        row(anchor_status="probably").validate()


def test_probs_must_sum_to_one():
    with pytest.raises(SchemaError, match="probs sum"):
        row(probs={"a": 0.3, "b": 0.3}).validate()


def test_confidence_out_of_range_rejected():
    with pytest.raises(SchemaError, match="outside"):
        row(confidence=1.4).validate()


def test_unknown_field_rejected_on_load():
    d = row().to_dict()
    d["surprise"] = 1
    with pytest.raises(SchemaError, match="unexpected fields"):
        ItemResult.from_dict(d)


def test_rectangular_matrix_accepted():
    rows = [row(f"i{i}", f"m{j}") for i in range(3) for j in range(2)]
    items, members = check_rectangular(rows)
    assert items == ["i0", "i1", "i2"]
    assert members == ["m0", "m1"]


def test_ragged_matrix_rejected():
    """A missing cell biases every consensus metric, so it must be loud."""
    rows = [row(f"i{i}", f"m{j}") for i in range(3) for j in range(2)]
    rows.pop()
    with pytest.raises(SchemaError, match="ragged"):
        check_rectangular(rows)


def test_duplicate_cell_rejected():
    rows = [row("i0", "m0"), row("i0", "m0")]
    with pytest.raises(SchemaError, match="duplicate cell"):
        check_rectangular(rows)
