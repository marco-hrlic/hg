"""LoRA bookkeeping. No GPU: these check the parts that can silently corrupt
the experiment rather than the training loop itself."""

from __future__ import annotations

from pathlib import Path

import pytest

from hg.anchors.panel import PanelItem, build_anchored_items
from hg.finetune.budget import budget_report, plan_budget
from hg.finetune.lora import (
    LoRASettings,
    adapter_hash,
    check_no_adapter_sharing,
    read_adapter_card,
    train_lora,
    write_adapter_card,
)


def items(n=10):
    return build_anchored_items(
        [PanelItem(f"i{k}", f"text {k}", ["a", "a", "a"]) for k in range(n)]
    )


def test_adapter_hash_depends_on_training_items():
    s = LoRASettings()
    common = dict(checkpoint="ckpt", settings=s, adapter_seed=0, prompt_version="classify.v1")
    a = adapter_hash(train_item_ids=["i0", "i1"], **common)
    b = adapter_hash(train_item_ids=["i0", "i2"], **common)
    assert a != b


def test_adapter_hash_ignores_training_item_order():
    """Slice membership changes what was learned; slice order does not."""
    s = LoRASettings()
    common = dict(checkpoint="ckpt", settings=s, adapter_seed=0, prompt_version="classify.v1")
    assert adapter_hash(train_item_ids=["i1", "i0"], **common) == adapter_hash(
        train_item_ids=["i0", "i1"], **common
    )


def test_adapter_hash_depends_on_seed_checkpoint_and_prompt():
    s = LoRASettings()
    base = dict(
        checkpoint="ckpt", train_item_ids=["i0"], settings=s,
        adapter_seed=0, prompt_version="classify.v1",
    )
    h = adapter_hash(**base)
    assert adapter_hash(**{**base, "adapter_seed": 1}) != h
    assert adapter_hash(**{**base, "checkpoint": "other"}) != h
    assert adapter_hash(**{**base, "prompt_version": "policy.v1"}) != h


def test_adapter_card_keeps_the_full_item_list_not_just_a_count():
    """O_jk cannot be recomputed from a count, so auditing a 'disjoint' claim
    later requires the ids themselves."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "adapter_card.json"
        write_adapter_card(
            path, member_id="m0", checkpoint="ckpt", family="qwen", lineage="qwen",
            train_item_ids=["i1", "i0"], settings=LoRASettings(),
            adapter_seed=0, prompt_version="classify.v1",
        )
        card = read_adapter_card(path)
        assert card["train_item_ids"] == ["i0", "i1"]
        assert card["n_train_items"] == 2


def test_reusing_an_adapter_across_families_is_refused(tmp_path):
    """Protocol section 13: do not share adapters across model families."""
    write_adapter_card(
        tmp_path / "adapter_card.json",
        member_id="m0", checkpoint="qwen-ckpt", family="qwen", lineage="qwen",
        train_item_ids=["i0"], settings=LoRASettings(),
        adapter_seed=0, prompt_version="classify.v1",
    )
    check_no_adapter_sharing(tmp_path, "qwen-ckpt")  # same base: fine
    with pytest.raises(RuntimeError, match="must not be shared across model"):
        check_no_adapter_sharing(tmp_path, "gemma-ckpt")


def test_dry_run_writes_a_card_without_touching_a_gpu(tmp_path):
    result = train_lora(
        checkpoint="ckpt", output_dir=tmp_path / "adapter", train_items=items(8),
        labels=["not_harmful", "harmful"], prompt_version="classify.v1",
        member_id="m0", family="qwen", lineage="qwen", dry_run=True,
    )
    assert result["adapter_path"] is None
    assert result["adapter_hash"]
    assert len(result["train_item_ids"]) == 8
    assert (tmp_path / "adapter" / "adapter_card.json").exists()


def test_equal_examples_is_the_default_budget_mode():
    """The training slices are the independent variable, so every family must
    see the same items whatever that costs in tokens."""
    texts = ["a b c", "d e", "f g h i"]
    b = plan_budget("m0", texts, lambda t: len(t.split()))
    assert b.mode == "equal_examples"
    assert b.n_examples == 3
    assert b.n_tokens == 9


def test_equal_tokens_truncates_the_slice_not_the_examples():
    b = plan_budget(
        "m0", ["a b c", "d e", "f g h i"], lambda t: len(t.split()),
        mode="equal_tokens", target_tokens=5,
    )
    assert b.n_examples == 2
    assert b.n_tokens == 5
    assert "record the reduced item set" in b.notes


def test_equal_tokens_requires_a_target():
    with pytest.raises(ValueError, match="requires target_tokens"):
        plan_budget("m0", ["a"], lambda t: 1, mode="equal_tokens")


def test_budget_report_flags_unequal_example_counts():
    budgets = [
        plan_budget("m0", ["a b"] * 10, lambda t: 2),
        plan_budget("m1", ["a b"] * 5, lambda t: 2),
    ]
    report = budget_report(budgets)
    assert any("different numbers of examples" in w for w in report["warnings"])


def test_budget_report_flags_a_large_token_spread():
    budgets = [
        plan_budget("m0", ["a b c d"] * 10, lambda t: 4),
        plan_budget("m1", ["a"] * 10, lambda t: 1),
    ]
    report = budget_report(budgets)
    assert any("token budgets differ" in w for w in report["warnings"])


def test_training_prompt_matches_the_evaluation_template():
    """A train/eval template mismatch is the cheapest way to manufacture a
    family-specific gap, and it would read as representational diversity."""
    from hg.finetune.lora import build_training_texts
    from hg.inference.prompts import render_prompt

    its = items(1)
    labels = ["not_harmful", "harmful"]
    text = build_training_texts(its, labels, "classify.v1")[0]
    prompt = render_prompt("classify.v1", its[0].x, labels)
    assert text == f"{prompt} {its[0].reference}"
