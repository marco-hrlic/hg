"""Equalizing the training budget across model families (protocol section 13).

"Use a common effective token budget per model." This is harder than it sounds
and matters more than it sounds, because families tokenize differently: the
same text is a different number of tokens for Qwen than for Gemma. Two ways to
equalize, and they are not the same experiment:

* **Equal tokens** -- each family sees the same *token* count. Fair in compute,
  unfair in information: the family with a denser tokenizer sees more text.
* **Equal examples** -- each family sees the same *items*. Fair in information,
  unfair in compute.

For M2-LLM the question is about information diversity, so `equal_examples` is
the default: the training slices are the independent variable and every family
must see the same slice, whatever that costs in tokens. The realized token
counts are recorded either way, so a compute confound can be checked after the
fact instead of being assumed absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

MODES = ("equal_examples", "equal_tokens")


@dataclass(slots=True)
class TokenBudget:
    """Realized budget for one member, recorded in its adapter card."""

    member_id: str
    mode: str
    n_examples: int
    n_tokens: int
    n_epochs: float
    effective_tokens: int  # n_tokens * n_epochs, what the optimizer actually saw
    truncated_examples: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "member_id": self.member_id,
            "mode": self.mode,
            "n_examples": self.n_examples,
            "n_tokens": self.n_tokens,
            "n_epochs": self.n_epochs,
            "effective_tokens": self.effective_tokens,
            "truncated_examples": self.truncated_examples,
            "notes": self.notes,
        }


def plan_budget(
    member_id: str,
    texts: Sequence[str],
    count_tokens: Callable[[str], int],
    *,
    mode: str = "equal_examples",
    target_tokens: int | None = None,
    n_epochs: float = 1.0,
    max_seq_len: int = 1024,
) -> TokenBudget:
    """Compute the realized budget for one member.

    Under `equal_tokens`, examples are taken in order until `target_tokens` is
    reached -- deliberately truncating the *slice*, not each example, so the
    items a member trained on remain a well-defined set for the O_jk overlap
    computation. Truncating individual examples would leave the overlap matrix
    describing items the model only partly saw.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    counts = [count_tokens(t) for t in texts]
    truncated = sum(1 for c in counts if c > max_seq_len)
    counts = [min(c, max_seq_len) for c in counts]

    if mode == "equal_examples":
        used = len(texts)
        tokens = sum(counts)
        note = "all slice items used; token count is a realized value, not a target"
    else:
        if target_tokens is None:
            raise ValueError("equal_tokens mode requires target_tokens")
        tokens = 0
        used = 0
        for c in counts:
            if tokens + c > target_tokens:
                break
            tokens += c
            used += 1
        note = (
            f"slice truncated to {used}/{len(texts)} items to hit "
            f"{target_tokens} tokens; record the reduced item set for O_jk"
        )

    return TokenBudget(
        member_id=member_id,
        mode=mode,
        n_examples=used,
        n_tokens=tokens,
        n_epochs=n_epochs,
        effective_tokens=int(tokens * n_epochs),
        truncated_examples=truncated,
        notes=note,
    )


def effective_steps(budget: TokenBudget, tokens_per_step: int) -> int:
    """Optimizer steps implied by a budget, for a fixed tokens-per-step."""
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")
    return max(1, budget.effective_tokens // tokens_per_step)


def budget_report(budgets: Sequence[TokenBudget]) -> dict[str, object]:
    """Cross-member budget comparison, with the confound flagged explicitly."""
    if not budgets:
        return {"n_members": 0}
    tokens = [b.effective_tokens for b in budgets]
    examples = [b.n_examples for b in budgets]
    lo, hi = min(tokens), max(tokens)
    spread = (hi - lo) / hi if hi else 0.0
    warnings: list[str] = []
    if len(set(examples)) > 1:
        warnings.append(
            f"members saw different numbers of examples ({min(examples)}-{max(examples)}); "
            "information diversity is confounded with training-set size"
        )
    if spread > 0.25:
        warnings.append(
            f"effective token budgets differ by {spread:.0%} across members; "
            "a family that saw materially more tokens will look stronger, and "
            "capability differences inflate apparent diversity benefit "
            "(protocol section 18)"
        )
    return {
        "n_members": len(budgets),
        "effective_tokens": {b.member_id: b.effective_tokens for b in budgets},
        "n_examples": {b.member_id: b.n_examples for b in budgets},
        "token_spread": spread,
        "warnings": warnings,
    }
