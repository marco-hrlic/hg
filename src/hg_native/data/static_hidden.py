"""Family A — Static Hidden Variable generator (spec section 5.1).

Purpose: grounding discrimination, structured minimal queries, and
cost-sensitive routing, on a task simple enough that ground truth (the
minimal external-information cost `B*`, the minimal query set, and the exact
distortion of every available action) is computed exactly rather than
estimated.

All six subclasses share one underlying decision family: two priced actions
`ACT_A`/`ACT_B`, a boolean "B is cheaper" fact from visible costs, and a rule
that additionally requires zero, one, or two independent hidden bits
(`h1`, `h2`, ...) to equal `YES`. What varies per subclass is which hidden
bits the rule actually references and what decorative variables (extra
visible distractors, extra irrelevant hidden vars, a redundant copy of a
hidden bit) are layered on top — see `_RULES` below.

Design choice worth flagging back to the theory/planning track (see
[[hg-theory-implementation-split]]): the spec's M0 action space has one
`ASK_k` per hidden variable but no multi-turn dialogue, so a genuine
multi-bit deficit (`multi_bit_deficit`) cannot be *fully* resolved by a
single action. `correct_action` for that subclass is therefore the first
element of `minimal_query_set` (asking is still strictly better than acting
blind), while `min_grounding_bits` and `minimal_query_set` correctly report
that two bits/two asks are needed. `distortion_profile` reflects this
honestly: asking only one of the two required bits still carries positive
expected distortion, computed by resolving the asked variable(s) to their
true value and defaulting every other unresolved rule variable to `NO`
(see `_resolve_decision`) — not by pretending a partial ask fully resolves
the episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rule_expr import Expr, render
from .schema import Episode, rng_for

SUBCLASSES = (
    "no_deficit",
    "one_bit_deficit",
    "multi_bit_deficit",
    "redundant_hidden",
    "irrelevant_hidden",
    "distractor_visible",
)

COST_LOW = 10
COST_HIGH = 99
DEFAULT_UNKNOWN_BIT = "NO"  # convention for guessing an unresolved rule bit

# M1 kickoff section 6.4 (distractor-density OOD): the pool decorative
# distractor names are drawn from, indexed 0.. by `n_distractor_visible`.
# The first two entries are the original two names (`distractor_visible`'s
# `_Rule.n_distractor_visible=2` default), so IID behavior is byte-identical
# to before this pool existed; entries beyond that support OOD densities up
# to `len(DISTRACTOR_NAMES)`.
DISTRACTOR_NAMES: tuple[str, ...] = ("C.cost", "A.weight") + tuple(f"decoy_{i}.cost" for i in range(2, 30))


@dataclass(frozen=True)
class _Rule:
    """One subclass's decision rule and the decorative variables around it."""

    rule_vars: tuple[str, ...]  # hidden bits the rule actually references
    extra_hidden_vars: tuple[str, ...]  # present but not referenced by the rule
    # action name -> rule vars it resolves to their true value when asked.
    # A rule var not covered by any action's reveal set can never be fully
    # resolved in one hop (the multi_bit_deficit case, by construction).
    reveals: dict[str, tuple[str, ...]]
    force_b_cheaper: bool | None  # None = unconstrained
    rule_text: str
    n_distractor_visible: int = 0


def _decide(rule_vars: tuple[str, ...], b_cheaper: bool, bits: dict[str, str]) -> str:
    """`ACT_B` iff B is cheaper and every rule bit is YES; else `ACT_A`."""
    if not b_cheaper:
        return "ACT_A"
    if all(bits.get(v) == "YES" for v in rule_vars):
        return "ACT_B"
    return "ACT_A"


def _resolve_decision(rule: _Rule, b_cheaper: bool, true_bits: dict[str, str], asked: str | None) -> str:
    """What a single-shot agent would decide after (optionally) asking `asked`.

    Every rule variable not revealed by `asked` defaults to `DEFAULT_UNKNOWN_BIT`
    — the honest "still had to guess" outcome, not an assumption that any ask
    fully resolves the episode.
    """
    resolved = dict.fromkeys(rule.rule_vars, DEFAULT_UNKNOWN_BIT)
    if asked is not None:
        for var in rule.reveals.get(asked, ()):
            resolved[var] = true_bits[var]
    return _decide(rule.rule_vars, b_cheaper, resolved)


_RULES: dict[str, _Rule] = {
    "no_deficit": _Rule(
        rule_vars=(),
        extra_hidden_vars=(),
        reveals={},
        force_b_cheaper=None,
        rule_text="choose B iff B is cheaper; otherwise choose A.",
    ),
    "one_bit_deficit": _Rule(
        rule_vars=("h1",),
        extra_hidden_vars=(),
        reveals={"ASK_h1": ("h1",)},
        force_b_cheaper=True,
        rule_text="choose B iff B is cheaper AND h1 = YES; otherwise choose A.",
    ),
    "multi_bit_deficit": _Rule(
        rule_vars=("h1", "h2"),
        extra_hidden_vars=(),
        reveals={"ASK_h1": ("h1",), "ASK_h2": ("h2",)},
        force_b_cheaper=True,
        rule_text="choose B iff B is cheaper AND h1 = YES AND h2 = YES; otherwise choose A.",
    ),
    "redundant_hidden": _Rule(
        rule_vars=("h1",),
        extra_hidden_vars=("h2",),
        # h2 is a hard-wired copy of h1 (see generate_episode): asking either
        # one reveals h1's true value, so B* stays 1 bit, not 2.
        reveals={"ASK_h1": ("h1",), "ASK_h2": ("h1",)},
        force_b_cheaper=True,
        rule_text="choose B iff B is cheaper AND h1 = YES; otherwise choose A. "
        "(h2 is always a copy of h1.)",
    ),
    "irrelevant_hidden": _Rule(
        rule_vars=(),
        extra_hidden_vars=("h1", "h2"),
        reveals={},
        force_b_cheaper=None,
        rule_text="choose B iff B is cheaper; otherwise choose A. (h1, h2 are not needed.)",
    ),
    "distractor_visible": _Rule(
        rule_vars=("h1",),
        extra_hidden_vars=(),
        reveals={"ASK_h1": ("h1",)},
        force_b_cheaper=True,
        rule_text="choose B iff B is cheaper AND h1 = YES; otherwise choose A. "
        "(C.cost and A.weight are not needed.)",
        n_distractor_visible=2,
    ),
}


def _draw_costs(rng, force_b_cheaper: bool | None, cost_low: int, cost_high: int) -> tuple[int, int]:
    while True:
        a_cost = int(rng.integers(cost_low, cost_high + 1))
        b_cost = int(rng.integers(cost_low, cost_high + 1))
        if a_cost == b_cost:
            continue
        if force_b_cheaper is True and not (b_cost < a_cost):
            continue
        if force_b_cheaper is False and not (b_cost > a_cost):
            continue
        return a_cost, b_cost


def _draw_bit(rng) -> str:
    return "YES" if int(rng.integers(0, 2)) == 1 else "NO"


def generate_episode(
    seed: int,
    subclass: str,
    index: int,
    *,
    cost_range: tuple[int, int] | None = None,
    n_distractor_visible: int | None = None,
) -> Episode:
    """Generate one deterministic episode of `subclass` (same seed+index+subclass -> identical episode).

    `cost_range`/`n_distractor_visible` default to `None`, reproducing
    exactly today's behavior (`(COST_LOW, COST_HIGH)`, the subclass's own
    `_Rule.n_distractor_visible`) -- an OOD pool (M1 kickoff section 6.3
    parameter OOD / section 6.4 distractor-density OOD) passes an explicit
    override instead. `_decide`/`_resolve_decision` never read either
    knob, so overriding them changes only the *distribution* costs/decoys
    are drawn from, never the decision rule itself.
    """
    if subclass not in _RULES:
        raise ValueError(f"unknown static_hidden subclass {subclass!r}; expected one of {SUBCLASSES}")
    rule = _RULES[subclass]
    rng = rng_for(seed, "static_hidden", subclass, index)

    cost_low, cost_high = cost_range if cost_range is not None else (COST_LOW, COST_HIGH)
    n_distractors = rule.n_distractor_visible if n_distractor_visible is None else n_distractor_visible
    if n_distractors > len(DISTRACTOR_NAMES):
        raise ValueError(f"n_distractor_visible={n_distractors} exceeds DISTRACTOR_NAMES pool size {len(DISTRACTOR_NAMES)}")

    a_cost, b_cost = _draw_costs(rng, rule.force_b_cheaper, cost_low, cost_high)
    b_cheaper = b_cost < a_cost
    true_bits = {v: _draw_bit(rng) for v in rule.rule_vars}
    extra_bits = {v: _draw_bit(rng) for v in rule.extra_hidden_vars}
    if subclass == "redundant_hidden":
        extra_bits["h2"] = true_bits["h1"]  # hard-wired copy, per _RULES doc

    visible_state: dict[str, Any] = {"A.cost": a_cost, "B.cost": b_cost}
    for i in range(n_distractors):
        # Deterministic decorative distractors: not referenced by any rule,
        # not fed into any head's target, present purely to test that the
        # generator's own ground truth (and later, the model) ignores them.
        visible_state[DISTRACTOR_NAMES[i]] = int(rng.integers(cost_low, cost_high + 1))

    all_hidden_names = list(rule.rule_vars) + list(rule.extra_hidden_vars)
    hidden_state = {v: "UNKNOWN" for v in all_hidden_names}
    full_state = {**visible_state, **true_bits, **extra_bits}

    ground_truth_decision = _decide(rule.rule_vars, b_cheaper, true_bits)
    # True iff some single action's reveal set covers every rule var at once
    # -- not merely whether the union across all actions does (multi_bit_deficit
    # fails this: h1 and h2 are each revealed by a different action, so no
    # single ASK resolves the conjunction in one hop).
    rule_vars_set = set(rule.rule_vars)
    fully_resolvable = not rule_vars_set or any(rule_vars_set <= set(vs) for vs in rule.reveals.values())
    min_grounding_bits = float(len(rule.rule_vars))
    needs_grounding = len(rule.rule_vars) > 0
    minimal_query_set = list(rule.rule_vars)
    correct_action = ground_truth_decision if not needs_grounding else f"ASK_{minimal_query_set[0]}"

    available_actions = ["ACT_A", "ACT_B"] + [f"ASK_{h}" for h in all_hidden_names]
    distortion_profile: dict[str, float] = {}
    for action in available_actions:
        if action in ("ACT_A", "ACT_B"):
            distortion_profile[action] = 0.0 if action == ground_truth_decision else 1.0
        else:
            asked = action  # "ASK_h1" etc. matches the reveals-dict keys directly
            resolved_decision = _resolve_decision(rule, b_cheaper, true_bits, asked)
            distortion_profile[action] = 0.0 if resolved_decision == ground_truth_decision else 1.0

    episode_id = f"static_hidden-{subclass}-{seed}-{index}"
    return Episode(
        episode_id=episode_id,
        family="static_hidden",
        visible_state=visible_state,
        hidden_state=hidden_state,
        full_state=full_state,
        available_actions=available_actions,
        correct_action=correct_action,
        distortion_profile=distortion_profile,
        needs_grounding=needs_grounding,
        min_grounding_bits=min_grounding_bits,
        minimal_query_set=minimal_query_set,
        metadata={
            "subclass": subclass,
            "rule_text": rule.rule_text,
            "rule_vars": list(rule.rule_vars),
            "ground_truth_decision": ground_truth_decision,
            "fully_resolvable_in_one_ask": fully_resolvable,
        },
    )


def generate_batch(
    seed: int,
    subclass: str,
    n_items: int,
    *,
    cost_range: tuple[int, int] | None = None,
    n_distractor_visible: int | None = None,
) -> list[Episode]:
    return [
        generate_episode(seed, subclass, i, cost_range=cost_range, n_distractor_visible=n_distractor_visible)
        for i in range(n_items)
    ]


# M1 kickoff section 6.1 (lexical OOD): "do not allow shared sentence
# templates between train and lexical-OOD test" -- every literal fragment
# below is deliberately disjoint (down to the word level) from the
# `"canonical"` fragments in `render_text`, so `hg_native.data.ood`'s
# `assert_no_shared_lexical_templates` has something real to check.
# `rule_text` itself (drawn from `_RULES`, not from either fragment set
# below) is shared between surface forms on purpose: it is the substantive
# statement of the decision rule, not decorative wording -- an OOD render
# still has to describe the same logic, just never in the same sentence
# scaffold around it.
def _lexical_ood_key(key: str) -> str:
    if key == "A.cost":
        return "Option A price"
    if key == "B.cost":
        return "Option B price"
    name, _, field = key.partition(".")
    return f"Extra option {name} {field}"


def render_text(episode: Episode, surface_form: str = "canonical") -> str:
    """Render the spec section 5.1 `TASK STATIC-CHOICE` text block for an episode.

    `surface_form="canonical"` (default) is today's exact wording, unchanged.
    `surface_form="lexical_ood"` (M1 kickoff section 6.1) renders the same
    ground truth under a fully alternate vocabulary/serialization -- see
    `hg_native.data.ood.lexical_ood_pool`.
    """
    if surface_form == "canonical":
        lines = ["TASK STATIC-CHOICE"]
        for key, value in episode.visible_state.items():
            lines.append(f"{key} = {value}")
        for var in episode.metadata.get("rule_vars", ()):
            lines.append(f"B.requires = {var}")
        lines.append("RULE:")
        lines.append(episode.metadata.get("rule_text", ""))
        lines.append("KNOWN:")
        for key, value in episode.visible_state.items():
            lines.append(f"{key} = {value}")
        for var, value in episode.hidden_state.items():
            lines.append(f"{var} = {value}")
        lines.append("DECIDE OR REQUEST INFORMATION.")
        return "\n".join(lines)
    if surface_form == "lexical_ood":
        # `B.requires = h<k>` reflects the rule's structure, independent of
        # whether this instance's costs make it decision-relevant -- same
        # contrast the canonical render's docstring notes, just reworded.
        lines = ["OPTION SELECTION"]
        for key, value in episode.visible_state.items():
            lines.append(f"{_lexical_ood_key(key)}: {value}")
        for var in episode.metadata.get("rule_vars", ()):
            lines.append(f"Option B needs {var}")
        lines.append("CONDITION:")
        lines.append(episode.metadata.get("rule_text", ""))
        lines.append("GIVEN:")
        for key, value in episode.visible_state.items():
            lines.append(f"{_lexical_ood_key(key)}: {value}")
        for var, value in episode.hidden_state.items():
            lines.append(f"{var}: {value}")
        lines.append("MAKE YOUR CHOICE.")
        return "\n".join(lines)
    raise ValueError(f"unknown surface_form {surface_form!r}; expected 'canonical' or 'lexical_ood'")


# --- structural OOD (M1 kickoff section 6.2) --------------------------------
#
# `_decide`/`_RULES` above (the 6 named subclasses) are untouched by
# everything below -- this is a separate, additive code path so nothing
# already relied on can regress. `_decide`'s fixed AND-only conjunction
# only covers one composition shape; `generate_structural_episode`
# generalizes to an arbitrary `rule_expr.Expr` (AND/OR/NOT) over however
# many hidden bits the expression actually references.


def _condition_key(condition: Expr) -> str:
    return repr(condition.signature())


def generate_structural_episode(seed: int, index: int, condition: Expr, *, cost_range: tuple[int, int] | None = None) -> Episode:
    """`ACT_B` iff `b_cheaper AND condition.evaluate(bits)`, `bits` covering
    exactly `condition.free_vars()`.

    Ground truth conservatively declares *every* free var required
    (`min_grounding_bits = len(free_vars)`, `correct_action` asks the
    first one) -- the same single-hop-ASK simplification `multi_bit_deficit`
    already documents (module docstring above): a genuinely
    short-circuiting grounding cost for an OR-shaped condition (where one
    bit can sometimes already determine the answer) is a bigger change
    than this generator's scope. `distortion_profile` still resolves
    exactly the asked variable and defaults the rest to `NO`
    (`_resolve_decision`'s pattern), so it stays honest about partial asks.
    """
    rule_vars = tuple(sorted(condition.free_vars()))
    if not rule_vars:
        raise ValueError("condition must reference at least one free variable")
    rng = rng_for(seed, "static_hidden_structural", _condition_key(condition), index)
    cost_low, cost_high = cost_range if cost_range is not None else (COST_LOW, COST_HIGH)
    a_cost, b_cost = _draw_costs(rng, True, cost_low, cost_high)  # force_b_cheaper=True, matches the deficit subclasses
    b_cheaper = b_cost < a_cost
    true_bits = {v: _draw_bit(rng) for v in rule_vars}

    ground_truth_decision = "ACT_B" if (b_cheaper and condition.evaluate(true_bits)) else "ACT_A"
    minimal_query_set = list(rule_vars)
    correct_action = f"ASK_{minimal_query_set[0]}"

    def _resolved_decision(asked: str | None) -> str:
        resolved = dict.fromkeys(rule_vars, DEFAULT_UNKNOWN_BIT)
        if asked is not None:
            var = asked.removeprefix("ASK_")
            if var in resolved:
                resolved[var] = true_bits[var]
        return "ACT_B" if (b_cheaper and condition.evaluate(resolved)) else "ACT_A"

    available_actions = ["ACT_A", "ACT_B"] + [f"ASK_{v}" for v in rule_vars]
    distortion_profile: dict[str, float] = {}
    for action in available_actions:
        if action in ("ACT_A", "ACT_B"):
            distortion_profile[action] = 0.0 if action == ground_truth_decision else 1.0
        else:
            distortion_profile[action] = 0.0 if _resolved_decision(action) == ground_truth_decision else 1.0

    visible_state: dict[str, Any] = {"A.cost": a_cost, "B.cost": b_cost}
    hidden_state = {v: "UNKNOWN" for v in rule_vars}
    full_state = {**visible_state, **true_bits}
    condition_text = render(condition)

    return Episode(
        episode_id=f"static_hidden-structural-{_condition_key(condition)}-{seed}-{index}",
        family="static_hidden",
        visible_state=visible_state,
        hidden_state=hidden_state,
        full_state=full_state,
        available_actions=available_actions,
        correct_action=correct_action,
        distortion_profile=distortion_profile,
        needs_grounding=True,
        min_grounding_bits=float(len(rule_vars)),
        minimal_query_set=minimal_query_set,
        metadata={
            "subclass": "structural",
            "rule_text": f"choose B iff B is cheaper AND {condition_text}; otherwise choose A.",
            "rule_vars": list(rule_vars),
            "ground_truth_decision": ground_truth_decision,
            "condition_signature": condition.signature(),
        },
    )


def generate_structural_batch(seed: int, n_items: int, condition: Expr, cost_range: tuple[int, int] | None = None) -> list[Episode]:
    return [generate_structural_episode(seed, i, condition, cost_range=cost_range) for i in range(n_items)]
