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


def _draw_costs(rng, force_b_cheaper: bool | None) -> tuple[int, int]:
    while True:
        a_cost = int(rng.integers(COST_LOW, COST_HIGH + 1))
        b_cost = int(rng.integers(COST_LOW, COST_HIGH + 1))
        if a_cost == b_cost:
            continue
        if force_b_cheaper is True and not (b_cost < a_cost):
            continue
        if force_b_cheaper is False and not (b_cost > a_cost):
            continue
        return a_cost, b_cost


def _draw_bit(rng) -> str:
    return "YES" if int(rng.integers(0, 2)) == 1 else "NO"


def generate_episode(seed: int, subclass: str, index: int) -> Episode:
    """Generate one deterministic episode of `subclass` (same seed+index+subclass -> identical episode)."""
    if subclass not in _RULES:
        raise ValueError(f"unknown static_hidden subclass {subclass!r}; expected one of {SUBCLASSES}")
    rule = _RULES[subclass]
    rng = rng_for(seed, "static_hidden", subclass, index)

    a_cost, b_cost = _draw_costs(rng, rule.force_b_cheaper)
    b_cheaper = b_cost < a_cost
    true_bits = {v: _draw_bit(rng) for v in rule.rule_vars}
    extra_bits = {v: _draw_bit(rng) for v in rule.extra_hidden_vars}
    if subclass == "redundant_hidden":
        extra_bits["h2"] = true_bits["h1"]  # hard-wired copy, per _RULES doc

    visible_state: dict[str, Any] = {"A.cost": a_cost, "B.cost": b_cost}
    for i in range(rule.n_distractor_visible):
        # Deterministic decorative distractors: not referenced by any rule,
        # not fed into any head's target, present purely to test that the
        # generator's own ground truth (and later, the model) ignores them.
        visible_state[["C.cost", "A.weight"][i]] = int(rng.integers(COST_LOW, COST_HIGH + 1))

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


def generate_batch(seed: int, subclass: str, n_items: int) -> list[Episode]:
    return [generate_episode(seed, subclass, i) for i in range(n_items)]


def render_text(episode: Episode) -> str:
    """Render the spec section 5.1 `TASK STATIC-CHOICE` text block for an episode.

    `B.requires = h<k>` lines reflect the rule's structure (which hidden
    variables it references), independent of whether this particular
    instance's costs make that hidden variable decision-relevant — matching
    the spec's own contrast example, where `B.requires = h3` is still shown
    even though `h3` turns out to be irrelevant once B is not the cheaper
    action.
    """
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
