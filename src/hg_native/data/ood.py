"""OOD split machinery -- M1 kickoff section 6.

Mirrors the sibling `hg.diversity` package's philosophy (`hg/diversity/splits.py`'s
assertive `SplitPlan.validate()`, `hg/diversity/overlap.py`'s "measure it,
don't assume it" `overlap_report`): a genuine leakage condition (shared
sentence templates between train and lexical-OOD, an overlapping numeric
range) is checked structurally and raises, rather than trusted by
convention.

Coverage so far, one family at a time (see each family's section below for
what's built and what's deliberately not, and why):

- **Family A** (`static_hidden.py`): all 3 built axes -- lexical (6.1),
  parameter (6.3), distractor-density (6.4).
- **Family B** (`accessibility.py`): parameter (6.3) only.
- **Family C** (`sequential_quotient.py`): parameter (6.3) only.
- **Family D** (`capability_refinement.py`): none yet -- no numeric-range
  knob exists at all (`Z=(r,f,u,n,v)` is all booleans), so none of the 3
  built axes apply without a structural change to that generator, which is
  a follow-up slice, not a fit to force here.

Structural OOD (kickoff section 6.2) is deliberately not covered for any
family -- it needs a rule-composition DSL none of the generators have
(`static_hidden._decide` is a fixed AND-only conjunction).
"""

from __future__ import annotations

from collections import Counter

from hg_native.data import accessibility, sequential_quotient, static_hidden
from hg_native.data.schema import Episode, rng_for

IID_COST_RANGE = (static_hidden.COST_LOW, static_hidden.COST_HIGH)
PARAMETER_OOD_COST_RANGE = (100, 199)
DISTRACTOR_OOD_RANGE = (10, 30)  # kickoff section 6.4's own example band

# Family B (accessibility.py) gets parameter OOD only. Its 8 subclasses
# already render genuinely different *information layouts* (the family's
# actual purpose), not different *vocabulary* for the same content the way
# kickoff section 6.1's "Route A costs..." vs. "Option red incurs..."
# example intends -- several subclasses share literal template lines
# (`f"A.base = {c.a_base}"` appears verbatim in `reordered_form`,
# `irrelevant_constraints_block`, `redundant_restatement`, and
# `misleading_distractor_scaffold`), and `BASE_LOW..BASE_HIGH` (10-50) is
# small enough that `assert_no_shared_lexical_templates`'s recurrence
# heuristic would flag ordinary numeric coincidence as "leakage" rather
# than real template reuse. Forcing a lexical-OOD split onto this family's
# subclass axis would misrepresent section 6.1 and risk a flaky test, so
# it's skipped rather than shoehorned in. Distractor-density doesn't apply
# either: Family B's state is fixed at exactly 2 components, with no
# "irrelevant visible variable count" axis the way `static_hidden`'s
# decorative distractors or Family C/D's `u`/`n` have.
ACCESSIBILITY_IID_BASE_RANGE = (accessibility.BASE_LOW, accessibility.BASE_HIGH)
ACCESSIBILITY_IID_ADJ_RANGE = (accessibility.ADJ_LOW, accessibility.ADJ_HIGH)
ACCESSIBILITY_PARAMETER_OOD_BASE_RANGE = (100, 200)
ACCESSIBILITY_PARAMETER_OOD_ADJ_RANGE = (50, 100)

# Family C (sequential_quotient.py) gets parameter OOD only, via
# `generate_trajectory`'s existing `n_steps` argument -- unlike Family A/B,
# no new parameter needed to be added to the generator itself, and these
# ranges are the kickoff's own section 6.3 worked example verbatim
# ("training path lengths: 2-6, test path lengths: 7-10"). The IID range
# also covers `training/dataset.py`'s existing `n_steps_c=4` default.
# Lexical OOD doesn't fit: `render_text` returns
# `metadata["prompt_text"]`, built once at generation time specifically to
# carry the growing-context `ANSWER=h:<value>` disclosure the Family-C
# memory-gap fix depends on (see that module's docstring) -- swapping in
# alternate wording there risks breaking that disclosure mechanism, a
# materially bigger change than a template swap. Distractor-density
# doesn't fit either: `u`/`n` are each a single field, not a count.
SEQUENTIAL_QUOTIENT_IID_N_STEPS_RANGE = (2, 6)
SEQUENTIAL_QUOTIENT_PARAMETER_OOD_N_STEPS_RANGE = (7, 10)


def _tag(episodes: list[Episode], condition: str) -> list[Episode]:
    for ep in episodes:
        ep.metadata["ood_condition"] = condition
    return episodes


def iid_pool(seed: int, subclass: str, n_items: int) -> list[Episode]:
    return _tag(static_hidden.generate_batch(seed, subclass, n_items), "iid")


def lexical_ood_pool(seed: int, subclass: str, n_items: int) -> list[Episode]:
    """Same generator, same ground truth -- only `render_text`'s caller needs
    to pass `surface_form="lexical_ood"`; the tag here just marks the pool so
    an eval driver can route to that surface form and report results
    separately from IID."""
    return _tag(static_hidden.generate_batch(seed, subclass, n_items), "lexical")


def parameter_ood_pool(seed: int, subclass: str, n_items: int, cost_range: tuple[int, int] = PARAMETER_OOD_COST_RANGE) -> list[Episode]:
    assert_disjoint_ranges(IID_COST_RANGE, cost_range)
    return _tag(static_hidden.generate_batch(seed, subclass, n_items, cost_range=cost_range), "parameter")


def distractor_ood_pool(
    seed: int, subclass: str, n_items: int, n_distractor_visible_range: tuple[int, int] = DISTRACTOR_OOD_RANGE
) -> list[Episode]:
    """One draw of `n_distractor_visible` per episode, uniform over the
    (inclusive) `n_distractor_visible_range` -- not a single fixed count, so
    the OOD pool actually spans the kickoff's "10-30" band rather than
    pinning one arbitrary point in it. The count draw uses its own RNG
    stream (keyed separately from `static_hidden`'s internal `rng_for`
    calls), so it doesn't perturb the costs/bits a given `(seed, index)`
    would otherwise draw."""
    lo, hi = n_distractor_visible_range
    episodes = []
    for i in range(n_items):
        count_rng = rng_for(seed, "ood_distractor_count", subclass, i)
        n_distractor_visible = int(count_rng.integers(lo, hi + 1))
        episodes.append(static_hidden.generate_episode(seed, subclass, i, n_distractor_visible=n_distractor_visible))
    return _tag(episodes, "distractor")


def accessibility_iid_pool(seed: int, subclass: str, n_items: int) -> list[Episode]:
    return _tag(accessibility.generate_batch(seed, subclass, n_items), "iid")


def accessibility_parameter_ood_pool(
    seed: int,
    subclass: str,
    n_items: int,
    base_range: tuple[int, int] = ACCESSIBILITY_PARAMETER_OOD_BASE_RANGE,
    adj_range: tuple[int, int] = ACCESSIBILITY_PARAMETER_OOD_ADJ_RANGE,
) -> list[Episode]:
    assert_disjoint_ranges(ACCESSIBILITY_IID_BASE_RANGE, base_range)
    assert_disjoint_ranges(ACCESSIBILITY_IID_ADJ_RANGE, adj_range)
    episodes = accessibility.generate_batch(seed, subclass, n_items, base_range=base_range, adj_range=adj_range)
    return _tag(episodes, "parameter")


def _draw_n_steps(seed: int, index: int, n_steps_range: tuple[int, int]) -> int:
    # Own RNG stream, keyed separately from sequential_quotient's internal
    # rng_for calls, so the draw doesn't perturb a given (seed, index)'s
    # own r/f/u/n/h draws -- same convention as static_hidden's
    # ood_distractor_count stream above.
    lo, hi = n_steps_range
    count_rng = rng_for(seed, "ood_n_steps", index)
    return int(count_rng.integers(lo, hi + 1))


def sequential_quotient_iid_pool(
    seed: int, n_trajectories: int, n_steps_range: tuple[int, int] = SEQUENTIAL_QUOTIENT_IID_N_STEPS_RANGE
) -> list[Episode]:
    episodes = []
    for i in range(n_trajectories):
        n_steps = _draw_n_steps(seed, i, n_steps_range)
        episodes += sequential_quotient.generate_trajectory(seed, i, n_steps)
    return _tag(episodes, "iid")


def sequential_quotient_parameter_ood_pool(
    seed: int,
    n_trajectories: int,
    n_steps_range: tuple[int, int] = SEQUENTIAL_QUOTIENT_PARAMETER_OOD_N_STEPS_RANGE,
) -> list[Episode]:
    assert_disjoint_ranges(SEQUENTIAL_QUOTIENT_IID_N_STEPS_RANGE, n_steps_range)
    episodes = []
    for i in range(n_trajectories):
        n_steps = _draw_n_steps(seed, i, n_steps_range)
        episodes += sequential_quotient.generate_trajectory(seed, i, n_steps)
    return _tag(episodes, "parameter")


def assert_no_shared_lexical_templates(
    train_texts: list[str], ood_texts: list[str], allowed_shared: frozenset[str] = frozenset()
) -> None:
    """Raise if any fixed template fragment (a whole rendered line, minus
    its data payload) appears in both pools. Compares whole lines rather
    than individual words: `render_text`'s per-episode data lines (e.g.
    `"A.cost = 37"`) differ incidentally line-by-line anyway, so the
    meaningful leakage check is on the fixed scaffold lines
    (`"TASK STATIC-CHOICE"`, `"RULE:"`, ...) that don't depend on episode
    data at all -- exactly what would leak if an OOD pool accidentally
    reused a training template.

    `allowed_shared` carves out lines that are expected to recur across
    surface forms on purpose -- e.g. `render_text`'s own `rule_text` line,
    the substantive statement of the decision rule rather than decorative
    wording (see `static_hidden.render_text`'s module comment). Pass the
    actual `metadata["rule_text"]` values in use, not a blanket bypass."""
    # A line only counts as leakage if it recurs verbatim across *multiple*
    # different episodes within *each* pool -- i.e. it's part of the fixed
    # scaffold, not a coincidental data line (e.g. two different episodes
    # both happening to show "A.cost = 37"). Data lines are, by construction,
    # keyed off per-episode random draws and vanishingly unlikely to recur
    # verbatim across many episodes; scaffold lines recur in *every* episode
    # of their own pool, so intersecting the two pools' "recurs >1x" sets is
    # exactly the shared-scaffold check.
    def _recurring(texts: list[str]) -> set[str]:
        counts = Counter(line for text in texts for line in text.splitlines())
        return {line for line, n in counts.items() if n > 1}

    scaffold_leak = (_recurring(train_texts) & _recurring(ood_texts)) - allowed_shared
    if scaffold_leak:
        raise ValueError(f"shared lexical templates between train and OOD pools: {sorted(scaffold_leak)}")


def assert_disjoint_ranges(train_range: tuple[int, int], ood_range: tuple[int, int]) -> None:
    """Raise if the two inclusive `(low, high)` ranges overlap at all."""
    t_lo, t_hi = train_range
    o_lo, o_hi = ood_range
    if t_lo <= o_hi and o_lo <= t_hi:
        raise ValueError(f"train range {train_range} overlaps OOD range {ood_range}")
