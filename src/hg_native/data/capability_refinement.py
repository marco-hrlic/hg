"""Family D — Capability Refinement generator (M0 spec section 5.4; reworked
for the M1 kickoff section 8 design, 2026-09-02).

Purpose: does the learned decision representation refine when the action
frontier changes? State `Z=(r,f,u,n,v)` (kickoff section 8.1's own names):

- `r` (current relevant): under capability `C1={ACT_A,ACT_B}`, `correct_action`
  is `ACT_B` if `r=="YES"` else `ACT_A` -- `r` drives *this* step's decision.
- `f` (future relevant): drives the *next* step's `r`, via the same exact
  deterministic transition Family C uses (`r_{t+1} = f_t`) -- this is what
  makes "recovers future-relevant state" checkable rather than a vibe, and
  is the only reason this generator produces a 2-step trajectory (`t=0,1`)
  per member instead of one isolated decision: without a real next step,
  `f` would be an unused free draw a probe could never meaningfully be
  said to "recover."
- `u` (persistent irrelevant): drawn once per member, constant across `t`
  and across `C1`/`C2` -- the H-M1-5 control: it must never affect
  `correct_action` under either capability, and a probe's `Delta_refine(u)`
  should stay near zero going from `C1` to `C2`.
- `n` (irrelevant noise): freshly random every step, never used.
- `v` (capability-refined): irrelevant under `C1` (never touches
  `correct_action`/`distortion_profile`); under `C2={ACT_A,ACT_B,ACT_C}`,
  `v=="YES"` makes `ACT_C` strictly correct (regardless of `r`), while
  `v=="NO"` leaves `C2`'s correct action identical to `C1`'s. `v` is
  otherwise identical in distribution between capabilities and between `z`
  and `z'` except for the deliberate `z`/`z'` split below -- kickoff section
  8.4's "identical `v` distributions under `C1`/`C2`, paired counterfactual
  states" control.

Each call generates one **pair** of underlying members `z` (`v="YES"`),
`z'` (`v="NO"`), sharing the same `r,f,u,n` draws by construction -- i.e.
`z` and `z'` differ *only* in `v`, the exact paired counterfactual kickoff
section 8.1 asks for. Under `C1` both members get the same `correct_action`
(`z ~_C1 z'`, since `v` is irrelevant there); under `C2` they diverge
whenever `r`'s own `C1` action isn't already `ACT_C` (impossible, since
`ACT_C` isn't in `C1`'s action set) -- `z !~_C2 z'` always holds, since `z`'s
`C2` action is always `ACT_C` and `z'`'s never is.

Each member x capability combination is rendered at both `t=0` and `t=1`
(`r_1 = f_0`, `u`/`v` held constant across `t`, `f_1`/`n` drawn fresh) --
8 episodes per pair total, sharing one `sequential_state_id` per
member (distinct between `z` and `z'`, since they are different
trajectories). Zero information deficit throughout (`hidden_state={}`,
`needs_grounding=False`) -- every one of `r/f/u/n/v` is directly visible in
`visible_state`; this family is entirely about *quotient compression*
(what a probe can recover from the compact decision state), not grounding.
"""

from __future__ import annotations

from typing import Any

from .schema import Episode, rng_for

MEMBERS = ("z", "z_prime")
CAPABILITIES = ("C1", "C2")
STEPS = (0, 1)


def _draw_bit(rng) -> str:
    return "YES" if int(rng.integers(0, 2)) == 1 else "NO"


def _decide_c1(r: str) -> str:
    return "ACT_B" if r == "YES" else "ACT_A"


def _decide_c2(r: str, v: str) -> str:
    return "ACT_C" if v == "YES" else _decide_c1(r)


def _build_episode(pair_id: str, member: str, capability: str, t: int, r: str, f: str, u: str, n: str, v: str) -> Episode:
    visible_state: dict[str, Any] = {"r": r, "f": f, "u": u, "n": n, "v": v}
    if capability == "C1":
        available_actions = ["ACT_A", "ACT_B"]
        correct_action = _decide_c1(r)
    else:
        available_actions = ["ACT_A", "ACT_B", "ACT_C"]
        correct_action = _decide_c2(r, v)
    distortion_profile = {a: (0.0 if a == correct_action else 1.0) for a in available_actions}

    return Episode(
        episode_id=f"capability_refinement-{pair_id}-{member}-{capability}-t{t}",
        family="capability_refinement",
        visible_state=visible_state,
        hidden_state={},
        full_state=dict(visible_state),
        available_actions=available_actions,
        correct_action=correct_action,
        distortion_profile=distortion_profile,
        needs_grounding=False,
        min_grounding_bits=0.0,
        minimal_query_set=[],
        sequential_state_id=f"capD-{pair_id}-{member}",
        true_quotient_state={"r": r, "f": f, "u": u, "n": n, "v": v},
        metadata={
            "pair_id": pair_id,
            "member": member,
            "capability": capability,
            "t": t,
        },
    )


def generate_pair(seed: int, index: int) -> dict[str, Episode]:
    """Generate the 8 episodes for one (z, z') pair across both capabilities
    and both timesteps. Keys: `"{member}_t{t}_{capability}"`, e.g. `"z_t0_C1"`."""
    rng = rng_for(seed, "capability_refinement", index)
    r0 = _draw_bit(rng)
    f0 = _draw_bit(rng)
    u = _draw_bit(rng)
    n0 = _draw_bit(rng)
    r1 = f0  # the exact transition: this step's future-relevant becomes next step's current-relevant
    f1 = _draw_bit(rng)  # drawn for parity with r/f always co-occurring; no t=2 exists to consume it
    n1 = _draw_bit(rng)

    pair_id = f"pair-{seed}-{index}"
    episodes: dict[str, Episode] = {}
    for member, v in (("z", "YES"), ("z_prime", "NO")):
        for capability in CAPABILITIES:
            episodes[f"{member}_t0_{capability}"] = _build_episode(pair_id, member, capability, 0, r0, f0, u, n0, v)
            episodes[f"{member}_t1_{capability}"] = _build_episode(pair_id, member, capability, 1, r1, f1, u, n1, v)
    return episodes


def generate_batch(seed: int, n_pairs: int) -> list[dict[str, Episode]]:
    return [generate_pair(seed, i) for i in range(n_pairs)]


def render_text(episode: Episode) -> str:
    capability = episode.metadata["capability"]
    t = episode.metadata["t"]
    facts = " ".join(f"{key}={value}" for key, value in episode.visible_state.items())
    return f"T{t} CAPABILITY={capability}: {facts}\nDECIDE."
