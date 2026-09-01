"""Family C — Sequential Known Quotient generator (spec section 5.3).

Purpose: dynamic decision-state learning and grounding reuse. One call
produces a whole trajectory (a list of `Episode`s sharing one
`sequential_state_id`), not a single item, because the reuse question --
"does the model stop re-asking for `h` once it has it?" -- is only
observable across a sequence.

Markov state `Z_t = (r_t, f_t, u_t, n_t, h)` (spec's own names):

- `r_t` (current relevant): drives *this* step's decision together with `h`.
- `f_t` (future relevant): drives the *next* step's `r`; the exact,
  deterministic transition `r_{t+1} = f_t` is what makes "recovers
  future-relevant state" a checkable claim rather than a vibe.
- `u_t` (persistent irrelevant): drawn once, constant for the whole
  trajectory, never used by any rule.
- `n_t` (irrelevant noise): freshly random every step, never used.
- `h` (hidden, reusable): drawn once per trajectory, constant, and required
  by every step's decision rule -- but only *unknown* at t=0. Spec section
  5.3: "if h remains valid and is retained, subsequent optimal grounding
  cost should be zero" -- this generator always keeps `h` valid for the
  trajectory's whole length, so `needs_grounding` is `True` at t=0 only,
  and `correct_action` at every later step assumes the idealized
  grounding-reuse policy (ground once, never re-ask), which is exactly the
  ground truth `R_repeat` (spec section 10.7) is measured against: a real
  model that keeps asking `ASK_h` after t=0 is the failure this exists to
  catch, and the generator does not need to reward that behavior for
  the metric to be meaningful.

`true_quotient_state` is `{"r": r_t, "f": f_t, "h": h}` per the spec's own
worked example (`S_t* = (r_t, f_t, h)`); `u_t`/`n_t` are still recorded in
`full_state` for the quotient-probe irrelevant-variable baselines (spec
section 10.6), just excluded from the quotient itself.
"""

from __future__ import annotations

from typing import Any

from .schema import Episode, rng_for


def _draw_bit(rng) -> str:
    return "YES" if int(rng.integers(0, 2)) == 1 else "NO"


def generate_trajectory(seed: int, index: int, n_steps: int) -> list[Episode]:
    """Generate one deterministic length-`n_steps` trajectory.

    Same (seed, index, n_steps) -> identical trajectory; a shorter prefix
    (same seed/index, smaller n_steps) is a strict prefix of a longer one,
    since the RNG advances step-by-step rather than being re-seeded per call.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    rng = rng_for(seed, "sequential_quotient", index)
    sequential_state_id = f"seqC-{seed}-{index}"

    h = _draw_bit(rng)
    u = _draw_bit(rng)
    r = _draw_bit(rng)  # r_0 has no prior f to inherit from; drawn fresh

    episodes: list[Episode] = []
    for t in range(n_steps):
        f = _draw_bit(rng)  # f_t: fresh each step; becomes r_{t+1}
        n = _draw_bit(rng)  # n_t: fresh noise, never used

        ground_truth_decision = "ACT_B" if (r == "YES" and h == "YES") else "ACT_A"
        grounded = t > 0  # h was asked at t=0 and (by construction) remains valid
        needs_grounding = not grounded
        correct_action = "ASK_h" if needs_grounding else ground_truth_decision

        visible_state: dict[str, Any] = {"r": r, "f": f, "u": u, "n": n}
        hidden_state = {"h": "UNKNOWN"}  # the environment never re-states h; retention is the model's job
        full_state = {**visible_state, "h": h}

        available_actions = ["ACT_A", "ACT_B", "ASK_h"]
        distortion_profile = {
            "ACT_A": 0.0 if ground_truth_decision == "ACT_A" else 1.0,
            "ACT_B": 0.0 if ground_truth_decision == "ACT_B" else 1.0,
            "ASK_h": 0.0,  # asking always leads to a correct subsequent decision
        }

        episodes.append(
            Episode(
                episode_id=f"{sequential_state_id}-t{t}",
                family="sequential_quotient",
                visible_state=visible_state,
                hidden_state=hidden_state,
                full_state=full_state,
                available_actions=available_actions,
                correct_action=correct_action,
                distortion_profile=distortion_profile,
                needs_grounding=needs_grounding,
                min_grounding_bits=1.0 if needs_grounding else 0.0,
                minimal_query_set=["h"] if needs_grounding else [],
                sequential_state_id=sequential_state_id,
                true_quotient_state={"r": r, "f": f, "h": h},
                metadata={
                    "t": t,
                    # n_steps deliberately not recorded: it would make a
                    # shorter trajectory's episodes differ from the matching
                    # prefix of a longer one generated from the same
                    # (seed, index), breaking that determinism guarantee for
                    # no benefit (len(trajectory) already gives it when the
                    # whole list is in hand).
                    "ground_truth_decision": ground_truth_decision,
                    "grounded": grounded,
                    "u": u,  # persistent irrelevant, recorded for the quotient-probe baseline
                    "n": n,  # random irrelevant, recorded for the quotient-probe baseline
                },
            )
        )
        r = f  # the exact transition: next step's current-relevant is this step's future-relevant

    return episodes


def generate_batch(seed: int, n_trajectories: int, n_steps: int) -> list[list[Episode]]:
    return [generate_trajectory(seed, i, n_steps) for i in range(n_trajectories)]
