"""Family E — Resource Allocation / Scheduling generator (M1 kickoff
section 7, cross-family transfer).

Purpose: does the HG representation transfer to a genuinely new semantic
domain, or does it only recognize the specific route-choice/cost-comparison
syntax the training families (A-D) happen to use? Deliberately reuses
`sequential_quotient.py`'s exact generative mechanism -- same Markov state
shape `Z_t=(r_t, f_t, u_t, n_t, h)`, same `true_quotient_state`/`metadata`
key names (`"r"`, `"f"`, `"h"`, `"u"`, `"n"`) so every existing Family-C
eval function (`quotient_probes.quotient_recovery`'s default
`RELEVANT_VARS`/`IRRELEVANT_VARS`, `ood_report.condition_matrix`) works on
Family E's episodes with zero var-name overrides -- rendered under
scheduling vocabulary (job urgency/capacity/site/weather instead of
abstract r/f/u/n/h), so a probe genuinely has to generalize past the
training families' exact *wording*.

Action names are **deliberately kept identical** to Family C's
(`ACT_A`/`ACT_B`/`ASK_h`), not renamed the way the visible-state vocabulary
is: `training.actions.ACTIONS` is the distortion/transition heads' fixed
output width (`HeadsConfig.n_actions`, kept in lockstep with it by every
HG config, see `training/actions.py`'s own docstring), and
`serialize_episode`/`targets.distortion_target` compute those targets
*unconditionally* for any episode passed through them -- including via
`extract_latents`, the eval-only path this generator is meant for, not
just training. A first attempt at renamed actions (`ALLOCATE_A`/
`ALLOCATE_B`/`ASK_capacity`) broke exactly that path (`action_id` raising
on an out-of-vocabulary name), caught by actually running a real eval
smoke test, not by a unit test -- reusing the existing action names
sidesteps the coupling entirely, since `quotient_recovery` (what this
generator is actually evaluated with) never looks at actions at all.

This is a disclosed simplification, not hidden: kickoff section 7's own
framing is "test whether HG heads generalize to a new semantic domain
rather than simply recognizing route-choice syntax," not a requirement
that the generative *mechanism* itself be structurally novel. Reusing an
already-validated mechanism (Family C's memory-gap fix, its growing-context
disclosure, its exact-transition ground truth) under new vocabulary is the
low-risk way to satisfy that framing without inventing and re-validating a
second sequential mechanism from scratch.

**Never included in training** (see `hg_native/eval` wiring in
`run_eval.py`): Family E does not appear in `training.dataset.SyntheticMix`
or `training.ordinary_lm`'s text mix, so a checkpoint has literally zero
exposure to it in any form -- a deliberately *stronger* transfer test than
kickoff section 7's stated minimum (which allows ordinary-LM-only
exposure to the withheld family). Any zero-shot probe success on Family E
is therefore unconfounded transfer, not leaked familiarity.
"""

from __future__ import annotations

from typing import Any

from .schema import Episode, rng_for


def _draw_bit(rng) -> str:
    return "YES" if int(rng.integers(0, 2)) == 1 else "NO"


def _render_turn(t: int, visible_state: dict[str, Any], capacity_display: str) -> str:
    facts = " ".join(f"{key}={value}" for key, value in visible_state.items())
    return f"JOB{t}: {facts} capacity={capacity_display}"


def generate_trajectory(seed: int, index: int, n_steps: int) -> list[Episode]:
    """Generate one deterministic length-`n_steps` trajectory -- same
    determinism/prefix guarantees as `sequential_quotient.generate_trajectory`,
    same growing-context disclosure mechanism (turn 0's `ASK_h` is followed
    by a disclosed `ANSWER=capacity:<value>` line kept in every later
    turn's context)."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    rng = rng_for(seed, "resource_allocation", index)
    sequential_state_id = f"seqE-{seed}-{index}"

    h = _draw_bit(rng)  # capacity_ok: hidden, reusable once grounded
    u = _draw_bit(rng)  # priority_site: persistent irrelevant
    r = _draw_bit(rng)  # urgent: r_0 has no prior f to inherit from

    episodes: list[Episode] = []
    history_lines: list[str] = []
    for t in range(n_steps):
        f = _draw_bit(rng)  # future_urgent: becomes r_{t+1}
        n = _draw_bit(rng)  # weather: irrelevant noise, fresh each step

        ground_truth_decision = "ACT_B" if (r == "YES" and h == "YES") else "ACT_A"
        grounded = t > 0
        needs_grounding = not grounded
        correct_action = "ASK_h" if needs_grounding else ground_truth_decision

        visible_state: dict[str, Any] = {"urgent": r, "future_urgent": f, "priority_site": u, "weather": n}
        hidden_state = {"capacity": "UNKNOWN"}
        full_state = {**visible_state, "capacity": h}

        turn_line = _render_turn(t, visible_state, "UNKNOWN")
        prompt_text = "\n".join([*history_lines, turn_line, "ALLOCATE OR REQUEST CAPACITY INFO."])

        available_actions = ["ACT_A", "ACT_B", "ASK_h"]
        distortion_profile = {
            "ACT_A": 0.0 if ground_truth_decision == "ACT_A" else 1.0,
            "ACT_B": 0.0 if ground_truth_decision == "ACT_B" else 1.0,
            "ASK_h": 0.0,
        }

        episodes.append(
            Episode(
                episode_id=f"{sequential_state_id}-t{t}",
                family="resource_allocation",
                visible_state=visible_state,
                hidden_state=hidden_state,
                full_state=full_state,
                available_actions=available_actions,
                correct_action=correct_action,
                distortion_profile=distortion_profile,
                needs_grounding=needs_grounding,
                min_grounding_bits=1.0 if needs_grounding else 0.0,
                minimal_query_set=["capacity"] if needs_grounding else [],
                sequential_state_id=sequential_state_id,
                # Key names deliberately match sequential_quotient.py's
                # convention (not the display vocabulary above) so every
                # existing quotient-probe/condition-matrix function works
                # on Family E unmodified -- see module docstring.
                true_quotient_state={"r": r, "f": f, "h": h},
                metadata={
                    "t": t,
                    "ground_truth_decision": ground_truth_decision,
                    "grounded": grounded,
                    "u": u,
                    "n": n,
                    "prompt_text": prompt_text,
                },
            )
        )

        outcome = f"{turn_line} -> ACTION={correct_action}"
        if correct_action == "ASK_h":
            outcome += f" ANSWER=capacity:{h}"
        history_lines.append(outcome)

        r = f

    return episodes


def generate_batch(seed: int, n_trajectories: int, n_steps: int) -> list[list[Episode]]:
    return [generate_trajectory(seed, i, n_steps) for i in range(n_trajectories)]


def render_text(episode: Episode) -> str:
    return episode.metadata["prompt_text"]
