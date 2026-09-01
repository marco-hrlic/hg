"""What the AI can ask, and what an answer costs to transmit.

Theory (Human Grounding v0.6, plan sections 1 and 16)
-----------------------------------------------------
For a decision variable J and AI-internal information M_A, the exact grounding
requirement is the residual decision deficit H(J | M_A). A one-way message from
an external source carries C_-> bits, and the *coordination overhead* is

    Omega = C_-> - H(J | M_A)

which is >= 0 by the source-coding bound. The architectural claim under test is
that a system which knows its own residual decision state can synthesize a
query whose answer costs far less to transmit than a generic dump of everything
the source knows -- i.e. that it can drive Omega toward zero.

That makes the *response alphabet* the object worth modelling, so a query here
carries its own response cost. Phase 0 uses the fixed-length upper bound

    B_raw = log2 |Y|

(plan section 16). This is an upper bound, not an entropy estimate: a real
source could compress a redundant answer. Empirical entropy is only reported by
the analysis where the alphabet is small enough to estimate it honestly.

Nothing in this module is Shannon-exact except in the synthetic setting of
Phase 0, where the joint distribution is constructed and therefore known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


class Route(Enum):
    """The Phase 0 route menu.

    Deliberately reduced (plan section 4.2 permits this). The full v0.6 menu is
    ACT / THINK / RETRIEVE / OBSERVE / ASK / DELIBERATE, and the ASK-vs-
    DELIBERATE distinction is load-bearing -- ASK assumes the information
    already exists on the source's side, DELIBERATE means the reference itself
    may have to be formed (plan section 20). Those routes are left undefined
    until an experiment measures them, so that nothing can collapse them by
    accident in the meantime.
    """

    ACT = "act"
    ASK_GENERIC = "ask_generic"
    ASK_TARGETED = "ask_targeted"

    def asks(self) -> bool:
        return self is not Route.ACT


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """A structured query, plus the cost of the answer it demands.

    Structured rather than generated (plan section 15): for a synthetic task the
    optimal QuerySpec can be produced algorithmically, which gives Phase 2 a
    gold standard to compare an LM-synthesized query against.

    `response_bits` is log2 of the response alphabet size and is the quantity
    every cost metric uses. `response_alphabet_size` is the exact |Y| when it is
    representable, and `candidate_values` enumerates the alphabet only when it
    is small -- for a generic query over n binary variables |Y| = 2**n, which is
    exact and cheap to store as an integer but absurd to enumerate.
    """

    variable_names: tuple[str, ...]
    response_bits: float
    natural_language_query: str
    response_alphabet_size: int | None = None
    candidate_values: tuple[str, ...] | None = None
    candidate_actions_affected: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.response_bits < 0:
            raise ValueError(f"response_bits must be >= 0, got {self.response_bits}")
        if (
            self.candidate_values is not None
            and self.response_alphabet_size is not None
            and len(self.candidate_values) != self.response_alphabet_size
        ):
            raise ValueError(
                f"{len(self.candidate_values)} candidate values but "
                f"response_alphabet_size={self.response_alphabet_size}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable_names": list(self.variable_names),
            "response_bits": self.response_bits,
            "natural_language_query": self.natural_language_query,
            "response_alphabet_size": self.response_alphabet_size,
            "candidate_values": (
                list(self.candidate_values) if self.candidate_values is not None else None
            ),
            "candidate_actions_affected": list(self.candidate_actions_affected),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QuerySpec":
        cv = d.get("candidate_values")
        return cls(
            variable_names=tuple(d["variable_names"]),
            response_bits=float(d["response_bits"]),
            natural_language_query=str(d["natural_language_query"]),
            response_alphabet_size=d.get("response_alphabet_size"),
            candidate_values=tuple(cv) if cv is not None else None,
            candidate_actions_affected=tuple(d.get("candidate_actions_affected") or ()),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class Response:
    """What the external source transmitted.

    `informative` records whether the source actually held the variables that
    were asked about. It is not cosmetic: the joint-sufficiency control (plan
    section 24) removes the source's knowledge of the queried variable, and the
    resulting episode must show cost paid with no distortion reduction. Without
    this flag that case is indistinguishable from a source that answered.
    """

    content: Any  # JSON-serializable
    raw_bits: float
    informative: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "raw_bits": self.raw_bits,
            "informative": self.informative,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Response":
        return cls(
            content=d["content"],
            raw_bits=float(d["raw_bits"]),
            informative=bool(d.get("informative", True)),
            metadata=dict(d.get("metadata") or {}),
        )


@runtime_checkable
class ExternalSource(Protocol):
    """Something holding private information M_H that answers queries.

    A simulated human in Phase 0; a scripted human or a second model later. The
    interface is one-shot on purpose: multi-turn interaction changes the
    communication-complexity accounting and belongs to a later phase.
    """

    def respond(self, query: QuerySpec) -> Response: ...


def log2_alphabet(size: int) -> float:
    """log2 |Y| for a response alphabet of `size` distinct answers."""
    if size < 1:
        raise ValueError(f"alphabet size must be >= 1, got {size}")
    return float(math.log2(size))


# --------------------------------------------------------------- bit source ---
def targeted_bit_query(index: int, n_bits: int) -> QuerySpec:
    """Ask for exactly the one variable the decision turns on.

    Response alphabet {0, 1}, so B_raw = 1 bit = H(J | M_A) for the index task.
    This is the analytic optimum and the gold standard Phase 2 compares against.
    """
    if not 0 <= index < n_bits:
        raise ValueError(f"index {index} out of range for {n_bits} bits")
    return QuerySpec(
        variable_names=(f"X{index}",),
        response_bits=log2_alphabet(2),
        response_alphabet_size=2,
        candidate_values=("0", "1"),
        candidate_actions_affected=("0", "1"),
        natural_language_query=f"What is the value of X{index}?",
        metadata={"kind": "targeted", "index": index, "n_bits": n_bits},
    )


def generic_bit_query(n_bits: int) -> QuerySpec:
    """Ask the source for everything it knows -- generic human escalation.

    Response alphabet is every n-bit string, so B_raw = n bits. The gap to the
    1-bit targeted query *is* the coordination overhead Omega = n - 1, and it is
    the whole point of Phase 0 that this gap is exact rather than estimated.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be >= 1, got {n_bits}")
    names = tuple(f"X{i}" for i in range(n_bits))
    return QuerySpec(
        variable_names=names,
        response_bits=float(n_bits),
        response_alphabet_size=2**n_bits,
        # Not enumerated: 2**n strings. The alphabet size above is exact.
        candidate_values=None,
        candidate_actions_affected=("0", "1"),
        natural_language_query=(
            "Tell me everything you know: report the values of "
            f"X0 through X{n_bits - 1}."
        ),
        metadata={"kind": "generic", "n_bits": n_bits},
    )


class BitVectorSource:
    """A source privately holding M_H = (X_0, ..., X_{n-1}).

    `known` optionally restricts which variables the source actually holds. For
    a variable it does not hold it answers with an independent coin flip rather
    than refusing, which is the harder and more realistic control: the AI pays
    the full transmission cost and receives a well-formed but useless answer.
    """

    def __init__(
        self,
        bits: Sequence[int],
        *,
        rng: np.random.Generator | None = None,
        known: Sequence[bool] | None = None,
    ):
        self.bits = tuple(int(b) for b in bits)
        if any(b not in (0, 1) for b in self.bits):
            raise ValueError("bits must be 0/1")
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.known = tuple(bool(k) for k in known) if known is not None else (True,) * len(self.bits)
        if len(self.known) != len(self.bits):
            raise ValueError("`known` must have one flag per bit")

    def _value(self, index: int) -> tuple[int, bool]:
        if self.known[index]:
            return self.bits[index], True
        return int(self.rng.integers(0, 2)), False

    def respond(self, query: QuerySpec) -> Response:
        indices = [_bit_index(name) for name in query.variable_names]
        values: list[int] = []
        informative = True
        for i in indices:
            v, known = self._value(i)
            values.append(v)
            informative = informative and known
        content = "".join(str(v) for v in values)
        return Response(
            content=content,
            raw_bits=query.response_bits,
            informative=informative,
            metadata={"n_variables": len(indices)},
        )


def _bit_index(name: str) -> int:
    if not name.startswith("X"):
        raise ValueError(f"not a bit variable name: {name!r}")
    return int(name[1:])
