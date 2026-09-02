"""Tiny boolean-expression AST -- M1 kickoff section 6.2 structural OOD.

Only what `static_hidden.py`'s structural-OOD path needs: named boolean
variables plus AND/OR/NOT, evaluated against a `bits` dict of `"YES"`/`"NO"`
strings (matching every generator's boolean convention -- no Python `bool`
leaks into `Episode` fields).

`.signature()` is deliberately a *structural* identity, not a logical one:
`Or((Var("h1"), Var("h2")))` and `Not(And((Not(Var("h1")), Not(Var("h2")))))`
always agree on truth value (De Morgan's) but have different signatures.
That's exactly what kickoff section 6.2 means by "composition" -- "the
component primitives may be seen in training, but the exact composition
must be unseen" is a syntactic claim about the AST shape, not a semantic
one about the boolean function it computes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Var:
    name: str

    def evaluate(self, bits: dict[str, str]) -> bool:
        return bits[self.name] == "YES"

    def free_vars(self) -> frozenset[str]:
        return frozenset({self.name})

    def signature(self) -> tuple:
        return ("Var", self.name)


@dataclass(frozen=True)
class Not:
    expr: "Expr"

    def evaluate(self, bits: dict[str, str]) -> bool:
        return not self.expr.evaluate(bits)

    def free_vars(self) -> frozenset[str]:
        return self.expr.free_vars()

    def signature(self) -> tuple:
        return ("Not", self.expr.signature())


@dataclass(frozen=True)
class And:
    exprs: tuple["Expr", ...]

    def evaluate(self, bits: dict[str, str]) -> bool:
        return all(e.evaluate(bits) for e in self.exprs)

    def free_vars(self) -> frozenset[str]:
        return frozenset().union(*(e.free_vars() for e in self.exprs)) if self.exprs else frozenset()

    def signature(self) -> tuple:
        return ("And", tuple(e.signature() for e in self.exprs))


@dataclass(frozen=True)
class Or:
    exprs: tuple["Expr", ...]

    def evaluate(self, bits: dict[str, str]) -> bool:
        return any(e.evaluate(bits) for e in self.exprs)

    def free_vars(self) -> frozenset[str]:
        return frozenset().union(*(e.free_vars() for e in self.exprs)) if self.exprs else frozenset()

    def signature(self) -> tuple:
        return ("Or", tuple(e.signature() for e in self.exprs))


Expr = Union[Var, Not, And, Or]


def render(expr: Expr) -> str:
    """Human-readable rendering, for `static_hidden`'s structural-OOD `rule_text`."""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Not):
        return f"NOT {render(expr.expr)}"
    if isinstance(expr, And):
        return "(" + " AND ".join(render(e) for e in expr.exprs) + ")"
    if isinstance(expr, Or):
        return "(" + " OR ".join(render(e) for e in expr.exprs) + ")"
    raise TypeError(f"unknown Expr node: {expr!r}")
