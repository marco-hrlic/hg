"""Boolean-expression AST -- exhaustive on small cases."""

from __future__ import annotations

from hg_native.data.rule_expr import And, Not, Or, Var, render

ALL_BITS = [{"h1": h1, "h2": h2} for h1 in ("YES", "NO") for h2 in ("YES", "NO")]


def test_var_evaluate():
    v = Var("h1")
    assert v.evaluate({"h1": "YES"}) is True
    assert v.evaluate({"h1": "NO"}) is False


def test_not_evaluate():
    expr = Not(Var("h1"))
    for bits in ALL_BITS:
        assert expr.evaluate(bits) == (bits["h1"] == "NO")


def test_and_evaluate_exhaustive():
    expr = And((Var("h1"), Var("h2")))
    for bits in ALL_BITS:
        assert expr.evaluate(bits) == (bits["h1"] == "YES" and bits["h2"] == "YES")


def test_or_evaluate_exhaustive():
    expr = Or((Var("h1"), Var("h2")))
    for bits in ALL_BITS:
        assert expr.evaluate(bits) == (bits["h1"] == "YES" or bits["h2"] == "YES")


def test_nested_expression_exhaustive():
    # (h1) OR (h2 AND NOT h1) -- the kickoff's own structural-OOD shape.
    expr = Or((And((Var("h1"),)), And((Var("h2"), Not(Var("h1"))))))
    for bits in ALL_BITS:
        expected = (bits["h1"] == "YES") or (bits["h2"] == "YES" and bits["h1"] == "NO")
        assert expr.evaluate(bits) == expected


def test_free_vars():
    assert Var("h1").free_vars() == {"h1"}
    assert And((Var("h1"), Var("h2"))).free_vars() == {"h1", "h2"}
    assert Or((Var("h1"), Not(Var("h1")))).free_vars() == {"h1"}


def test_signature_is_structural_not_logical():
    # h1 OR h2  vs.  NOT(NOT h1 AND NOT h2): logically equivalent (De
    # Morgan's), must have different signatures.
    a = Or((Var("h1"), Var("h2")))
    b = Not(And((Not(Var("h1")), Not(Var("h2")))))
    for bits in ALL_BITS:
        assert a.evaluate(bits) == b.evaluate(bits)
    assert a.signature() != b.signature()


def test_signature_equal_for_structurally_identical_expressions():
    assert And((Var("h1"), Var("h2"))).signature() == And((Var("h1"), Var("h2"))).signature()


def test_render_produces_readable_text():
    expr = Or((And((Var("h1"),)), And((Var("h2"), Not(Var("h1"))))))
    text = render(expr)
    assert "h1" in text and "h2" in text
    assert "OR" in text and "AND" in text and "NOT" in text
