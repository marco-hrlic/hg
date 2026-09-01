"""Query synthesis: deciding what to ask.

Plan section 15 is explicit that this does not start as a trained generative
model. It starts as a structured representation (`QuerySpec`) with an
algorithmically-produced optimum, so that:

* the gold query exists before any model tries to produce one, and
* an LM-generated query is scored against that gold standard rather than
  against a human's impression of whether it looked sensible.

A synthesizer therefore returns a `QuerySpec`, and every quality judgement is
made on what the query does to the decision problem (plan section 16), not on
its wording.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..grounding.channels import QuerySpec


@runtime_checkable
class QuerySynthesizer(Protocol):
    """Turns an item plus the asker's own state into one query."""

    name: str

    def synthesize(self, item: Any) -> QuerySpec: ...
