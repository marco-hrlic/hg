"""Source snapshots: independently authored documents and read-only Git code.

A source is *data*, never instructions. Every source has a stable ``source_id``,
a ``version`` label and a ``sha256`` of its exact content; citations are
validated against these. Git sources record the immutable commit SHA; code is
read with ``git show`` and is never written back.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SOURCE_TYPES = ("stakeholder_notes", "requirements", "architecture", "decision", "code", "test")

# Authority ordering for *intent*. Code and tests are authoritative for what the
# implementation does, not for what the business wants.
INTENT_AUTHORITY = {"stakeholder_notes": 3, "decision": 3, "requirements": 3, "architecture": 2,
                    "test": 1, "code": 1}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class SourceDocument:
    source_id: str  # e.g. "toolshare:spec/requirements.md" or "git:toolshare@<sha>:src/x.py"
    project: str
    path: str
    source_type: str
    version: str  # document version label or git commit SHA
    content: str
    origin: str = "file"  # "file" | "git"
    repo: str | None = None
    commit: str | None = None

    @property
    def sha256(self) -> str:
        return sha256_text(self.content)

    @property
    def lines(self) -> list[str]:
        return self.content.splitlines()

    def span(self, start: int, end: int) -> str | None:
        """1-based inclusive line span, or None if out of range."""
        ls = self.lines
        if start < 1 or end < start or end > len(ls):
            return None
        return "\n".join(ls[start - 1:end])

    def meta(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "content"}
        d["sha256"] = self.sha256
        d["line_count"] = len(self.lines)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class SourceBundle:
    """The bounded set of sources supplied for one step."""

    project: str
    documents: list[SourceDocument] = field(default_factory=list)
    diffs: list[dict[str, Any]] = field(default_factory=list)  # {"source_id","from","to","text"}

    def get(self, source_id: str) -> SourceDocument | None:
        for d in self.documents:
            if d.source_id == source_id:
                return d
        return None

    def ids(self) -> list[str]:
        return [d.source_id for d in self.documents]

    def manifest(self) -> list[dict[str, Any]]:
        return [d.meta() for d in self.documents]

    def fingerprint(self) -> str:
        return sha256_text(json.dumps([self.manifest(), self.diffs], sort_keys=True))


class SourceCatalog:
    """Every source version ever supplied, so that stale citations are recognisable."""

    def __init__(self) -> None:
        self._by_id: dict[str, dict[str, SourceDocument]] = {}

    def register(self, docs: Iterable[SourceDocument]) -> None:
        for d in docs:
            self._by_id.setdefault(d.source_id, {})[d.sha256] = d

    def versions(self, source_id: str) -> dict[str, SourceDocument]:
        return dict(self._by_id.get(source_id, {}))

    def known(self, source_id: str, sha: str) -> SourceDocument | None:
        return self._by_id.get(source_id, {}).get(sha)

    def to_list(self) -> list[dict[str, Any]]:
        return [d.meta() for sid in sorted(self._by_id) for _, d in sorted(self._by_id[sid].items())]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_file_source(root: Path, project: str, rel_path: str, source_type: str, version: str,
                     logical_path: str | None = None) -> SourceDocument:
    """Load a document from disk. ``logical_path`` lets several versions share one source_id."""
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unknown source type {source_type!r}")
    content = (root / rel_path).read_text(encoding="utf-8")
    lp = logical_path or rel_path
    return SourceDocument(source_id=f"{project}:{lp}", project=project, path=lp, source_type=source_type,
                          version=version, content=content)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


def git_head(repo: Path, ref: str = "HEAD") -> str:
    return _git(repo, "rev-parse", ref).strip()


def load_git_sources(repo: Path, repo_name: str, project: str, commit: str,
                     paths: list[str] | None = None) -> list[SourceDocument]:
    """Read files at an immutable commit (read-only; the working tree is never touched)."""
    sha = git_head(repo, commit)
    listed = _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
    out = []
    for p in sorted(listed if paths is None else [p for p in listed if p in paths]):
        content = _git(repo, "show", f"{sha}:{p}")
        stype = "test" if p.startswith("tests/") or "/test_" in p or p.startswith("test_") else "code"
        out.append(SourceDocument(source_id=f"git:{repo_name}@{sha}:{p}", project=project, path=p,
                                  source_type=stype, version=sha, content=content, origin="git",
                                  repo=repo_name, commit=sha))
    return out


def git_diff(repo: Path, repo_name: str, base: str, head: str) -> dict[str, Any]:
    b, h = git_head(repo, base), git_head(repo, head)
    text = _git(repo, "diff", "--no-color", "-U2", b, h)
    return {"source_id": f"git-diff:{repo_name}@{b}..{h}", "from": b, "to": h, "text": text}


# ---------------------------------------------------------------------------
# Deterministic input packer
# ---------------------------------------------------------------------------

def pack_sources(bundle: SourceBundle, max_chars_per_doc: int = 12000, max_total_chars: int = 60000) -> dict[str, Any]:
    """Render sources with line numbers, ids and hashes; disclose any truncation.

    Order is deterministic (bundle order). Truncation is at line granularity.
    """
    parts: list[str] = []
    truncation: list[dict[str, Any]] = []
    total = 0
    for d in bundle.documents:
        header = (f"=== SOURCE {d.source_id}\n"
                  f"type: {d.source_type} | version: {d.version} | sha256: {d.sha256}\n")
        body_lines, used = [], 0
        lines = d.lines
        shown = 0
        for i, line in enumerate(lines, 1):
            rendered = f"L{i:03d}| {line}\n"
            if used + len(rendered) > max_chars_per_doc or total + len(header) + used + len(rendered) > max_total_chars:
                break
            body_lines.append(rendered)
            used += len(rendered)
            shown = i
        if shown < len(lines):
            truncation.append({"source_id": d.source_id, "shown_lines": shown, "total_lines": len(lines)})
            body_lines.append(f"[TRUNCATED: showing lines 1-{shown} of {len(lines)}]\n")
        block = header + "".join(body_lines) + f"=== END SOURCE {d.source_id}\n"
        parts.append(block)
        total += len(block)
    for df in bundle.diffs:
        text = df["text"]
        if total + len(text) > max_total_chars:
            keep = max(0, max_total_chars - total)
            truncation.append({"source_id": df["source_id"], "shown_chars": keep, "total_chars": len(text)})
            text = text[:keep] + "\n[TRUNCATED DIFF]\n"
        block = f"=== DIFF {df['source_id']}\n{text}=== END DIFF\n"
        parts.append(block)
        total += len(block)
    text = "".join(parts)
    return {"text": text, "chars": len(text), "truncation": truncation, "sha256": sha256_text(text)}
