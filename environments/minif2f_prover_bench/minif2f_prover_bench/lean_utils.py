from __future__ import annotations

from typing import Tuple


def parse_theorem_signature(snippet: str) -> Tuple[str, str, str]:
    """Extract `(name, binders, goal)` from a Lean theorem header."""
    lines = [ln.strip() for ln in (snippet or "").splitlines() if ln.strip()]
    header_line = None
    acc: list[str] = []
    for ln in lines:
        acc.append(ln)
        if ":= by" in ln:
            header_line = " ".join(acc)
            break
    if header_line is None:
        raise ValueError("Missing ':= by' in theorem snippet")

    s = header_line
    for prefix in ("theorem ", "lemma ", "example ", "def ", "corollary "):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break

    left, _ = s.rsplit(":=", 1)
    name, remainder = (left.split(None, 1) + [""])[:2]
    binder_segment, goal = _split_binders_and_goal(remainder)
    return name.strip(), binder_segment.strip(), goal.strip()


def _split_binders_and_goal(remainder: str) -> tuple[str, str]:
    """Split the binder segment from the goal at the first ':' at depth 0."""
    depth = 0
    for idx, ch in enumerate(remainder):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth = max(depth - 1, 0)
        elif ch == ":" and depth == 0:
            return remainder[:idx].strip(), remainder[idx + 1 :].strip()
    raise ValueError("Missing ':' separating binders and goal")

