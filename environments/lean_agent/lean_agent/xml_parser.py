"""XML tag parsing for lean_agent environment."""

from __future__ import annotations

import re


def _strip_markdown_fencing(code: str) -> str:
    """Strip markdown code fences if present."""
    code = code.strip()
    # Match ```lean4, ```lean, or just ```
    match = re.match(r'^```(?:lean4?|)\n?(.*?)```$', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code


def parse_lean_check(text: str) -> str | None:
    """Extract content from <lean_check>...</lean_check> tag."""
    if not text:
        return None
    # Try with closing tag
    match = re.search(r'<lean_check>(.*?)</lean_check>', text, re.DOTALL)
    if match:
        return _strip_markdown_fencing(match.group(1))
    # Try without closing tag (stopped by stop sequence)
    match = re.search(r'<lean_check>(.*?)$', text, re.DOTALL)
    if match and match.group(1).strip():
        return _strip_markdown_fencing(match.group(1))
    return None


def parse_prove(text: str) -> str | None:
    """Extract content from <prove>...</prove> tag."""
    if not text:
        return None
    # Try with closing tag
    match = re.search(r'<prove>(.*?)</prove>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try without closing tag (stopped by stop sequence)
    match = re.search(r'<prove>(.*?)$', text, re.DOTALL)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return None


def parse_submit(text: str) -> str | None:
    """Extract content from <submit>...</submit> tag."""
    if not text:
        return None
    # Try with closing tag
    match = re.search(r'<submit>(.*?)</submit>', text, re.DOTALL)
    if match:
        return _strip_markdown_fencing(match.group(1))
    # Try without closing tag (stopped by stop sequence)
    match = re.search(r'<submit>(.*?)$', text, re.DOTALL)
    if match and match.group(1).strip():
        return _strip_markdown_fencing(match.group(1))
    return None
