"""Lightweight Lean helpers: fenced-block parsing, proof-body surgery, and remote compile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import asyncio
import json
import re
import requests

from .http_utils import get_proxy_dict, get_ssl_verify


LEAN_FENCE = re.compile(r"```\s*lean4?\s*", re.IGNORECASE)
GENERIC_FENCE = re.compile(r"```\s*[\w-]*\s*", re.IGNORECASE)


@dataclass
class LeanCompileResult:
    ok: bool
    log: str
    messages: list[dict]


def extract_last_lean_block(text: str | None) -> Optional[str]:
    """Return the content of the final ```lean/lean4 fenced block."""
    if not text:
        return None
    lines = text.splitlines()
    fence_indices = [idx for idx, line in enumerate(lines) if GENERIC_FENCE.match(line.strip())]
    if len(fence_indices) < 2:
        return None
    for idx in range(len(fence_indices) - 2, -1, -1):
        start = fence_indices[idx]
        end = fence_indices[idx + 1]
        if start < end and LEAN_FENCE.match(lines[start].strip()):
            payload = "\n".join(lines[start + 1 : end]).strip()
            if payload:
                return payload
    start, end = fence_indices[-2], fence_indices[-1]
    payload = "\n".join(lines[start + 1 : end]).strip()
    return payload or None


def extract_proof_body(snippet: str | None) -> str:
    """Strip a Lean snippet down to the body following ':= by'."""
    if not snippet:
        return ""
    lines = snippet.splitlines()
    saw_header = False
    body: list[str] = []
    for line in lines:
        if not saw_header:
            if ":= by" in line:
                _, after = line.split(":= by", 1)
                saw_header = True
                tail = after.strip()
                if tail:
                    body.append(tail)
            continue
        body.append(line)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return ""
    indent = None
    for line in body:
        stripped = line.lstrip()
        if not stripped:
            continue
        delta = len(line) - len(stripped)
        indent = delta if indent is None else min(indent, delta)
    if indent:
        body = [ln[indent:] if len(ln) >= indent else ln for ln in body]
    return "\n".join(body)


def inject_proof_body(snippet: str, body: str) -> str:
    """Replace the proof in `snippet` with an indented `body`."""
    if ":= by" not in snippet:
        raise ValueError("snippet missing ':= by'")
    prefix, _ = snippet.split(":= by", 1)
    header = f"{prefix}:= by"
    payload = body.splitlines() or ["sorry"]
    indented = [f"  {line}" if line else "" for line in payload]
    return "\n".join([header, *indented]) + "\n"


async def lean_compile(
    *,
    code: str,
    verification_url: str,
    timeout: int,
    allow_sorry: bool,
    snippet_id: str,
    verification_key: str = "",
    client_timeout_buffer: int = 120,
    max_retries: int = 2,
) -> LeanCompileResult:
    """Submit `code` to the Lean service and return its response.

    Args:
        client_timeout_buffer: Extra seconds for client timeout to account for queue wait.
        max_retries: Number of retries on timeout/connection errors.
    """

    payload = {
        "snippets": [{"id": snippet_id, "code": code}],
        "timeout": timeout,
        "reuse": True,
        "debug": False,
    }

    # Client timeout = server timeout + buffer for queue wait
    http_timeout = timeout + client_timeout_buffer

    def _post() -> tuple[LeanCompileResult | None, bool]:
        """Returns (result, should_retry). Result is None if should retry."""
        try:
            headers = {"Content-Type": "application/json"}
            if verification_key:
                headers["Authorization"] = f"Bearer {verification_key}"
            response = requests.post(
                verification_url.rstrip("/") + "/check",
                json=payload,
                headers=headers,
                timeout=http_timeout,
                proxies=get_proxy_dict(),
                verify=get_ssl_verify(),
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            # Timeout is retryable (might be queue congestion)
            return None, True
        except requests.exceptions.ConnectionError:
            # Connection error is retryable
            return None, True
        except Exception as exc:  # pragma: no cover - network path
            return LeanCompileResult(ok=False, log=f"Lean service error: {exc}", messages=[]), False

        result = (data.get("results") or [{}])[0]
        resp_payload = result.get("response") or {}
        messages = resp_payload.get("messages") or []
        sorries = resp_payload.get("sorries") or []

        has_error = any(msg.get("severity") == "error" for msg in messages)
        has_sorry = bool(sorries) or any(
            msg.get("severity") == "warning" and "sorry" in (msg.get("data") or "").lower()
            for msg in messages
        )
        ok = (not has_error) and (allow_sorry or not has_sorry) and result.get("error") is None

        log_lines: list[str] = []
        if result.get("error"):
            log_lines.append(str(result["error"]))
        for msg in messages:
            sev = str(msg.get("severity", "")).lower()
            if sev != "error":
                continue
            pos = msg.get("pos") or {}
            line = pos.get("line", "?")
            col = pos.get("column", "?")
            log_lines.append(
                f"ERROR (line {line}, col {col}): {msg.get('data', '')}"
            )
        log = "\n".join(filter(None, log_lines))
        return LeanCompileResult(ok=ok, log=log, messages=messages), False

    loop = asyncio.get_running_loop()

    # Retry loop for transient failures (timeout, connection errors)
    last_error = "Unknown error"
    for attempt in range(max_retries + 1):
        result, should_retry = await loop.run_in_executor(None, _post)
        if result is not None:
            return result
        if not should_retry or attempt >= max_retries:
            break
        last_error = "Timeout or connection error"
        # Brief backoff before retry
        await asyncio.sleep(min(2 ** attempt, 10))

    return LeanCompileResult(ok=False, log=f"Lean service error after {max_retries + 1} attempts: {last_error}", messages=[])
