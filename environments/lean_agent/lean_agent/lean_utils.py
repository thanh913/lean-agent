"""Lean verification utilities."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class LeanCheckResult:
    """Result from Lean verification service."""

    ok: bool
    has_sorry: bool
    log: str
    messages: list[dict] = field(default_factory=list)


def create_httpx_client(timeout: float = 180.0) -> httpx.AsyncClient:
    """Create an async HTTP client with appropriate settings."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )


async def lean_check(
    code: str,
    verification_url: str,
    verification_key: str,
    timeout: int = 60,
    snippet_id: str = "check",
    max_retries: int = 2,
    client_timeout_buffer: int = 120,
) -> LeanCheckResult:
    """Submit code to the Lean verification service.

    Args:
        code: Lean code to verify.
        verification_url: URL of the verification service.
        verification_key: API key for the service.
        timeout: Server-side timeout in seconds.
        snippet_id: Identifier for this snippet.
        max_retries: Number of retries on timeout/connection errors.
        client_timeout_buffer: Extra seconds for client timeout.

    Returns:
        LeanCheckResult with verification status.
    """
    payload = {
        "snippets": [{"id": snippet_id, "code": code}],
        "timeout": timeout,
        "reuse": True,
        "debug": False,
    }

    http_timeout = timeout + client_timeout_buffer
    headers = {"Content-Type": "application/json"}
    if verification_key:
        headers["Authorization"] = f"Bearer {verification_key}"

    url = verification_url.rstrip("/") + "/check"
    last_error = "Unknown error"

    for attempt in range(max_retries + 1):
        try:
            async with create_httpx_client(timeout=http_timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            last_error = "Timeout"
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 10))
                continue
            return LeanCheckResult(
                ok=False,
                has_sorry=False,
                log=f"Lean service timeout after {max_retries + 1} attempts",
                messages=[],
            )
        except httpx.ConnectError:
            last_error = "Connection error"
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 10))
                continue
            return LeanCheckResult(
                ok=False,
                has_sorry=False,
                log=f"Lean service connection error after {max_retries + 1} attempts",
                messages=[],
            )
        except Exception as exc:
            return LeanCheckResult(
                ok=False,
                has_sorry=False,
                log=f"Lean service error: {exc}",
                messages=[],
            )

        # Parse response
        result = (data.get("results") or [{}])[0]
        resp_payload = result.get("response") or {}
        messages = resp_payload.get("messages") or []
        sorries = resp_payload.get("sorries") or []

        has_error = any(msg.get("severity") == "error" for msg in messages)
        has_sorry = bool(sorries) or any(
            msg.get("severity") == "warning" and "sorry" in (msg.get("data") or "").lower()
            for msg in messages
        )
        ok = (not has_error) and result.get("error") is None

        # Build log from error messages
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
            log_lines.append(f"ERROR (line {line}, col {col}): {msg.get('data', '')}")

        log = "\n".join(filter(None, log_lines))

        return LeanCheckResult(ok=ok, has_sorry=has_sorry, log=log, messages=messages)

    # Should not reach here
    return LeanCheckResult(
        ok=False,
        has_sorry=False,
        log=f"Lean service error after {max_retries + 1} attempts: {last_error}",
        messages=[],
    )


def format_check_result(result: LeanCheckResult, header_info: str = "") -> str:
    """Format a LeanCheckResult for display to the model."""
    if result.ok and not result.has_sorry:
        return "Verified (no sorries). Proof complete!"
    elif result.ok and result.has_sorry:
        return "Compiled with sorry. Fill in the remaining proofs."
    else:
        error_msg = result.log or "Unknown compilation error"
        return f"Compilation failed:\n```\n{error_msg}\n```"


