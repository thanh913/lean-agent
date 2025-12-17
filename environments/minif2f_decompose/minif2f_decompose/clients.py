"""API clients for LLM and Lean compilation services."""

from __future__ import annotations

import asyncio
import os
import ssl
from dataclasses import dataclass
from typing import Any

import httpx
import requests
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# HTTP Utilities
# ---------------------------------------------------------------------------


def _get_proxy_url() -> str | None:
    """Get proxy URL from environment variables."""
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        if proxy := os.getenv(var):
            return proxy
    return None


def _get_ssl_verify() -> bool | ssl.SSLContext:
    """Get SSL verification setting from environment."""
    ssl_verify = os.getenv("SSL_VERIFY", "true").lower()
    return ssl_verify not in ("false", "0", "no", "off")


def _get_proxy_dict() -> dict[str, str] | None:
    """Get proxy configuration for requests library."""
    if proxy_url := _get_proxy_url():
        return {"http": proxy_url, "https": proxy_url}
    return None


def create_httpx_client(**kwargs: Any) -> httpx.AsyncClient:
    """Create an httpx AsyncClient with proxy and SSL settings from environment."""
    client_kwargs: dict[str, Any] = {}
    if proxy_url := _get_proxy_url():
        client_kwargs["proxy"] = proxy_url
    if not _get_ssl_verify():
        client_kwargs["verify"] = False
    client_kwargs.update(kwargs)
    return httpx.AsyncClient(**client_kwargs)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Raised when LLM call fails after retries."""
    pass


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMClient:
    """Unified LLM API wrapper with concurrency control and retry."""

    # Class-level counter for debugging concurrent requests (prover only)
    _prover_active = 0
    _prover_max_seen = 0

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        max_parallel: int = 8,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        semaphore: asyncio.Semaphore | None = None,
        track_concurrency: bool = False,  # Only enable for prover
    ) -> None:
        self._client = client
        # Use provided semaphore or create a new one
        self._sem = semaphore if semaphore is not None else asyncio.Semaphore(max(1, max_parallel))
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._total_calls = 0
        self._failed_calls = 0
        self._track_concurrency = track_concurrency

    async def call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        request_id: str = "",
        started_event: asyncio.Event | None = None,
        **sampling: Any,
    ) -> LLMResponse:
        """Call the LLM with concurrency control.

        Args:
            model: Model identifier
            messages: Chat messages
            request_id: Optional request ID for logging
            started_event: Optional event to set when semaphore is acquired
                          (signals that execution has actually started)
            **sampling: Additional sampling parameters
        """
        async with self._sem:
            # Signal that we've acquired the semaphore (actually started)
            if started_event is not None:
                started_event.set()
            if self._track_concurrency:
                LLMClient._prover_active += 1
                if LLMClient._prover_active > LLMClient._prover_max_seen:
                    LLMClient._prover_max_seen = LLMClient._prover_active
                    print(f"[PROVER] New max concurrent: {LLMClient._prover_max_seen}")
            self._total_calls += 1
            try:
                return await self._call_with_retry(model, messages, sampling)
            finally:
                if self._track_concurrency:
                    LLMClient._prover_active -= 1

    async def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        sampling: dict[str, Any],
    ) -> LLMResponse:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._do_call(model, messages, sampling)
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay * (2 ** attempt))
        self._failed_calls += 1
        raise LLMError(f"LLM call failed after {self._max_retries + 1} attempt(s): {last_error}")

    async def _do_call(
        self,
        model: str,
        messages: list[dict[str, str]],
        sampling: dict[str, Any],
    ) -> LLMResponse:
        loop = asyncio.get_running_loop()
        start = loop.time()
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            **sampling,
        )
        latency_ms = (loop.time() - start) * 1000.0
        content = response.choices[0].message.content or ""

        # Token extraction (handles both dict and object response formats)
        usage = getattr(response, "usage", None)
        if usage is None:
            prompt_tokens, completion_tokens = 0, 0
        elif isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
        else:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        return LLMResponse(content, prompt_tokens, completion_tokens, latency_ms)

    @property
    def failure_rate(self) -> float:
        return self._failed_calls / self._total_calls if self._total_calls else 0.0

    @property
    def stats(self) -> dict[str, int]:
        return {"total_calls": self._total_calls, "failed_calls": self._failed_calls}


# ---------------------------------------------------------------------------
# Lean Client
# ---------------------------------------------------------------------------


class LeanError(Exception):
    """Raised when Lean service is unavailable after retries."""
    pass


@dataclass
class CompileResult:
    """Result from Lean compilation."""
    ok: bool
    messages: list[dict[str, Any]]
    error_log: str


class LeanClient:
    """Lean compilation service client with retry."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        max_retries: int = 2,
        client_timeout_buffer: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout_buffer = client_timeout_buffer
        self._total_calls = 0
        self._failed_calls = 0

    async def compile(
        self,
        *,
        code: str,
        timeout: int,
        allow_sorry: bool,
        snippet_id: str = "",
    ) -> CompileResult:
        """Compile Lean code with retry on transient failures."""
        self._total_calls += 1

        for attempt in range(self._max_retries + 1):
            result, should_retry = await self._try_compile(code, timeout, snippet_id)
            if result is not None:
                return self._check_result(result, allow_sorry)
            if not should_retry or attempt >= self._max_retries:
                break
            await asyncio.sleep(min(2 ** attempt, 10))

        self._failed_calls += 1
        return CompileResult(False, [], f"Lean service unavailable after {self._max_retries + 1} attempts")

    async def _try_compile(
        self,
        code: str,
        timeout: int,
        snippet_id: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        payload = {
            "snippets": [{"id": snippet_id, "code": code}],
            "timeout": timeout,
            "reuse": True,
            "debug": False,
        }
        http_timeout = timeout + self._timeout_buffer

        def _post() -> tuple[dict[str, Any] | None, bool]:
            try:
                headers: dict[str, str] = {"Content-Type": "application/json"}
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"
                response = requests.post(
                    f"{self._base_url}/check",
                    json=payload,
                    headers=headers,
                    timeout=http_timeout,
                    proxies=_get_proxy_dict(),
                    verify=_get_ssl_verify(),
                )
                response.raise_for_status()
                return response.json(), False
            except requests.exceptions.Timeout:
                return None, True
            except requests.exceptions.ConnectionError:
                return None, True
            except Exception as exc:
                return {"error": str(exc)}, False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _post)

    def _check_result(self, data: dict[str, Any], allow_sorry: bool) -> CompileResult:
        result = (data.get("results") or [{}])[0]
        resp_payload = result.get("response") or {}
        messages = resp_payload.get("messages") or []
        sorries = resp_payload.get("sorries") or []

        has_error = any(msg.get("severity") == "error" for msg in messages)
        has_sorry = bool(sorries) or any(
            msg.get("severity") == "warning" and "sorry" in (msg.get("data") or "").lower()
            for msg in messages
        )
        ok = not has_error and (allow_sorry or not has_sorry) and result.get("error") is None

        # Format error log
        log_lines: list[str] = []
        if result.get("error"):
            log_lines.append(str(result["error"]))
        for msg in messages:
            if str(msg.get("severity", "")).lower() == "error":
                pos = msg.get("pos") or {}
                log_lines.append(f"ERROR (line {pos.get('line', '?')}, col {pos.get('column', '?')}): {msg.get('data', '')}")

        return CompileResult(ok, messages, "\n".join(filter(None, log_lines)))

    @property
    def failure_rate(self) -> float:
        return self._failed_calls / self._total_calls if self._total_calls else 0.0

    @property
    def stats(self) -> dict[str, int]:
        return {"total_calls": self._total_calls, "failed_calls": self._failed_calls}
