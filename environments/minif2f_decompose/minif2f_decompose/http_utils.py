"""HTTP utilities for proxy and SSL configuration."""

from __future__ import annotations

import os
import ssl
from typing import Any

import httpx


def get_proxy_url() -> str | None:
    """Get proxy URL from environment variables.

    Checks HTTPS_PROXY, HTTP_PROXY (case-insensitive) in order.
    Returns None if no proxy is configured.
    """
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        proxy = os.getenv(var)
        if proxy:
            return proxy
    return None


def get_ssl_verify() -> bool | ssl.SSLContext:
    """Get SSL verification setting from environment.

    If SSL_VERIFY is set to 'false', '0', or 'no', returns False to disable verification.
    This is similar to curl's --ssl-revoke-best-effort for corporate proxies.

    Returns True (verify) by default.
    """
    ssl_verify = os.getenv("SSL_VERIFY", "true").lower()
    if ssl_verify in ("false", "0", "no", "off"):
        return False
    return True


def get_proxy_dict() -> dict[str, str] | None:
    """Get proxy configuration as a dict for the requests library.

    Returns a dict like {"http": "...", "https": "..."} or None if no proxy.
    """
    proxy_url = get_proxy_url()
    if not proxy_url:
        return None
    return {
        "http": proxy_url,
        "https": proxy_url,
    }


def create_httpx_client(**kwargs: Any) -> httpx.AsyncClient:
    """Create an httpx AsyncClient with proxy and SSL settings from environment.

    Additional kwargs are passed to the AsyncClient constructor.
    """
    proxy_url = get_proxy_url()
    ssl_verify = get_ssl_verify()

    client_kwargs: dict[str, Any] = {}

    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    if not ssl_verify:
        client_kwargs["verify"] = False

    client_kwargs.update(kwargs)
    return httpx.AsyncClient(**client_kwargs)
