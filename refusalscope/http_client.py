"""Small authenticated HTTP client built on Python's standard library."""

from __future__ import annotations

from dataclasses import dataclass
from email.message import Message
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import REQUEST_TIMEOUT_SECONDS


@dataclass(slots=True)
class HTTPResponse:
    status: int = 0
    body: bytes = b""
    headers: Message | None = None
    error: str | None = None

    def header(self, name: str, default: str = "") -> str:
        if self.headers is None:
            return default
        return self.headers.get(name, default)

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


def http_get(
    url: str,
    *,
    token: str = "",
    byte_range: str | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> HTTPResponse:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "ModelDebugger/0.4-python",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if byte_range:
        headers["Range"] = f"bytes={byte_range}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return HTTPResponse(
                status=response.status,
                body=response.read(),
                headers=response.headers,
            )
    except HTTPError as error:
        return HTTPResponse(
            status=error.code,
            body=error.read(),
            headers=error.headers,
        )
    except (URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        return HTTPResponse(error=str(reason))


def http_request(
    url: str,
    *,
    method: str = "GET",
    token: str = "",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    content_type: str = "application/json",
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> HTTPResponse:
    """Make a small authenticated request to a ModelDebugger execution worker."""
    request_headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": "ModelDebugger/0.4-python",
    }
    for name, value in (headers or {}).items():
        if "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            raise ValueError("HTTP header names and values cannot contain newlines")
        request_headers[name] = value
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        request_headers["Content-Type"] = content_type
        request_headers["Content-Length"] = str(len(body))
    request = Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return HTTPResponse(
                status=response.status,
                body=response.read(),
                headers=response.headers,
            )
    except HTTPError as error:
        return HTTPResponse(
            status=error.code,
            body=error.read(),
            headers=error.headers,
        )
    except (URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        return HTTPResponse(error=str(reason))
