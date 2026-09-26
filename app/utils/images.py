"""Conditioning-image loading with SSRF and size guards."""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


class ImageFetchError(ValueError):
    """Raised when a conditioning image cannot be loaded."""


def _assert_public_host(url: str) -> None:
    """Reject loopback/private/link-local targets to limit SSRF exposure."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ImageFetchError(f"unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ImageFetchError("URL has no host")

    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ImageFetchError(f"cannot resolve host: {parsed.hostname}") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            raise ImageFetchError(
                f"refusing to fetch image from non-public address {address}"
            )


def load_image_from_url(url: str, *, max_bytes: int, timeout: float = 30.0) -> Any:
    """Download and decode an image, enforcing scheme, host and size limits."""
    _assert_public_host(url)

    from PIL import Image  # noqa: PLC0415 - heavy optional dependency

    buffer = bytearray()
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
                raise ImageFetchError(f"unsupported image content-type: {content_type!r}")
            for chunk in response.iter_bytes():
                buffer.extend(chunk)
                if len(buffer) > max_bytes:
                    raise ImageFetchError(
                        f"image exceeds the configured limit of {max_bytes} bytes"
                    )

    import io  # noqa: PLC0415

    try:
        image = Image.open(io.BytesIO(bytes(buffer)))
        image.load()
    except Exception as exc:
        raise ImageFetchError("downloaded file is not a valid image") from exc
    return image.convert("RGB")


def load_image_from_path(path: Path, *, max_bytes: int) -> Any:
    from PIL import Image  # noqa: PLC0415

    size = path.stat().st_size
    if size > max_bytes:
        raise ImageFetchError(f"image exceeds the configured limit of {max_bytes} bytes")
    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        raise ImageFetchError(f"not a valid image file: {path.name}") from exc
    return image.convert("RGB")
