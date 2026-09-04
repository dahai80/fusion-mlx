# SPDX-License-Identifier: Apache-2.0
"""URL safety helpers — block SSRF and path traversal for image/video params."""

import ipaddress
import logging
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.google.internal.",
    }
)


def _is_private_addr(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    for net in _PRIVATE_NETWORKS:
        if addr in net:
            return True
    if addr.is_loopback or addr.is_link_local or addr.is_reserved:
        return True
    if isinstance(addr, ipaddress.IPv6Address):
        mapped = addr.ipv4_mapped
        if mapped is not None:
            return _is_private_addr(mapped)
    return False


def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname.lower() in _BLOCKED_HOSTNAMES:
            return False
        try:
            addr = ipaddress.ip_address(hostname)
            if _is_private_addr(addr):
                return False
        except ValueError:
            pass
        return True
    except Exception:
        logger.debug("is_safe_url: failed to parse %r", url, exc_info=True)
        return False


def resolve_safe_ips(url: str) -> list[str] | None:
    """Resolve ``url``'s host to a list of safe public IP strings.

    Returns None if the URL is malformed or resolves to a private/internal
    address. The returned IPs are the ones validated NOW; callers that make
    an outbound fetch SHOULD pin the connection to one of these IPs (e.g.
    via a custom HTTP adapter / Host header) to close the DNS-rebinding
    TOCTOU window between this check and the actual connect.

    Single-hostname A records can still rotate under us; the robust fix is
    to re-resolve and re-check at connect time. See ``is_safe_url_with_dns``
    for the boolean convenience wrapper.
    """
    if not is_safe_url(url):
        return None
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return None
    try:
        addr = ipaddress.ip_address(hostname)
        if _is_private_addr(addr):
            return None
        return [str(addr)]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except Exception:
        logger.debug(
            "resolve_safe_ips: DNS lookup failed for %s", hostname, exc_info=True
        )
        return None
    safe: list[str] = []
    for _family, _type, _proto, _canon, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _is_private_addr(addr):
            logger.warning(
                "resolve_safe_ips: resolved %s -> %s (private), blocking",
                hostname,
                ip_str,
            )
            return None
        safe.append(ip_str)
    if not safe:
        logger.warning("resolve_safe_ips: %s resolved to no usable IPs", hostname)
        return None
    return safe


def is_safe_url_with_dns(url: str) -> bool:
    return resolve_safe_ips(url) is not None


_ALLOWED_READ_DIRS: list[str] = [
    os.path.expanduser("~/.fusion-mlx/models"),
    os.path.expanduser("~/.fusion-mlx/cache"),
    "/tmp",
    "/var/tmp",
]

# Issue #633: operator-extensible read dirs. FUSION_MLX_ALLOWED_READ_DIRS is a
# colon-separated list (like PATH) of extra directories appended to the base
# allow-list, so scene-continuity condition images from custom output dirs
# (e.g. fusion-comfyui) are accepted without writing to /tmp.
_EXTRA_READ_DIRS_ENV = "FUSION_MLX_ALLOWED_READ_DIRS"


def get_allowed_read_dirs() -> list[str]:
    base = list(_ALLOWED_READ_DIRS)
    extra_raw = os.environ.get(_EXTRA_READ_DIRS_ENV, "")
    if extra_raw:
        for part in extra_raw.split(":"):
            part = part.strip()
            if part and part not in base:
                base.append(part)
    return base


def _resolve_and_check(path_str: str) -> Path:
    resolved = Path(path_str).resolve()
    for allowed in get_allowed_read_dirs():
        allowed_resolved = Path(allowed).resolve()
        try:
            resolved.relative_to(allowed_resolved)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"Path traversal blocked: {path_str} is outside allowed directories"
    )


def is_safe_local_path(path_str: str) -> bool:
    if not path_str or not isinstance(path_str, str):
        return False
    if path_str.startswith("file://"):
        path_str = path_str[7:]
    if "\0" in path_str:
        logger.warning("is_safe_local_path: null byte in path %r", path_str[:100])
        return False
    try:
        _resolve_and_check(path_str)
        return True
    except ValueError as e:
        logger.warning("is_safe_local_path: %s", e)
        return False
