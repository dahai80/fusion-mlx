# SPDX-License-Identifier: Apache-2.0
"""URL safety helpers — block SSRF and path traversal for image/video params."""

import ipaddress
import logging
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _tick_ssrf_reject(reason: str = "private_ip") -> None:
    # OP-2 (#0907 audit): surface SSRF denials as a Prometheus counter so an
    # operator can alert on a non-zero reject rate instead of grepping logs.
    try:
        from ..middleware.degradation_metrics import record_ssrf_rejection

        record_ssrf_rejection(reason)
    except Exception:
        logger.debug("record_ssrf_rejection('%s') failed", reason, exc_info=True)


# requests is an optional dependency for the safe-fetch helpers; imported
# lazily inside the functions so this module stays importable without it.
try:
    import requests
except ImportError:  # pragma: no cover - requests is a core dep in practice
    requests = None  # type: ignore[assignment]

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


def _resolve_safe_ips_or_raise(url: str) -> list[str]:
    # S-1/S-2 (#0907 audit): SSRF DNS-rebinding TOCTOU. is_safe_url_with_dns
    # validates once and the real connect re-resolves; a public DNS answer
    # can flip to a private/loopback IP between the two. Resolve to concrete
    # IPs here, validate them, and have the caller pin the connection to one
    # of these IPs so the on-the-wire connect goes where we checked.
    ips = resolve_safe_ips(url)
    if not ips:
        _tick_ssrf_reject("private_ip")
        raise ValueError(
            f"URL targets a private/internal or unresolvable address: {url}"
        )
    return ips


def make_safe_session(url: str, timeout: int) -> "requests.Session":
    # S-1 + S-2 (#0907 audit): build a requests.Session that (a) pins the
    # connect to a pre-validated IP and (b) refuses to follow redirects
    # automatically — redirects are re-validated per hop by the caller via
    # _resolve_safe_ips_or_raise, so a 302 to http://169.254.169.254/ cannot
    # smuggle a private target past the guard.
    if requests is None:
        raise RuntimeError(
            "requests is required for safe outbound fetch but is not installed"
        )

    class _PinnedHTTPAdapter(requests.adapters.HTTPAdapter):
        # S-1 (#0907 audit): override DNS at connect time. Connect to a
        # pre-validated IP instead of re-resolving the hostname, closing the
        # DNS-rebinding TOCTOU (A record flips to 127.0.0.1 after our check).

        def __init__(
            self, *args, pinned_hosts: dict[str, list[str]] | None = None, **kwargs
        ):
            self._pinned_hosts = pinned_hosts or {}
            super().__init__(*args, **kwargs)

        def get_connection(self, url, proxies=None):
            pinned = self._pinned_hosts
            if pinned:
                parsed = urlparse(url)
                host = parsed.hostname or ""
                if host in pinned:
                    safe_ip = pinned[host][0]
                    scheme = parsed.scheme or "https"
                    port = parsed.port
                    netloc = safe_ip if port is None else f"{safe_ip}:{port}"
                    pinned_url = f"{scheme}://{netloc}{parsed.path}"
                    if parsed.query:
                        pinned_url += f"?{parsed.query}"
                    logger.debug(
                        "SSRF pin: connecting %s -> %s (pinned IP for host %s)",
                        url,
                        pinned_url,
                        host,
                    )
                    return super().get_connection(pinned_url, proxies)
            return super().get_connection(url, proxies)

    ips = _resolve_safe_ips_or_raise(url)
    parsed = urlparse(url)
    host = parsed.hostname or ""
    session = requests.Session()
    adapter = _PinnedHTTPAdapter(pinned_hosts={host: ips})
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.max_redirects = 0
    # Preserve the original virtual-host Host header despite the IP-literal
    # connect URL the pinned adapter builds.
    if host:
        if parsed.port:
            session.headers["Host"] = f"{host}:{parsed.port}"
        else:
            session.headers["Host"] = host
    session.headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    )
    logger.info(
        "SSRF safe session: host=%s pinned_ips=%s timeout=%d",
        host,
        ips,
        timeout,
    )
    return session


def safe_fetch(
    url: str,
    *,
    timeout: int = 30,
    max_size: int,
    stream: bool = True,
    max_hops: int = 5,
):
    # S-1/S-2 (#0907 audit): single safe entry point for outbound media fetch.
    # Re-validates the target on every redirect hop (S-2) and pins each connect
    # to a validated IP (S-1). Returns the final streaming requests.Response;
    # caller is responsible for reading + size-capping the body.
    if requests is None:
        raise RuntimeError(
            "requests is required for safe outbound fetch but is not installed"
        )

    current = url
    for hop in range(max_hops + 1):
        session = make_safe_session(current, timeout)
        try:
            response = session.get(current, timeout=timeout, stream=stream, verify=True)
        except requests.RequestException as e:
            logger.warning("SSRF safe_fetch: request failed for %s: %s", current, e)
            raise
        if response.is_redirect:
            location = response.headers.get("location", "")
            if not location:
                response.close()
                _tick_ssrf_reject("redirect_no_location")
                raise ValueError(f"redirect with no Location from {current}")
            # S-2: re-resolve + re-check the redirect target before following.
            logger.info(
                "SSRF safe_fetch: redirect hop %d %s -> %s (re-validating)",
                hop,
                current,
                location,
            )
            response.close()
            current = location
            # Re-validate will raise if the new target is unsafe.
            _resolve_safe_ips_or_raise(current)
            continue
        return response
    raise ValueError(f"too many redirects (>{max_hops}) for {url}")


async def safe_fetch_async(
    url: str,
    *,
    timeout: int = 30,
    max_hops: int = 5,
    max_size: int | None = None,
):
    # S-1/S-2 (#0907 audit): httpx counterpart of safe_fetch. Resolves+pins
    # each hop to a validated IP (S-1) and re-validates redirect targets
    # per hop (S-2). follow_redirects=False so no hop is followed blindly.
    import httpx

    current = url
    for hop in range(max_hops + 1):
        ips = _resolve_safe_ips_or_raise(current)
        parsed = urlparse(current)
        host = parsed.hostname or ""
        safe_ip = ips[0]
        port = parsed.port
        scheme = parsed.scheme or "https"
        netloc = safe_ip if port is None else f"{safe_ip}:{port}"
        connect_url = f"{scheme}://{netloc}{parsed.path}"
        if parsed.query:
            connect_url += f"?{parsed.query}"
        # Preserve the original virtual host: Host header for HTTP, SNI for
        # TLS. httpx AsyncHTTPTransport uses the connect_url's host for SNI,
        # so for https we must carry the original host via an extension /
        # ssl context server_hostname. Simplest robust: use the original
        # URL for SNI by pinning via a custom transport that swaps the
        # network address only. httpx lacks a clean per-call address pin,
        # so we fall back to Host-header pinning for http and, for https,
        # rely on the resolved IP being the same host we validated (the
        # re-resolution at each hop keeps the validated set fresh).
        headers = {"Host": f"{host}:{port}" if port else host}
        logger.info(
            "SSRF safe_fetch_async: host=%s pinned_ip=%s hop=%d",
            host,
            safe_ip,
            hop,
        )
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=timeout,
            headers=headers,
        ) as client:
            try:
                response = await client.get(connect_url)
            except httpx.RequestError as e:
                logger.warning(
                    "SSRF safe_fetch_async: request failed for %s: %s", current, e
                )
                raise
        if response.is_redirect:
            location = response.headers.get("location", "")
            if not location:
                raise ValueError(f"redirect with no Location from {current}")
            logger.info(
                "SSRF safe_fetch_async: redirect hop %d %s -> %s (re-validating)",
                hop,
                current,
                location,
            )
            current = location
            _resolve_safe_ips_or_raise(current)
            continue
        if max_size is not None:
            cl = response.headers.get("content-length")
            if cl and int(cl) > max_size:
                raise ValueError(f"resource at {url} exceeds max size {max_size} bytes")
        return response
    raise ValueError(f"too many redirects (>{max_hops}) for {url}")


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
