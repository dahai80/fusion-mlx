# SPDX-License-Identifier: Apache-2.0
"""URL safety helpers — block SSRF by rejecting private/internal IPs."""

import ipaddress
import logging
import socket
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

_BLOCKED_HOSTNAMES = frozenset({
    "localhost",
    "metadata.google.internal",
    "metadata.google.internal.",
})


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


def is_safe_url_with_dns(url: str) -> bool:
    if not is_safe_url(url):
        return False
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return False
    try:
        addr = ipaddress.ip_address(hostname)
        return not _is_private_addr(addr)
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _type, _proto, _canon, sockaddr in infos:
            ip_str = sockaddr[0]
            try:
                addr = ipaddress.ip_address(ip_str)
                if _is_private_addr(addr):
                    logger.warning("is_safe_url_with_dns: resolved %s -> %s (private)", hostname, ip_str)
                    return False
            except ValueError:
                continue
    except Exception:
        logger.debug("is_safe_url_with_dns: DNS lookup failed for %s", hostname, exc_info=True)
        return False
    return True
