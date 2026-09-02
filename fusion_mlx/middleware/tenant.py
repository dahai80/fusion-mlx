# SPDX-License-Identifier: Apache-2.0
"""Multi-tenant isolation helpers (issue #756).

Gateway (fusion-gateway #152) derives an authoritative tenant from the
api_key -> team binding and stamps it as ``X-Fusion-Tenant`` on every
upstream request, alongside ``X-Fusion-Route: gateway-decision`` as the
origin signal.

This module gives fusion-mlx the matching backend half:

- ``tenant_from_request``: read the authoritative ``X-Fusion-Tenant``
  header. ``X-Space-Id`` is a non-authoritative passthrough and is
  deliberately ignored for tenant derivation (a spoofed X-Space-Id must
  not cross tenant boundaries).
- ``tenant_isolation_enabled``: env ``FUSION_TENANT_ISOLATION=true`` gate
  (default OFF — single-tenant dev deployments are unaffected).
- ``GATEWAY_DECISION_ROUTE``: the contract value the gateway stamps on
  every outbound request. When isolation is on and no
  ``FUSION_ROUTE_TOKEN`` shared secret is configured, the route_guard
  middleware additionally requires the header VALUE to equal this
  constant (a bare presence check is not enough — a direct-port caller
  could inject any value).
- ``scoped_principal``: compose the per-tenant session principal so
  sessions, stats, and context caps stay isolated per tenant even when
  two tenants share the same bearer api_key shape (rate-limit bucket).

The ASGI enforcement lives in route_guard.py (RouteGuardMiddleware),
which already rejects missing X-Fusion-Route by default (#349). The
tenant value check is layered on top there; the helpers here are the
single source of truth for the header names + semantics so route and
session code do not re-derive them.
"""

from __future__ import annotations

import logging
import os

from fastapi import Request

logger = logging.getLogger(__name__)

# Contract headers (gateway -> backend). Keep names stable; the gateway
# stamps exactly these.
TENANT_HEADER = "x-fusion-tenant"
SPACE_ID_HEADER = "x-space-id"  # non-authoritative passthrough, ignored
GATEWAY_DECISION_ROUTE = "gateway-decision"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def tenant_isolation_enabled() -> bool:
    # Default OFF: single-tenant dev + standalone deployments keep the
    # existing behavior. Multi-tenant deployments behind a gateway opt in.
    return _env_truthy("FUSION_TENANT_ISOLATION")


def tenant_from_request(request: Request) -> str | None:
    # Authoritative tenant = gateway-stamped X-Fusion-Tenant only.
    # X-Space-Id is deliberately not read here — it is a client-supplied
    # passthrough and must not influence tenant scoping (#756 Gap1c).
    raw = request.headers.get(TENANT_HEADER)
    if raw is None:
        return None
    tenant = raw.strip()
    if not tenant:
        return None
    return tenant


def scoped_principal(request: Request, base_principal: str) -> str:
    # Compose the per-tenant session principal. When isolation is on and
    # a tenant is present, namespace the base principal (HMAC rate-limit
    # bucket) with the tenant so two tenants cannot reach each other's
    # sessions / stats / context caps via a shared principal shape.
    # ``base_principal`` is the existing request_principal() value
    # (HMAC of bearer key, else client subnet).
    if not tenant_isolation_enabled():
        return base_principal
    tenant = tenant_from_request(request)
    if not tenant:
        return base_principal
    return f"t:{tenant}:{base_principal}"
