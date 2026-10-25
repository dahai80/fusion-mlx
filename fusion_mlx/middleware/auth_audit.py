# SPDX-License-Identifier: Apache-2.0
# #1381: auth coverage audit. The gateway-as-sole-entry deployment needs to
# know which routes a keyless caller can still reach. Walking the live FastAPI
# route table is the only check that cannot drift: every router that forgets a
# `Depends(verify_api_key...)` shows up here the moment it is mounted.

import logging

logger = logging.getLogger(__name__)

# Paths (and prefixes) intentionally reachable without a key. Everything else
# must carry an auth dependency or appear in `UNGUARDED_ALLOWLIST` with a
# written reason — the unit test fails on anything new.
EXEMPT_PATHS = {
    "/",
    "/health",
    "/health/ready",
    "/healthz",
    "/livez",
    "/readyz",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}

# Unguarded routes that are deliberate. Map path -> reason. Empty today: every
# non-exempt route carries a key check. Add an entry only with the reason the
# route must stay open, so the test failure below is a decision, not a shrug.
UNGUARDED_ALLOWLIST: dict[str, str] = {}

_AUTH_CALLABLE_NAMES = frozenset(
    {
        "verify_api_key",
        "verify_api_key_or_x_api_key",
        "verify_scoped_api_key",
        "verify_management_access",
        "require_model_hub_source",
        "require_admin",
        "require_ready",
        "check_rate_limit",
        "check_rate_limit_or_x_api_key",
    }
)


def _guarded(route) -> bool:
    dependant = getattr(route, "dependant", None)
    seen = set()
    stack = [dependant]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        call = getattr(node, "call", None)
        if call is not None:
            name = getattr(call, "__name__", "")
            if name in _AUTH_CALLABLE_NAMES:
                return True
        stack.extend(getattr(node, "dependencies", []) or [])
    return False


def audit_route_auth(app) -> list[tuple[str, str]]:
    # Returns [(path, "METHOD,METHOD"), ...] for every mounted route that no
    # auth dependency guards and that is not exempt/allowlisted.
    findings: list[tuple[str, str]] = []
    for route in getattr(app, "routes", []):
        path = getattr(route, "path", None)
        if not path or path in EXEMPT_PATHS or path in UNGUARDED_ALLOWLIST:
            continue
        methods = getattr(route, "methods", None)
        if not methods:
            continue
        if _guarded(route):
            continue
        findings.append((path, ",".join(sorted(methods))))
    return findings


def log_auth_audit(app) -> list[tuple[str, str]]:
    findings = audit_route_auth(app)
    if findings:
        logger.warning(
            "auth audit: %d mounted route(s) accept requests without an API " "key: %s",
            len(findings),
            ", ".join(f"{m} {p}" for p, m in findings[:20]),
        )
    else:
        logger.info(
            "auth audit: every non-exempt route enforces an API key " "(exempt: %s)",
            ", ".join(sorted(EXEMPT_PATHS)),
        )
    return findings
