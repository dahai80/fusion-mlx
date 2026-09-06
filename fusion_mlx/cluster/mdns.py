import hashlib
import hmac
import logging
import os
import re

logger = logging.getLogger(__name__)

_ZEROCONF_AVAILABLE = True
try:
    from zeroconf import ServiceInfo, Zeroconf
except ImportError:
    _ZEROCONF_AVAILABLE = False

_SERVICE_TYPE = "_fusion-mlx._tcp.local."
_TXT_REFRESH_INTERVAL = 60
# E-23: explicit mDNS TTL. zeroconf's default is ~75s, so a SIGKILL/OOM that
# skips the goodbye (unregister) leaves a stale service entry on the network
# for over a minute — peers keep routing to a dead node. A short TTL makes
# stale entries age out faster. Kept comfortably above the TXT refresh
# interval so a live node's periodic refresh re-arms the TTL.
_SERVICE_TTL_SECONDS = 15
# CL-4 (#811 audit 0906): cluster-shared secret used to authenticate mDNS
# peer advertisements. mDNS is an unauthenticated broadcast — without a
# shared secret any host on the subnet can advertise a fake high-capacity
# node and have the gateway route prompts to it (prompt exfiltration +
# request hijack). The token is an HMAC-SHA256 of an explicit
# FUSION_CLUSTER_TOKEN (preferred) or the configured api_key, keyed by a
# fixed service salt. A consumer that ingests discovered nodes MUST verify
# verify_cluster_token() before registering the peer; a mismatched/absent
# token means the advertiser is not part of this cluster and must be
# rejected. When no secret material is configured the token is empty and
# verify_cluster_token() returns False — fail-closed: a cluster with no
# configured secret rejects ALL discovered peers rather than trusting
# unauthenticated advertisements.
_CLUSTER_TOKEN_SALT = b"fusion-mlx-cluster-mdns-v1"


def _cluster_secret_material() -> str | None:
    token = os.environ.get("FUSION_CLUSTER_TOKEN", "").strip()
    if token:
        return token
    try:
        from ..middleware.auth import _get_configured_api_key

        key = _get_configured_api_key()
        if key:
            return key
    except Exception:
        logger.debug("CL-4: cluster secret material lookup failed", exc_info=True)
    return None


def compute_cluster_token() -> str:
    material = _cluster_secret_material()
    if not material:
        return ""
    return hmac.new(_CLUSTER_TOKEN_SALT, material.encode(), hashlib.sha256).hexdigest()


def verify_cluster_token(provided: str | None) -> bool:
    expected = compute_cluster_token()
    if not expected:
        logger.warning(
            "CL-4: no cluster secret configured — rejecting discovered peer "
            "(set FUSION_CLUSTER_TOKEN or api_key to enable mDNS auth)"
        )
        return False
    if not provided:
        return False
    return hmac.compare_digest(provided, expected)


def _sanitize_name(node_id: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9-]", "-", node_id)
    name = re.sub(r"-+", "-", name).strip("-")
    return name or "fusion-mlx"


def build_txt_records(snapshot: dict) -> dict[str, str]:
    records: dict[str, str] = {}
    records["node_id"] = str(snapshot.get("node_id", ""))
    records["host"] = str(snapshot.get("host", ""))
    records["port"] = str(snapshot.get("port", ""))
    # Platform tag for fusion-gateway platform routing (#365).
    records["platform"] = str(snapshot.get("platform", "mac"))
    loaded = [m["id"] for m in snapshot.get("models", []) if m.get("loaded")]
    records["models_csv"] = ",".join(loaded)
    mem = snapshot.get("memory", {})
    records["available_percent"] = f"{mem.get('available_percent', 0.0):.1f}"
    # CL-4 (#811 audit 0906): cluster-shared-secret token so a consumer can
    # authenticate this advertisement (reject rogue-host fake nodes).
    records["cluster_token"] = compute_cluster_token()
    return records


class MdnsAdvertiser:
    def __init__(self, node_id: str, host: str, port: int, txt_records: dict[str, str]):
        self._node_id = node_id
        self._host = host
        self._port = port
        self._txt_records = txt_records
        self._zc: Zeroconf | None = None
        self._info: ServiceInfo | None = None
        self._refresh_task = None

    async def start(self, refresh_fn=None):
        if not _ZEROCONF_AVAILABLE:
            logger.warning("mDNS: zeroconf not installed, advertising disabled")
            return
        try:
            self._zc = Zeroconf()
            service_name = _sanitize_name(self._node_id)
            self._info = ServiceInfo(
                _SERVICE_TYPE,
                name=f"{service_name}.{_SERVICE_TYPE}",
                port=self._port,
                properties={
                    k: v.encode("utf-8") if isinstance(v, str) else v
                    for k, v in self._txt_records.items()
                },
                server=f"{service_name}.local.",
                # E-23: short TTL so stale entries (SIGKILL/OOM, no goodbye)
                # age out fast instead of lingering ~75s.
                host_ttl=_SERVICE_TTL_SECONDS,
                other_ttl=_SERVICE_TTL_SECONDS,
            )
            await self._zc.async_register_service(self._info)
            logger.info("mDNS: advertising %s on port %d", _SERVICE_TYPE, self._port)
            if refresh_fn is not None:
                self._refresh_task = _create_refresh_task(refresh_fn, self)
        except Exception:
            logger.warning("mDNS: failed to start advertising", exc_info=True)
            await self._safe_stop()

    async def stop(self):
        await self._safe_stop()
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None

    async def update_txt(self, records: dict[str, str]):
        if self._zc is None or self._info is None:
            return
        try:
            self._txt_records = records
            encoded = {
                k: v.encode("utf-8") if isinstance(v, str) else v
                for k, v in records.items()
            }
            self._info.properties = encoded
            await self._zc.async_update_service(self._info)
            logger.debug("mDNS: updated TXT records")
        except Exception:
            logger.debug("mDNS: TXT update failed", exc_info=True)

    async def _safe_stop(self):
        if self._zc is not None:
            try:
                if self._info is not None:
                    await self._zc.async_unregister_service(self._info)
                    logger.info("mDNS: unregistered service")
            except Exception:
                logger.debug("mDNS: unregister failed", exc_info=True)
            try:
                await self._zc.async_close()
            except Exception:
                logger.debug("mDNS: close failed", exc_info=True)
            self._zc = None
            self._info = None


def _create_refresh_task(refresh_fn, advertiser: MdnsAdvertiser):
    import asyncio

    async def _loop():
        while True:
            await asyncio.sleep(_TXT_REFRESH_INTERVAL)
            try:
                snapshot = refresh_fn()
                records = build_txt_records(snapshot)
                await advertiser.update_txt(records)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("mDNS: periodic refresh failed", exc_info=True)

    return asyncio.ensure_future(_loop())
