import logging
import re

logger = logging.getLogger(__name__)

_ZEROCONF_AVAILABLE = True
try:
    from zeroconf import ServiceInfo, Zeroconf
except ImportError:
    _ZEROCONF_AVAILABLE = False

_SERVICE_TYPE = "_fusion-mlx._tcp.local."
_TXT_REFRESH_INTERVAL = 60


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
