"""
fusion_gui server — compatibility layer integrating mlx-gui endpoints with fusion-mlx.

Exports:
- create_gui_router()    : full FastAPI app (standalone, with lifespan)
- get_gui_compat_router() : APIRouter with non-conflicting routes (for fusion_mlx integration)
"""

import asyncio
import base64
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union, Any
from urllib.parse import urlparse

import httpx
import tempfile
from fastapi import Depends, FastAPI, HTTPException, Header, status, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from fusion_mlx.gui_compat import __version__
from fusion_mlx.gui_compat.database import get_db_session, get_database_manager
from fusion_mlx.gui_compat.models import Model, AppSettings
from fusion_mlx.gui_compat.system_monitor import get_system_monitor
from fusion_mlx.gui_compat.huggingface_integration import get_huggingface_client
from fusion_mlx.gui_compat.model_manager import get_model_manager, shutdown_model_manager
from fusion_mlx.gui_compat.mlx_integration import GenerationConfig, get_inference_engine
from fusion_mlx.gui_compat.inference_queue_manager import get_inference_manager, shutdown_inference_manager, QueuedRequest
from fusion_mlx.gui_compat.queued_inference import (
    queued_generate_text,
    queued_generate_text_stream,
    queued_transcribe_audio,
    queued_generate_speech,
    queued_generate_embeddings,
    queued_generate_vision,
)

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def validate_api_key(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
) -> Optional[str]:
    if authorization:
        return authorization.credentials
    if x_api_key:
        return x_api_key
    return None


# ── Pydantic models ──

class ChatMessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Union[Dict[str, str], str]] = None


class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[Union[str, List[ChatMessageContent]]] = None
    image: Optional[str] = None
    images: Optional[List[str]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 8192
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0
    seed: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInstallRequest(BaseModel):
    model_id: str
    name: Optional[str] = None


class AudioSpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# ── Helpers ──

def _parse_harmony_response(text: str) -> str:
    if "<|channel|>" not in text:
        return text
    m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    parts = [mt.group(1).strip() for mt in re.finditer(r"<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|<\|channel\|>|$)", text, re.DOTALL)]
    if parts:
        return " ".join(parts)
    cleaned = " ".join(re.sub(r"<\|[^|]+\|>", " ", text).split())
    return cleaned or ""


def _format_chat_prompt(messages: List[ChatMessage]) -> tuple:
    chat_msgs, all_imgs = [], []
    for msg in messages:
        content, role = msg.content, msg.role
        if isinstance(content, list):
            txt, imgs = [], []
            for p in content:
                if p.type == "text" and p.text:
                    txt.append(p.text)
                elif p.type == "image_url" and p.image_url:
                    u = ""
                    if isinstance(p.image_url, dict):
                        u = p.image_url.get("url", "") or p.image_url.get("image", "") or p.image_url.get("data", "")
                    elif isinstance(p.image_url, str):
                        u = p.image_url
                    if u:
                        imgs.append(u)
                        all_imgs.append(u)
            chat_msgs.append({"role": role, "content": " ".join(txt)})
        elif isinstance(content, str):
            chat_msgs.append({"role": role, "content": content})
        if hasattr(msg, "image") and msg.image:
            all_imgs.append(msg.image)
        if hasattr(msg, "images") and msg.images:
            all_imgs.extend(msg.images)
    return chat_msgs, all_imgs


async def _apply_chat_template(tokenizer, messages: List[Dict], model_name: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logger.warning(f"Chat template failed for {model_name}: {e}")
    parts = []
    gemma = "gemma" in model_name.lower()
    phi = "phi" in model_name.lower()
    for m in messages:
        r, c = m["role"], m["content"]
        if gemma:
            if r == "system": parts.append(f"<bos><start_of_turn>system\n{c}<end_of_turn>")
            elif r == "user": parts.append(f"<start_of_turn>user\n{c}<end_of_turn>")
            elif r == "assistant": parts.append(f"<start_of_turn>model\n{c}<end_of_turn>")
        elif phi:
            parts.append(f"{'System' if r=='system' else 'User' if r=='user' else 'Assistant'}: {c}")
        else:
            parts.append(f"<{r}\n{c}")
    if gemma: parts.append("<start_of_turn>model\n")
    elif phi: parts.append("Assistant:")
    else: parts.append("<|assistant")
    return "\n".join(parts)


async def _process_image_urls(urls: List[str]) -> List[str]:
    result = []
    for u in urls:
        try:
            if u.startswith("data:image/"):
                if "," not in u: continue
                hdr, data = u.split(",", 1)
                ext = ".png" if "png" in hdr else ".gif" if "gif" in hdr else ".webp" if "webp" in hdr else ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
                    t.write(base64.b64decode(data))
                    result.append(t.name)
            elif u.startswith(("http://", "https://")):
                async with httpx.AsyncClient() as c:
                    r = await c.get(u); r.raise_for_status()
                    ct = r.headers.get("content-type", "")
                    ext = ".png" if "png" in ct else ".gif" if "gif" in ct else ".webp" if "webp" in ct else ".jpg"
                    if not any(x in ct for x in ["png", "gif", "webp", "jpeg"]):
                        pe = os.path.splitext(urlparse(u).path)[1].lower()
                        if pe in [".jpg", ".jpeg", ".png", ".gif", ".webp"]: ext = pe
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
                        t.write(r.content); result.append(t.name)
            elif re.match(r"^[A-Za-z0-9+/]*={0,2}$", u) and len(u) > 100:
                bd = base64.b64decode(u)
                ext = ".jpg" if bd.startswith(b"\xFF\xD8\xFF") else ".png" if bd.startswith(b"\x89PNG") else ".gif" if bd.startswith(b"GIF") else ".webp" if bd.startswith(b"RIFF") else ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
                    t.write(bd); result.append(t.name)
        except Exception as e:
            logger.error(f"Image URL process failed {u[:100]}: {e}")
    return result


def _model_dict(m):
    return {"id": m.id, "name": m.name, "author": m.author, "downloads": m.downloads,
           "likes": m.likes, "model_type": m.model_type, "size_gb": m.size_gb,
           "estimated_memory_gb": m.estimated_memory_gb, "mlx_compatible": m.mlx_compatible,
           "has_mlx_version": m.has_mlx_version, "mlx_repo_id": m.mlx_repo_id,
           "tags": m.tags, "description": m.description, "updated_at": m.updated_at}


def _popular_model_dict(m):
    return {"id": m.id, "name": m.name, "author": m.author, "downloads": m.downloads,
           "likes": m.likes, "model_type": m.model_type, "size_gb": m.size_gb,
           "estimated_memory_gb": m.estimated_memory_gb, "mlx_compatible": m.mlx_compatible,
           "description": m.description}


# ── Compat APIRouter ──

def get_gui_compat_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/manager/models")
    async def list_models_int(db: Session = Depends(get_db_session)):
        return {"models": [{"id": m.id, "name": m.name, "type": m.model_type, "status": m.status,
                 "memory_required_gb": m.memory_required_gb, "use_count": m.use_count,
                 "last_used_at": m.last_used_at.isoformat() if m.last_used_at else None,
                 "created_at": m.created_at.isoformat() if m.created_at else None,
                 "huggingface_id": m.huggingface_id,
                 "author": m.huggingface_id.split("/")[0] if m.huggingface_id and "/" in m.huggingface_id else "unknown"}
               for m in db.query(Model).all()]}

    @router.get("/v1/models/{model_name}")
    async def get_model(model_name: str, db: Session = Depends(get_db_session)):
        m = db.query(Model).filter(Model.name == model_name).first()
        if not m: raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        return {"id": m.id, "name": m.name, "path": m.path, "version": m.version,
               "type": m.model_type, "status": m.status, "memory_required_gb": m.memory_required_gb,
               "use_count": m.use_count,
               "last_used_at": m.last_used_at.isoformat() if m.last_used_at else None,
               "created_at": m.created_at.isoformat() if m.created_at else None,
               "updated_at": m.updated_at.isoformat() if m.updated_at else None,
               "error_message": m.error_message, "metadata": m.get_metadata()}

    @router.post("/v1/models/{model_name}/load")
    async def load_model(model_name: str, priority: int = 0, db: Session = Depends(get_db_session)):
        mr = db.query(Model).filter(Model.name == model_name).first()
        if not mr: raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        try:
            mm = get_model_manager()
            if model_name in mm._loaded_models:
                return {"message": f"Model '{model_name}' already loaded", "status": "loaded"}
            can, msg = get_system_monitor().check_model_compatibility(mr.memory_required_gb)
            if not can and "MLX requires Apple Silicon" in msg:
                raise HTTPException(status_code=400, detail=msg)
            ok = await mm.load_model_async(model_name=model_name, model_path=mr.path, priority=priority)
            resp = {"message": f"Model '{model_name}' loaded", "status": "loaded"}
            if "warning" in msg.lower(): resp["memory_warning"] = msg
            if not ok: raise HTTPException(status_code=500, detail=f"Failed to load '{model_name}'")
            return resp
        except HTTPException: raise
        except Exception as e:
            logger.error(f"Load error {model_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/v1/models/{model_name}/unload")
    async def unload_model(model_name: str, db: Session = Depends(get_db_session)):
        mr = db.query(Model).filter(Model.name == model_name).first()
        if not mr: raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        try:
            if get_model_manager().unload_model(model_name):
                return {"message": f"Model '{model_name}' unloaded", "status": "unloaded"}
            return {"message": f"Model '{model_name}' was not loaded", "status": "not_loaded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/v1/models/{model_name}")
    async def delete_model(model_name: str, remove_files: bool = True, db: Session = Depends(get_db_session)):
        mr = db.query(Model).filter(Model.name == model_name).first()
        if not mr: raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        try:
            get_model_manager().unload_model(model_name)
            db.delete(mr); db.commit()
            if remove_files and mr.path:
                try:
                    if ".cache" in mr.path and "models--" in mr.path:
                        cp = Path(mr.path)
                        if cp.exists():
                            for p in cp.parents:
                                if p.name.startswith("models--") and p.exists():
                                    shutil.rmtree(p); break
                    elif os.path.exists(mr.path):
                        shutil.rmtree(mr.path) if os.path.isdir(mr.path) else os.remove(mr.path)
                except Exception as fe: logger.warning(f"File remove failed: {fe}")
            return {"message": f"Model '{model_name}' deleted", "removed_files": remove_files, "status": "deleted"}
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/models/{model_name}/health")
    async def model_health(model_name: str, db: Session = Depends(get_db_session)):
        m = db.query(Model).filter(Model.name == model_name).first()
        if not m: raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        return {"model": model_name, "status": m.status, "healthy": m.status == "loaded",
               "last_used": m.last_used_at.isoformat() if m.last_used_at else None}

    @router.post("/v1/models/{model_name}/generate")
    async def gen_text(model_name: str, rd: dict, db: Session = Depends(get_db_session)):
        mr = db.query(Model).filter(Model.name == model_name).first()
        if not mr: raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        try:
            lm = get_model_manager().get_model_for_inference(model_name)
            if not lm: raise HTTPException(status_code=400, detail=f"Model '{model_name}' not loaded.")
            prompt = rd.get("prompt", "")
            if not prompt: raise HTTPException(status_code=400, detail="Prompt is required")
            cfg = GenerationConfig(max_tokens=rd.get("max_tokens", 100), temperature=rd.get("temperature", 0.0),
                                  top_p=rd.get("top_p", 1.0), top_k=rd.get("top_k", 0),
                                  repetition_penalty=rd.get("repetition_penalty", 1.0),
                                  repetition_context_size=rd.get("repetition_context_size", 20), seed=rd.get("seed"))
            r = await queued_generate_text(model_name, prompt, cfg)
            return {"model": model_name, "prompt": r.prompt, "text": _parse_harmony_response(r.text),
                   "usage": {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "total_tokens": r.total_tokens},
                   "timing": {"generation_time_seconds": r.generation_time_seconds, "tokens_per_second": r.tokens_per_second}}
        except HTTPException: raise
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/system/status")
    async def sys_status():
        ss = get_system_monitor().get_system_summary()
        return {"status": "running", "system": ss, "model_manager": get_model_manager().get_system_status(), "mlx_compatible": ss["mlx_compatible"]}

    @router.get("/v1/system/version")
    async def sys_version():
        from fusion_mlx.gui_compat import __author__, __description__
        return {"version": __version__, "author": __author__, "description": __description__, "name": "fusion_gui"}

    @router.post("/v1/system/shutdown")
    async def sys_shutdown():
        asyncio.get_event_loop().call_later(1.0, lambda: os._exit(0))
        return {"message": "Server shutting down"}

    @router.post("/v1/system/restart")
    async def sys_restart():
        asyncio.get_event_loop().call_later(1.0, lambda: (subprocess.Popen([sys.executable] + sys.argv), os._exit(0)))
        return {"message": "Server restarting"}

    @router.get("/v1/settings")
    async def get_settings(db: Session = Depends(get_db_session)):
        return {s.key: s.get_typed_value() for s in db.query(AppSettings).all()}

    @router.put("/v1/settings/{key}")
    async def put_setting(key: str, value: dict, db: Session = Depends(get_db_session)):
        s = db.query(AppSettings).filter(AppSettings.key == key).first()
        if not s: raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
        nv = value.get("value")
        if nv is None: raise HTTPException(status_code=400, detail="Value is required")
        s.set_typed_value(nv); db.commit()
        return {"key": key, "value": s.get_typed_value(), "updated": True}

    @router.get("/v1/discover/models")
    async def disc_models(query: str = "", limit: int = 20, sort: str = "downloads"):
        try:
            ms = get_huggingface_client().search_mlx_models(query=query, limit=limit, sort=sort)
            return {"models": [_model_dict(m) for m in ms], "total": len(ms)}
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/discover/popular")
    async def disc_popular(limit: int = 20):
        try:
            ms = get_huggingface_client().get_popular_mlx_models(limit=limit)
            return {"models": [_popular_model_dict(m) for m in ms], "total": len(ms)}
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/discover/trending")
    async def disc_trending(limit: int = 20):
        try:
            ms = get_huggingface_client().search_trending_mlx_models(limit=limit)
            return {"models": [_model_dict(m) for m in ms], "total": len(ms)}
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/discover/categories")
    async def disc_categories():
        try: return {"categories": get_huggingface_client().get_model_categories()}
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/discover/vision")
    async def disc_vision(query: str = "", limit: int = 10):
        try:
            ms = get_huggingface_client().search_vision_models(query=query, limit=limit)
            return {"models": [_model_dict(m) for m in ms], "total": len(ms)}
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/discover/stt")
    async def disc_stt(query: str = "", limit: int = 10):
        try:
            ms = get_huggingface_client().search_stt_models(query=query, limit=limit)
            return {"models": [_model_dict(m) for m in ms], "total": len(ms)}
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/discover/embeddings")
    async def disc_emb(query: str = "", limit: int = 20):
        try:
            ms = get_huggingface_client().search_embedding_models(query=query, limit=limit)
            return {"models": [_model_dict(m) for m in ms], "total": len(ms)}
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/discover/compatible")
    async def disc_compat(query: str = "", max_memory_gb: Optional[float] = None):
        try:
            if max_memory_gb is None: max_memory_gb = get_system_monitor().get_memory_info().total_gb * 0.8
            ms = get_huggingface_client().search_compatible_models(query, max_memory_gb)
            return {"models": [{**_model_dict(m), "memory_fit": f"{m.estimated_memory_gb:.1f}GB required, {max_memory_gb:.1f}GB available"} for m in ms],
                   "max_memory_gb": max_memory_gb, "total": len(ms)}
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/discover/models/{model_id:path}")
    async def disc_model_detail(model_id: str):
        try:
            m = get_huggingface_client().get_model_details(model_id)
            if not m: raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found on HuggingFace")
            can, msg = get_system_monitor().check_model_compatibility(m.estimated_memory_gb or 0)
            return {**_model_dict(m), "library_name": m.library_name, "pipeline_tag": m.pipeline_tag,
                   "system_compatible": can, "compatibility_message": msg}
        except HTTPException: raise
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.post("/v1/models/install")
    async def install_model(req: ModelInstallRequest, db: Session = Depends(get_db_session)):
        try:
            mi = get_huggingface_client().get_model_details(req.model_id)
            if not mi: raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found on HuggingFace")
            if not mi.mlx_compatible: raise HTTPException(status_code=400, detail=f"Model '{req.model_id}' not MLX compatible")
            em = mi.estimated_memory_gb or 4.0
            can, msg = get_system_monitor().check_model_compatibility(em)
            if not can and "MLX requires Apple Silicon" in msg: raise HTTPException(status_code=400, detail=msg)
            name = req.name or mi.name
            if db.query(Model).filter(Model.name == name).first():
                return {"message": f"Model '{name}' already installed", "status": "already_installed"}
            nm = Model(name=name, path=req.model_id, model_type=mi.model_type, huggingface_id=req.model_id,
                      memory_required_gb=int(em), status="unloaded")
            nm.set_metadata({"author": mi.author, "downloads": mi.downloads, "likes": mi.likes,
                           "tags": mi.tags, "size_gb": mi.size_gb, "estimated_memory_gb": mi.estimated_memory_gb})
            db.add(nm); db.commit()
            resp = {"message": f"Model '{name}' installed", "model_name": name, "status": "installed"}
            if "warning" in msg.lower(): resp["memory_warning"] = msg
            return resp
        except HTTPException: raise
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/manager/status")
    async def mgr_status():
        mm = get_model_manager()
        return {"loaded_models": mm.get_loaded_models(), "system_status": mm.get_system_status(),
               "queue_status": mm._loading_queue.get_queue_status()}

    @router.get("/v1/manager/models/{model_name}/status")
    async def mgr_model_status(model_name: str):
        return get_model_manager().get_model_status(model_name)

    @router.post("/v1/manager/models/{model_name}/priority")
    async def mgr_model_priority(model_name: str, pd: dict):
        return {"model": model_name, "priority": pd.get("priority", 0), "message": "Priority update requested"}

    @router.get("/admin")
    async def admin_ui():
        tp = (Path(sys._MEIPASS) / "fusion_mlx" / "gui_compat" / "templates" / "admin.html"
              if (hasattr(sys, "frozen") and sys.frozen and hasattr(sys, "_MEIPASS"))
              else (Path(sys.executable).parent / "fusion_mlx" / "gui_compat" / "templates" / "admin.html"
                    if (hasattr(sys, "frozen") and sys.frozen)
                    else Path(__file__).parent / "templates" / "admin.html"))
        if not tp.exists():
            for a in [Path(sys.executable).parent / "templates" / "admin.html"]:
                if a.exists(): tp = a; break
            if not tp.exists(): raise HTTPException(status_code=404, detail=f"Admin template not found")
        return HTMLResponse(content=tp.read_text(encoding="utf-8").replace("{{ version }}", __version__))

    @router.get("/v1/debug/model/{model_id:path}")
    async def dbg_model(model_id: str):
        try:
            from huggingface_hub import model_info as hmi
            m = hmi(model_id)
            dbg = {"model_id": model_id, "description": getattr(m, "description", None),
                  "tags": getattr(m, "tags", []), "library_name": getattr(m, "library_name", None),
                  "debug_log": []}
            mn = m.id.lower(); desc = (getattr(m, "description", "") or "")
            for pat in [r"(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+param", r"Parameters?:\s*(\d+(?:\.\d+)?)\s*[Bb]"]:
                ms = re.findall(pat, desc, re.IGNORECASE)
                if ms: dbg["params_billion"] = float(ms[0]); break
            bits = 4 if "4bit" in mn or "4-bit" in mn else (8 if "8bit" in mn or "8-bit" in mn else 16)
            dbg["quant_bits"] = bits
            return dbg
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    return router


# ── Standalone app ──

@asynccontextmanager
async def _lifespan(app: FastAPI):
    db = get_database_manager()
    logger.info(f"fusion_gui DB at: {db.database_path}")
    yield
    try:
        shutdown_inference_manager()
        shutdown_model_manager()
        db.close()
    except Exception: pass


def create_gui_router() -> FastAPI:
    app = FastAPI(title="fusion_gui API", version=__version__, lifespan=_lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(Exception)
    async def _exc(req, exc):
        logger.error(f"Unhandled: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/")
    async def _root(): return {"name": "fusion_gui API", "version": __version__, "status": "running"}

    @app.get("/health")
    async def _health(): return {"status": "healthy"}

    app.include_router(get_gui_compat_router())
    return app


def create_app() -> FastAPI:
    return create_gui_router()


def main():
    import argparse
    from uvicorn.config import LOGGING_CONFIG
    pa = argparse.ArgumentParser()
    pa.add_argument("--host", default="0.0.0.0")
    pa.add_argument("--port", type=int, default=8000)
    args = pa.parse_args()
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    LOGGING_CONFIG["formatters"]["default"]["format"] = fmt
    LOGGING_CONFIG["formatters"]["access"]["format"] = fmt
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info", log_config=LOGGING_CONFIG)


if __name__ == "__main__":
    main()
