# SPDX-License-Identifier: Apache-2.0
"""Migration wizard API routes — 7-step HF→MLX conversion pipeline.

Callers: fusion_mlx.admin.routes (router registration via include_router)
API: /admin/api/migrate/analyze|download|download-status|convert|codegen|validate|register|list
Schema: MigrateAnalyzeRequest, MigrateConvertRequest, MigrateQuantizeRequest (pydantic BaseModel)
User instruction verbatim: "做一个端到端的功能，做模型迁移和量化的功能，以openpangu为例，把迁移的每个步骤展现在GUI上"
"""

import asyncio
import logging
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .auth import require_admin
from .helpers import _get_hf_downloader, _get_model_dirs

logger = logging.getLogger(__name__)

_router = APIRouter()

MIGRATION_DIR = os.path.expanduser("~/.fusion-mlx/migrations")


class MigrateAnalyzeRequest(BaseModel):
    hf_id: str
    mirror: bool = False


class MigrateDownloadRequest(BaseModel):
    hf_id: str
    hf_token: str = ""
    mirror: bool = False


class MigrateConvertRequest(BaseModel):
    migration_id: str
    quant_bits: int = 0
    quant_group_size: int = 64


class MigrateCodegenRequest(BaseModel):
    migration_id: str


class MigrateValidateRequest(BaseModel):
    migration_id: str
    prompt: str = "Hello, how are you?"
    max_tokens: int = 32


class MigrateRegisterRequest(BaseModel):
    migration_id: str
    model_name: str = ""


def _migration_dir(mid: str) -> str:
    return os.path.join(MIGRATION_DIR, mid)


def _ensure_migration_dir(mid: str) -> str:
    d = _migration_dir(mid)
    os.makedirs(d, exist_ok=True)
    return d


@_router.post("/api/migrate/analyze")
async def analyze_model(
    request: MigrateAnalyzeRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..migrate.analyzer import analyze_model

    try:
        result = await asyncio.to_thread(analyze_model, request.hf_id, request.mirror)
        return {"success": True, "analysis": asdict(result)}
    except Exception as e:
        logger.exception("Analyze failed for %s", request.hf_id)
        raise HTTPException(status_code=400, detail=str(e))


@_router.post("/api/migrate/download")
async def download_weights(
    request: MigrateDownloadRequest,
    is_admin: bool = Depends(require_admin),
):
    dl = _get_hf_downloader()
    if dl is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    mid = str(uuid.uuid4())[:8]
    mdir = _ensure_migration_dir(mid)

    try:
        task = await dl.start_download(
            request.hf_id,
            hf_token=request.hf_token or None,
        )
        import json

        meta = {
            "migration_id": mid,
            "hf_id": request.hf_id,
            "download_task_id": task.task_id,
        }
        with open(os.path.join(mdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Migration %s: download started for %s", mid, request.hf_id)
        return {"success": True, "migration_id": mid, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.get("/api/migrate/download-status/{migration_id}")
async def download_status(
    migration_id: str,
    is_admin: bool = Depends(require_admin),
):
    dl = _get_hf_downloader()
    if dl is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    mdir = _migration_dir(migration_id)
    meta_path = os.path.join(mdir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Migration not found")

    import json

    with open(meta_path) as f:
        meta = json.load(f)

    task_id = meta.get("download_task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="No download task")

    tasks = dl.get_tasks()
    task = next((t for t in tasks if t.get("task_id") == task_id), None)
    if task and task.get("status") == "completed":
        hf_id = meta.get("hf_id", "")
        if hf_id and not meta.get("hf_dir"):
            model_dir = os.path.expanduser("~/.fusion-mlx/models")
            candidate = os.path.join(model_dir, hf_id)
            if os.path.isdir(candidate):
                meta["hf_dir"] = candidate
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

    return {"success": True, "migration_id": migration_id, "task": task}


@_router.post("/api/migrate/convert")
async def convert_weights(
    request: MigrateConvertRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..migrate.architectures import HF_ARCH_TO_TEMPLATE, KNOWN_TEMPLATES, match_template
    from ..migrate.converter import convert_model

    mdir = _migration_dir(request.migration_id)
    meta_path = os.path.join(mdir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Migration not found")

    import json

    with open(meta_path) as f:
        meta = json.load(f)

    hf_dir = meta.get("hf_dir", "")
    if not hf_dir or not os.path.isdir(hf_dir):
        raise HTTPException(status_code=400, detail="HF weights not downloaded yet")

    try:
        config_path = os.path.join(hf_dir, "config.json")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=400, detail="config.json not found in HF dir")
        with open(config_path) as f:
            hf_config = json.load(f)

        arch_list = hf_config.get("architectures", [])
        arch_name = arch_list[0] if arch_list else hf_config.get("model_type", "")
        template, diff = match_template(arch_name, hf_config)

        output_dir = os.path.join(mdir, "mlx_model")
        result = await asyncio.to_thread(
            convert_model,
            hf_dir,
            output_dir,
            hf_config,
            template,
            quant_bits=request.quant_bits,
            quant_group_size=request.quant_group_size,
        )

        meta["mlx_dir"] = output_dir
        meta["template"] = template.name
        meta["diff"] = diff
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {"success": True, "result": asdict(result)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Convert failed for migration %s", request.migration_id)
        raise HTTPException(status_code=400, detail=str(e))


@_router.post("/api/migrate/codegen")
async def codegen(
    request: MigrateCodegenRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..migrate.architectures import KNOWN_TEMPLATES, match_template
    from ..migrate.codegen import generate_model_code

    mdir = _migration_dir(request.migration_id)
    meta_path = os.path.join(mdir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Migration not found")

    import json

    with open(meta_path) as f:
        meta = json.load(f)

    template_name = meta.get("template", "")
    hf_dir = meta.get("hf_dir", "")

    try:
        template = KNOWN_TEMPLATES.get(template_name)
        if template is None:
            config_path = os.path.join(hf_dir, "config.json") if hf_dir else ""
            if not config_path or not os.path.exists(config_path):
                raise HTTPException(status_code=400, detail="Cannot resolve template: no config.json")
            with open(config_path) as f:
                hf_config = json.load(f)
            arch_list = hf_config.get("architectures", [])
            arch_name = arch_list[0] if arch_list else hf_config.get("model_type", "")
            template, _ = match_template(arch_name, hf_config)

        if hf_dir and os.path.exists(os.path.join(hf_dir, "config.json")):
            with open(os.path.join(hf_dir, "config.json")) as f:
                hf_config = json.load(f)
        else:
            hf_config = {}

        output_dir = os.path.join(mdir, "codegen")
        result = await asyncio.to_thread(
            generate_model_code,
            template,
            hf_config,
            output_dir,
        )

        meta["codegen_dir"] = output_dir
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Copy generated model .py to mlx_model dir so mlx_lm can load it
        mlx_dir = meta.get("mlx_dir", "")
        if mlx_dir and os.path.isdir(mlx_dir):
            for gen_file in result.files_generated:
                if gen_file.endswith(".py"):
                    dest = os.path.join(mlx_dir, os.path.basename(gen_file))
                    shutil.copy2(gen_file, dest)
                    logger.info("Copied %s → %s", gen_file, dest)

        return {"success": True, "result": asdict(result)}
    except Exception as e:
        logger.exception("Codegen failed for migration %s", request.migration_id)
        raise HTTPException(status_code=400, detail=str(e))


@_router.post("/api/migrate/validate")
async def validate(
    request: MigrateValidateRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..migrate.validator import validate_model

    mdir = _migration_dir(request.migration_id)
    meta_path = os.path.join(mdir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Migration not found")

    import json

    with open(meta_path) as f:
        meta = json.load(f)

    mlx_dir = meta.get("mlx_dir", "")
    if not mlx_dir or not os.path.isdir(mlx_dir):
        raise HTTPException(status_code=400, detail="MLX model not converted yet")

    try:
        result = await asyncio.to_thread(
            validate_model,
            mlx_dir,
            request.prompt,
            request.max_tokens,
        )
        meta["validated"] = result.success
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return {"success": True, "result": asdict(result)}
    except Exception as e:
        logger.exception("Validate failed for migration %s", request.migration_id)
        raise HTTPException(status_code=400, detail=str(e))


@_router.post("/api/migrate/register")
async def register_model(
    request: MigrateRegisterRequest,
    is_admin: bool = Depends(require_admin),
):
    mdir = _migration_dir(request.migration_id)
    meta_path = os.path.join(mdir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Migration not found")

    import json

    with open(meta_path) as f:
        meta = json.load(f)

    mlx_dir = meta.get("mlx_dir", "")
    if not mlx_dir or not os.path.isdir(mlx_dir):
        raise HTTPException(status_code=400, detail="MLX model not converted yet")

    model_name = request.model_name or meta.get("hf_id", "").split("/")[-1]
    model_dirs = _get_model_dirs()
    if not model_dirs:
        raise HTTPException(status_code=503, detail="No model directories configured")

    dest = os.path.join(model_dirs[0], model_name)
    if os.path.exists(dest):
        raise HTTPException(
            status_code=409, detail=f"Model '{model_name}' already exists"
        )

    try:
        shutil.copytree(mlx_dir, dest)
        meta["registered"] = True
        meta["registered_path"] = dest
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Registered migrated model at %s", dest)
        return {"success": True, "path": dest, "model_name": model_name}
    except Exception as e:
        logger.exception("Register failed for migration %s", request.migration_id)
        raise HTTPException(status_code=400, detail=str(e))


@_router.get("/api/migrate/list")
async def list_migrations(is_admin: bool = Depends(require_admin)):
    if not os.path.isdir(MIGRATION_DIR):
        return {"success": True, "migrations": []}

    import json

    migrations = []
    for d in sorted(Path(MIGRATION_DIR).iterdir()):
        meta_path = d / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["migration_id"] = d.name
            migrations.append(meta)

    return {"success": True, "migrations": migrations}


router = _router
