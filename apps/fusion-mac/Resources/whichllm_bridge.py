#!/usr/bin/env python3
"""whichllm bridge — hardware detection + model recommendations.

Called by FusionMLX macOS app via PythonRuntime subprocess.
Returns JSON to stdout.

Usage:
  python3 whichllm_bridge.py detect          # hardware only
  python3 whichllm_bridge.py recommend       # hardware + model recommendations
  
Exit code 0 on success, 1 on error.
"""

import json
import sys
import os

# whichllm may or may not be installed — provide graceful fallback
_WHICHLLM_AVAILABLE = False
try:
    from whichllm.hardware.detector import detect_hardware
    from whichllm.engine.ranker import rank_models
    from whichllm.models.fetcher import fetch_models, dicts_to_models, models_to_dicts
    from whichllm.models.grouper import group_models
    from whichllm.models.benchmark import fetch_benchmark_scores, load_benchmark_cache, save_benchmark_cache
    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.artifacts import attach_resolved_artifacts
    from whichllm.output.json_output import display_json as _  # ensure module importable
    _WHICHLLM_AVAILABLE = True
except ImportError:
    pass


def _format_hardware(hw) -> dict:
    """Serialize HardwareInfo to a dict."""
    return {
        "gpus": [
            {
                "name": g.name,
                "vendor": g.vendor,
                "vram_bytes": g.vram_bytes,
                "usable_vram_bytes": g.usable_vram_bytes,
                "memory_bandwidth_gbps": g.memory_bandwidth_gbps,
                "shared_memory": g.shared_memory,
            }
            for g in hw.gpus
        ],
        "cpu": hw.cpu_name,
        "cpu_cores": hw.cpu_cores,
        "ram_bytes": hw.ram_bytes,
        "ram_budget_bytes": hw.ram_budget_bytes,
        "budget_notes": hw.budget_notes,
        "disk_free_bytes": hw.disk_free_bytes,
        "os": hw.os,
    }


def _detect():
    """Detect hardware using whichllm."""
    if not _WHICHLLM_AVAILABLE:
        # Fallback: use stdlib only (no psutil dependency)
        import platform
        import subprocess
        cpu = platform.processor() or "Unknown"
        cores = os.cpu_count() or 0
        # Get RAM via sysctl (macOS)
        ram = 0
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], timeout=5, text=True
            ).strip()
            ram = int(out)
        except Exception:
            pass
        # Get disk via shutil
        disk = 0
        try:
            import shutil
            disk = shutil.disk_usage(os.path.expanduser("~")).free
        except Exception:
            pass
        return {
            "gpus": [],
            "cpu": cpu,
            "cpu_cores": cores,
            "ram_bytes": ram,
            "disk_free_bytes": disk,
            "os": platform.system().lower(),
            "_fallback": True,
            "_note": "whichllm not installed; detected via sysctl",
        }
    hw = detect_hardware()
    return _format_hardware(hw)


def _recommend():
    """Detect hardware + fetch models + rank recommendations."""
    if not _WHICHLLM_AVAILABLE:
        return {"error": "whichllm not installed", "hardware": _detect(), "models": []}
    
    import asyncio
    
    async def run():
        hw = detect_hardware()
        hardware_dict = _format_hardware(hw)
        
        # Try cache first
        cached = load_cache()
        if cached is not None:
            models = dicts_to_models(cached)
        else:
            models = await fetch_models(include_vision=False)
            save_cache(models_to_dicts(models))
        
        group_models(models)
        all_models = []
        for family in models:
            all_models.append(family.base_model)
            all_models.extend(family.variants)
        
        # Fetch or load benchmark cache
        bench_scores = load_benchmark_cache()
        if bench_scores is None:
            try:
                bench_scores = await fetch_benchmark_scores()
                save_benchmark_cache(bench_scores)
            except Exception:
                bench_scores = {}
        
        results = rank_models(
            all_models, hw,
            context_length=4096, top_n=10,
            require_direct_top=True, min_params_b=None,
        )
        try:
            attach_resolved_artifacts(results, all_models)
        except Exception:
            pass
        
        models_out = []
        for i, r in enumerate(results[:10]):
            q = r.gguf_variant
            models_out.append({
                "rank": i + 1,
                "model_id": r.model.id,
                "model_name": r.model.name,
                "artifact_repo_id": r.artifact_model.id if r.artifact_model else None,
                "artifact_filename": r.artifact_variant.filename if r.artifact_variant else None,
                "parameter_count": r.model.parameter_count,
                "parameter_count_active": r.model.parameter_count_active,
                "architecture": r.model.architecture,
                "context_length": r.model.context_length,
                "quant_type": q.quant_type if q else None,
                "file_size_bytes": q.file_size_bytes if q else 0,
                "vram_required_bytes": r.vram_required_bytes,
                "vram_available_bytes": r.vram_available_bytes,
                "estimated_tok_per_sec": r.estimated_tok_per_sec,
                "speed_confidence": r.speed_confidence,
                "quality_score": round(r.quality_score, 1),
                "fit_type": r.fit_type,
                "benchmark_status": r.benchmark_status,
                "uses_multi_gpu": r.uses_multi_gpu,
            })
        
        return {"hardware": hardware_dict, "models": models_out}
    
    return asyncio.run(run())


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: whichllm_bridge.py detect|recommend"}))
        sys.exit(1)
    
    command = sys.argv[1]
    if command == "detect":
        result = _detect()
    elif command == "recommend":
        result = _recommend()
    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
        sys.exit(1)
    
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if "error" not in result else 1)


if __name__ == "__main__":
    main()
