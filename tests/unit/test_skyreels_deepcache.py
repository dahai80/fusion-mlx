"""B2 DeepCache 跳块缓存单元测试.

验证:
  - _deepcache_config: K=0 关 / K>1 开 / 非法值 / END clamp
  - _denoise_sample 路由: full 步 (cache_at) 与 cache 步 (skip_blocks) 交替
  - DeepCache 关闭 (K=0): 全步普通前向 (无 skip_blocks/cache_at)
  - B1 动态 CFG b=2->b=1 边界: cache 步 batch 不匹配自动降级 full 重缓存
"""

import mlx.core as mx

from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsBasePipeline
from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsPipelineConfig


def _make_base(n_layers=40):
    # 绕过 __init__, 手工装最小 pipeline (stub DiT + config).
    pipe = SkyReelsBasePipeline.__new__(SkyReelsBasePipeline)
    pipe.config = SkyReelsPipelineConfig(branch="r2v")
    pipe.config.num_inference_steps = 4
    pipe.dit = _StubDiT(n_layers=n_layers)
    pipe.step_strategy = _StubStepStrategy()
    return pipe


class _StubStepStrategy:
    def reset(self):
        pass

    def set_current_step(self, step_idx):
        pass


class _StubDiT:
    # 记录每次 __call__ 的 (mode, batch, skip_blocks, cache_at).
    # noise_pred = x (形状对齐 latent_input); captured = x (dummy 残差).
    # _supports_deepcache=True 使 pipeline 启用 DeepCache 路由.
    def __init__(self, n_layers=40):
        self.blocks = [object()] * n_layers
        self.patch_size = (1, 2, 2)
        self._supports_deepcache = True
        self.calls = []

    def __call__(
        self,
        x,
        t,
        context,
        seq_lens,
        grid_sizes,
        context_lens=None,
        rope_cos_sin=None,
        attn_mask=None,
        skip_blocks=None,
        cached_residual=None,
        cache_at=None,
    ):
        mode = "cached" if skip_blocks is not None else "full"
        self.calls.append(
            {
                "mode": mode,
                "batch": x.shape[0],
                "skip_blocks": skip_blocks,
                "cache_at": cache_at,
            }
        )
        if cache_at is not None:
            return x, x
        return x


def _stub_denoise_deps(monkeypatch, n_steps):
    # 桩掉 scheduler / flicker / perform_guidance / mx.eval, 只测路由.
    import fusion_mlx.video.skyreels_v3.pipelines as pm

    class _StubScheduler:
        def __init__(self, *a, **kw):
            self.timesteps = mx.array([float(i) for i in range(n_steps)])

        def set_timesteps(self, *a, **kw):
            pass

        def step(self, noise_pred, t, latents):
            class _R:
                prev_sample = latents

            return _R()

    monkeypatch.setattr(pm, "FlowUniPCMultistepScheduler", _StubScheduler)

    class _StubFlicker:
        def __init__(self, *a, **kw):
            self.enable_boundary_align = False

        def reset_step_filter(self):
            pass

        def filter_step(self, x):
            return x

        def smooth_temporal(self, x):
            return x

        def align_boundary(self, x, y):
            return x

    monkeypatch.setattr(pm, "TemporalFlickerFix", _StubFlicker)
    monkeypatch.setattr(
        pm, "_flicker_cfg_for_branch", lambda branch: _StubFlicker()
    )

    def _pg(noise_pred, guidance_scale, *, cond_first=False):
        # b=2 -> 取前半 (cond); b=1 -> 原样 (guidance=1.0 不调)
        b = noise_pred.shape[0]
        if b >= 2:
            return noise_pred[: b // 2]
        return noise_pred

    monkeypatch.setattr(pm, "perform_guidance", _pg)
    monkeypatch.setattr(mx, "eval", lambda *a, **kw: None)


def _run_denoise(pipe, n_steps, monkeypatch):
    _stub_denoise_deps(monkeypatch, n_steps)
    latents = mx.zeros((1, 16, 3, 4, 4))
    context = mx.zeros((1, 10, 4096))
    pipe._denoise_sample(
        latents,
        context,
        seq_lens=[12],
        grid_sizes=[(3, 2, 2)],
    )
    return pipe.dit.calls


# ---------------------------------------------------------------------------
# _deepcache_config
# ---------------------------------------------------------------------------


def test_deepcache_config_disabled_default(monkeypatch):
    monkeypatch.delenv("FUSION_SKYREELS_DEEPCACHE_K", raising=False)
    pipe = _make_base()
    assert pipe._deepcache_config() == (0, 0)


def test_deepcache_config_disabled_zero(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DEEPCACHE_K", "0")
    pipe = _make_base()
    assert pipe._deepcache_config() == (0, 0)


def test_deepcache_config_enabled_default_end(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DEEPCACHE_K", "3")
    monkeypatch.delenv("FUSION_SKYREELS_DEEPCACHE_END", raising=False)
    pipe = _make_base(n_layers=40)
    k, end = pipe._deepcache_config()
    assert k == 3
    assert end == 20  # 40 // 2


def test_deepcache_config_invalid_k(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DEEPCACHE_K", "abc")
    pipe = _make_base()
    assert pipe._deepcache_config() == (0, 0)


def test_deepcache_config_end_clamp(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DEEPCACHE_K", "3")
    monkeypatch.setenv("FUSION_SKYREELS_DEEPCACHE_END", "999")
    pipe = _make_base(n_layers=40)
    k, end = pipe._deepcache_config()
    assert k == 3
    assert end == 38  # n_layers - 2

    monkeypatch.setenv("FUSION_SKYREELS_DEEPCACHE_END", "0")
    k, end = pipe._deepcache_config()
    assert end == 1  # max(1, ...)


def test_deepcache_config_end_override(monkeypatch):
    monkeypatch.setenv("FUSION_SKYREELS_DEEPCACHE_K", "2")
    monkeypatch.setenv("FUSION_SKYREELS_DEEPCACHE_END", "10")
    pipe = _make_base(n_layers=40)
    k, end = pipe._deepcache_config()
    assert (k, end) == (2, 10)


# ---------------------------------------------------------------------------
# _denoise_sample 路由
# ---------------------------------------------------------------------------


def test_denoise_routes_deepcache_alternating(monkeypatch):
    # K=2, 4 步: step0 full, step1 cached, step2 full, step3 cached.
    # 关闭 B1 动态 CFG (全 b=2) 避免边界干扰.
    monkeypatch.setenv("FUSION_SKYREELS_DYNAMIC_CFG", "0")
    monkeypatch.setenv("FUSION_SKYREELS_DEEPCACHE_K", "2")
    monkeypatch.delenv("FUSION_SKYREELS_DEEPCACHE_END", raising=False)
    pipe = _make_base(n_layers=40)
    pipe.config.num_inference_steps = 4
    calls = _run_denoise(pipe, 4, monkeypatch)
    modes = [c["mode"] for c in calls]
    assert modes == ["full", "cached", "full", "cached"]
    # full 步带 cache_at, cache 步带 skip_blocks
    assert calls[0]["cache_at"] == 20
    assert calls[0]["skip_blocks"] is None
    assert calls[1]["skip_blocks"] == (0, 20)
    assert calls[1]["cache_at"] is None


def test_denoise_routes_deepcache_off(monkeypatch):
    # K=0: 全步普通前向, 无 skip_blocks / cache_at.
    monkeypatch.setenv("FUSION_SKYREELS_DYNAMIC_CFG", "0")
    monkeypatch.delenv("FUSION_SKYREELS_DEEPCACHE_K", raising=False)
    pipe = _make_base(n_layers=40)
    pipe.config.num_inference_steps = 3
    calls = _run_denoise(pipe, 3, monkeypatch)
    assert len(calls) == 3
    assert all(c["skip_blocks"] is None and c["cache_at"] is None for c in calls)


def test_denoise_routes_batch_mismatch_downgrade(monkeypatch):
    # B1 动态 CFG: step0 b=2 (keep=1), step1/2 b=1. DeepCache K=3.
    # step0: full (b=2, cache b=2)
    # step1: cache 步 b=1, dc_cached.shape[0]=2 != 1 -> MISS -> full (重缓存 b=1)
    # step2: cache 步 b=1, dc_cached.shape[0]=1 == 1 -> cached
    monkeypatch.setenv("FUSION_SKYREELS_DYNAMIC_CFG", "1")
    monkeypatch.setenv("FUSION_SKYREELS_CFG_KEEP_RATIO", "0.5")  # int(3*0.5)=1
    monkeypatch.setenv("FUSION_SKYREELS_DEEPCACHE_K", "3")
    monkeypatch.delenv("FUSION_SKYREELS_DEEPCACHE_END", raising=False)
    pipe = _make_base(n_layers=40)
    pipe.config.num_inference_steps = 3
    calls = _run_denoise(pipe, 3, monkeypatch)
    # batch 演变: step0 b=2, step1 b=1 (full 重缓存), step2 b=1 (cached)
    assert calls[0]["mode"] == "full" and calls[0]["batch"] == 2
    assert calls[1]["mode"] == "full" and calls[1]["batch"] == 1  # miss 降级
    assert calls[2]["mode"] == "cached" and calls[2]["batch"] == 1


# ---------------------------------------------------------------------------
# 真实 DiT __call__ 跳块/捕获/注入 (tiny 配置, 免权重)
# 验证 SkyReelsR2VDiT.__call__ 的 skip_blocks/cache_at/cached_residual
# 在真实 WanAttentionBlock 前向上不崩 + 形状正确 + cache_at 非变异.
# ---------------------------------------------------------------------------

_TINY_CFG = dict(
    num_layers=4,
    dim=64,
    ffn_dim=128,
    num_heads=4,
    num_kv_heads=4,
    patch_size=(1, 2, 2),
    in_dim=16,
    out_dim=16,
    text_dim=64,
    text_len=8,
    freq_dim=32,
)


def _tiny_dit():
    from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

    dit = SkyReelsR2VDiT(_TINY_CFG)
    return dit


def _tiny_inputs():
    # x=[B,C,T,H,W]; patch (1,2,2) -> L = T*(H/2)*(W/2) = 2*2*2 = 8
    x = mx.random.normal((1, 16, 2, 4, 4))
    t = mx.array([0.5])
    ctx = mx.random.normal((1, 8, 64))
    seq_lens = [8]
    grid_sizes = [(2, 2, 2)]
    return x, t, ctx, seq_lens, grid_sizes


def test_real_dit_cache_at_is_capture_only():
    # cache_at 只捕获残差, 不改变输出: full(out,resid) == plain(out).
    dit = _tiny_dit()
    x, t, ctx, seq_lens, grid_sizes = _tiny_inputs()
    plain = dit(x, t, ctx, seq_lens, grid_sizes)
    full, resid = dit(x, t, ctx, seq_lens, grid_sizes, cache_at=1)
    mx.eval(plain, full, resid)
    assert plain.shape == (1, 16, 2, 4, 4)
    assert full.shape == (1, 16, 2, 4, 4)
    assert resid.shape == (1, 8, 64)  # [B, L, dim]
    diff = float(mx.max(mx.abs(mx.subtract(full, plain))))
    assert diff == 0.0, f"cache_at 改变了输出: diff={diff}"


def test_real_dit_skip_inject_runs():
    # skip_blocks=(0,1) + cached_residual 注入: 跑 block 2,3, 形状不变.
    dit = _tiny_dit()
    x, t, ctx, seq_lens, grid_sizes = _tiny_inputs()
    _, resid = dit(x, t, ctx, seq_lens, grid_sizes, cache_at=1)
    mx.eval(resid)
    cached = dit(
        x, t, ctx, seq_lens, grid_sizes, skip_blocks=(0, 1), cached_residual=resid
    )
    mx.eval(cached)
    assert cached.shape == (1, 16, 2, 4, 4)
