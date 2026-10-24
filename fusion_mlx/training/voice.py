# SPDX-License-Identifier: Apache-2.0
# Voice fine-tune backend (#1336).
#
# Trains a Kokoro voice package from paired (audio, transcript) samples. The
# Kokoro family is the only mlx-audio TTS family whose `voice` argument is a
# plain tensor ("voice pack", shape [510, 1, 256]) that the pipeline loads
# from a .safetensors file, so a trained pack drops straight into the
# inference path with no engine change: `voice="<path>/voice.safetensors"`.
#
# What is trained: the pack rows the model selects by phoneme count. The
# frozen decoder turns a pack row into style codes, the row drives timbre,
# and nothing else in the model changes — so the package stays small (one
# tensor) and loads in the same pass as a stock voice.
#
# Why derivative-free: the waveform path through MLXSTFT.inverse is not
# differentiable in this mlx-audio build (`mlx_unwrap` + `dsp.istft` return
# NaN gradients), so the acoustic objective cannot be backpropagated. The
# generator's magnitude head IS differentiable, but training on it does not
# transfer: measured on the same generation, the internal magnitude LTAS and
# the rendered audio LTAS differ substantially (mean |delta| ~1.05), and a
# pack that minimizes the internal objective renders *further* from the
# reference than the stock voice. Training therefore optimizes the rendered
# waveform directly with SPSA (two forward passes per step, no gradients).

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

VOICE_BASE_DIR = Path(
    os.path.expanduser(os.environ.get("FUSION_VOICE_DIR") or "~/.fusion-mlx/voices")
)

VOICE_TRAIN_SAMPLE_RATE = 24000
VOICE_TRAIN_MIN_SECONDS = 1.0
VOICE_TRAIN_MIN_BYTES = 1024
VOICE_TRAIN_MIN_TOTAL_SECONDS = 3.0

SUPPORTED_BACKENDS = {
    "kokoro": {
        "family": "kokoro",
        "artifact": "voice.safetensors",
        "trainable": "voice pack (style rows)",
        "objective": "rendered-audio long-term spectrum + dynamics (SPSA)",
        "why": "the only mlx-audio TTS family whose voice argument is a "
        "loadable tensor; CosyVoice2/GPT-SoVITS need a PyTorch training "
        "stack that is not part of the Apple-Silicon MLX runtime",
    },
}


@dataclass
class VoiceTrainConfig:
    steps: int = 300
    learning_rate: float = 0.02
    perturbation: float = 0.005
    decay: float = 0.101
    base_voice: str = "af_heart"
    lang_code: str = "a"
    max_samples: int = 0
    dynamics_weight: float = 0.5
    level_weight: float = 0.3
    anchor_weight: float = 0.02
    seed: int = 0


@dataclass
class VoiceSample:
    audio_path: str
    text: str
    phonemes: str = ""
    duration: float = 0.0
    target_ltas: np.ndarray | None = field(default=None, repr=False)
    target_spread: np.ndarray | None = field(default=None, repr=False)
    target_rms: float = 0.0


class VoiceDatasetError(ValueError):
    pass


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != VOICE_TRAIN_SAMPLE_RATE:
        import librosa

        mono = librosa.resample(
            mono, orig_sr=sr, target_sr=VOICE_TRAIN_SAMPLE_RATE, res_type="soxr_hq"
        )
        sr = VOICE_TRAIN_SAMPLE_RATE
    return mono.astype(np.float32), sr


def _dataset_pairs(dataset_dir: Path) -> list[tuple[Path, str]]:
    meta = dataset_dir / "metadata.csv"
    jsonl = dataset_dir / "metadata.jsonl"
    pairs: list[tuple[Path, str]] = []
    if meta.is_file():
        for lineno, line in enumerate(meta.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                raise VoiceDatasetError(
                    f"metadata.csv line {lineno}: expected 'file|text', got {line!r}"
                )
            audio = (dataset_dir / parts[0].strip()).resolve()
            pairs.append((audio, parts[1].strip()))
        return pairs
    if jsonl.is_file():
        for lineno, line in enumerate(
            jsonl.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VoiceDatasetError(
                    f"metadata.jsonl line {lineno}: invalid JSON ({exc})"
                ) from exc
            audio_rel = item.get("audio") or item.get("file") or item.get("path")
            text = item.get("text") or item.get("transcript")
            if not audio_rel or not text:
                raise VoiceDatasetError(
                    f"metadata.jsonl line {lineno}: needs 'audio' and 'text'"
                )
            pairs.append(((dataset_dir / audio_rel).resolve(), str(text).strip()))
        return pairs
    for audio in sorted(dataset_dir.glob("*.wav")):
        txt = audio.with_suffix(".txt")
        if not txt.is_file():
            logger.warning("voice dataset: %s has no transcript, skipping", audio.name)
            continue
        pairs.append((audio.resolve(), txt.read_text(encoding="utf-8").strip()))
    return pairs


def load_dataset(dataset_dir: str | Path, max_samples: int = 0) -> list[VoiceSample]:
    root = Path(os.path.expanduser(str(dataset_dir)))
    if not root.is_dir():
        raise VoiceDatasetError(f"dataset directory not found: {root}")
    pairs = _dataset_pairs(root)
    if not pairs:
        raise VoiceDatasetError(
            f"no training pairs in {root}: provide metadata.csv, metadata.jsonl "
            "or <name>.wav + <name>.txt"
        )
    if max_samples and max_samples > 0:
        pairs = pairs[:max_samples]

    samples: list[VoiceSample] = []
    problems: list[str] = []
    total_seconds = 0.0
    for audio_path, text in pairs:
        if not audio_path.is_file():
            problems.append(f"{audio_path}: audio file missing")
            continue
        if audio_path.stat().st_size < VOICE_TRAIN_MIN_BYTES:
            problems.append(f"{audio_path}: file is empty or truncated")
            continue
        if not text:
            problems.append(f"{audio_path}: transcript is empty")
            continue
        try:
            mono, sr = _read_audio(audio_path)
        except Exception as exc:
            problems.append(f"{audio_path}: cannot decode audio ({exc})")
            continue
        duration = len(mono) / sr
        if duration < VOICE_TRAIN_MIN_SECONDS:
            problems.append(
                f"{audio_path}: {duration:.2f}s is shorter than "
                f"{VOICE_TRAIN_MIN_SECONDS:.0f}s"
            )
            continue
        samples.append(
            VoiceSample(audio_path=str(audio_path), text=text, duration=duration)
        )
        total_seconds += duration

    if problems:
        raise VoiceDatasetError(
            "dataset rejected: "
            + "; ".join(problems[:10])
            + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else "")
        )
    if total_seconds < VOICE_TRAIN_MIN_TOTAL_SECONDS:
        raise VoiceDatasetError(
            f"dataset is {total_seconds:.1f}s of audio; fine-tuning needs at "
            f"least {VOICE_TRAIN_MIN_TOTAL_SECONDS:.0f}s of clean speech"
        )
    logger.info(
        "voice dataset: %d sample(s), %.1fs total from %s",
        len(samples),
        total_seconds,
        root,
    )
    return samples


def _spectrum_stats(mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Log-domain long-term spectrum + log-domain dynamics, each with the
    # overall level removed (mean across frequency). Working in log space
    # keeps a handful of loud frames from dominating the average, and the
    # level removal makes these two terms depend on spectral SHAPE only; the
    # separate level term in the loss is what matches loudness.
    mag = np.asarray(mag, dtype=np.float64).reshape(mag.shape[-2], mag.shape[-1])
    log_mag = np.log(mag + 1e-6)
    mean = log_mag.mean(axis=1)
    mean = mean - mean.mean()
    spread = log_mag.std(axis=1)
    spread = spread - spread.mean()
    return mean.astype(np.float32), spread.astype(np.float32)


def _target_stats(
    audio: np.ndarray, n_fft: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    spec = np.abs(
        librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            center=True,
        )
    )
    return _spectrum_stats(spec)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)) + 1e-9)


def _phonemize(pipe, text: str) -> list[str]:
    _, tokens = pipe.g2p(text)
    out: list[str] = []
    for _gs, ps, _tks in pipe.en_tokenize(tokens):
        if ps:
            out.append(ps[:510])
    return out


def write_voice_package(
    out_dir: Path,
    pack,
    manifest: dict,
    metrics: dict,
) -> str:
    import mlx.core as mx

    out_dir.mkdir(parents=True, exist_ok=True)
    voice_path = out_dir / "voice.safetensors"
    mx.save_safetensors(str(voice_path), {"voice": pack})
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    logger.info(
        "voice package written: %s (spectral distance %.4f -> %.4f, stock voice %.4f)",
        voice_path,
        metrics.get("initial_spectral_distance", 0.0),
        metrics.get("final_spectral_distance", 0.0),
        metrics.get("stock_spectral_distance", 0.0),
    )
    if not metrics.get("beat_stock_voice", False):
        logger.warning(
            "voice package %s did not beat the stock voice on spectral distance "
            "(final %.4f >= stock %.4f); the package is written but flagged as "
            "regressed in manifest.json quality",
            out_dir,
            metrics.get("final_spectral_distance", 0.0),
            metrics.get("stock_spectral_distance", 0.0),
        )
    return str(voice_path)


def train_kokoro_voice(
    dataset_dir: str | Path,
    model_name: str,
    voice_id: str,
    config: VoiceTrainConfig | None = None,
    tenant_id: str = "default",
    output_dir: str | Path | None = None,
    on_step=None,
    should_cancel=None,
) -> dict:
    import mlx.core as mx
    from mlx_audio.tts.generate import load_model

    cfg = config or VoiceTrainConfig()
    samples = load_dataset(dataset_dir, cfg.max_samples)
    out_dir = Path(output_dir or (VOICE_BASE_DIR / voice_id))
    t0 = time.time()

    model = load_model(model_name, model_type="kokoro")
    pipe = model._get_pipeline(cfg.lang_code)
    generator = model.decoder.generator
    n_fft = int(getattr(generator, "post_n_fft", 20))
    hop = int(getattr(generator.stft, "hop_length", 5))

    stock = np.array(pipe.load_voice(cfg.base_voice), dtype=np.float32)
    if stock.ndim != 3:
        raise ValueError(
            f"voice pack for {cfg.base_voice!r} has shape {stock.shape}; "
            "expected [rows, 1, channels]"
        )
    rows = stock.shape[0]

    usable = []
    for sample in samples:
        phonemes = _phonemize(pipe, sample.text)
        if not phonemes:
            logger.warning(
                "voice dataset: no phonemes for %s, skipping", sample.audio_path
            )
            continue
        audio, _sr = _read_audio(Path(sample.audio_path))
        sample.target_ltas, sample.target_spread = _target_stats(audio, n_fft, hop)
        sample.target_rms = _rms(audio)
        for ps in phonemes:
            if len(ps) - 1 >= rows:
                raise VoiceDatasetError(
                    f"{sample.audio_path}: phoneme string is {len(ps)} long, "
                    f"but the {cfg.base_voice!r} pack has only {rows} rows"
                )
        usable.append((sample, phonemes))
    if not usable:
        raise VoiceDatasetError("no sample produced usable phonemes")

    def _render(pack_np, row_index, ps) -> np.ndarray:
        row = mx.array(pack_np[row_index])
        out = model(ps, row, 1.0, return_output=True)
        return np.array(out.audio).reshape(-1)

    def _audio_loss(audio, sample, row_index, pack_np) -> float:
        mean, spread = _target_stats(audio, n_fft, hop)
        timbre = float(np.mean(np.abs(mean - sample.target_ltas)))
        dynamics = float(np.mean(np.abs(spread - sample.target_spread)))
        level = abs(np.log(_rms(audio) / sample.target_rms))
        loss = timbre + cfg.dynamics_weight * dynamics + cfg.level_weight * level
        if cfg.anchor_weight > 0:
            delta = pack_np[row_index] - stock[row_index]
            loss += cfg.anchor_weight * float(np.mean(delta * delta))
        return loss

    def _measure(pack_np) -> tuple[float, float]:
        losses: list[float] = []
        timbres: list[float] = []
        for sample, phoneme_list in usable:
            for ps in phoneme_list:
                row_index = len(ps) - 1
                audio = _render(pack_np, row_index, ps)
                mean, _spread = _target_stats(audio, n_fft, hop)
                timbres.append(float(np.mean(np.abs(mean - sample.target_ltas))))
                losses.append(_audio_loss(audio, sample, row_index, pack_np))
        finite = [v for v in losses if np.isfinite(v)]
        if not finite:
            raise RuntimeError(f"voice training for {voice_id} produced no finite loss")
        return float(np.mean(finite)), float(np.mean(timbres))

    initial, _initial_timbre = _measure(stock)
    stock_loss, stock_timbre = initial, _initial_timbre
    pack = stock.copy()
    rng = np.random.default_rng(cfg.seed)
    history: list[float] = []
    touched: set[int] = set()
    skipped = 0

    for step in range(1, cfg.steps + 1):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("cancelled")
        sample, phoneme_list = usable[(step - 1) % len(usable)]
        ps = phoneme_list[(step - 1) % len(phoneme_list)]
        row_index = len(ps) - 1
        a_k = cfg.learning_rate / (step**cfg.decay)
        c_k = cfg.perturbation / (step**cfg.decay)
        delta = rng.choice(np.array([-1.0, 1.0]), size=pack[row_index].shape).astype(
            np.float32
        )

        plus = pack.copy()
        plus[row_index] += c_k * delta
        minus = pack.copy()
        minus[row_index] -= c_k * delta
        loss_plus = _audio_loss(_render(plus, row_index, ps), sample, row_index, plus)
        loss_minus = _audio_loss(
            _render(minus, row_index, ps), sample, row_index, minus
        )

        # SPSA: one two-sided measurement estimates the gradient along a
        # random direction; the direction itself is the gradient estimate's
        # multiplier, so only 2 forward passes are needed per step.
        if not (np.isfinite(loss_plus) and np.isfinite(loss_minus)):
            skipped += 1
            logger.warning(
                "voice train %s step %d skipped: non-finite loss (%s, %s)",
                voice_id,
                step,
                loss_plus,
                loss_minus,
            )
            continue
        estimate = (loss_plus - loss_minus) / (2.0 * c_k) * delta
        if not np.all(np.isfinite(estimate)):
            skipped += 1
            logger.warning(
                "voice train %s step %d skipped: non-finite gradient estimate",
                voice_id,
                step,
            )
            continue
        pack[row_index] -= a_k * estimate
        touched.add(row_index)
        value = 0.5 * (loss_plus + loss_minus)
        history.append(value)

        if step == 1 or step % 10 == 0 or step == cfg.steps:
            logger.info(
                "voice train %s step %d/%d loss=%.5f rows=%d",
                voice_id,
                step,
                cfg.steps,
                value,
                len(touched),
            )
            if on_step is not None:
                on_step(
                    step,
                    cfg.steps,
                    {
                        "train_loss": value,
                        "spectral_distance": value,
                        "skipped_steps": skipped,
                        "rows_touched": len(touched),
                    },
                )

    final, final_timbre = _measure(pack)
    beat = bool(final_timbre < stock_timbre)
    metrics = {
        "initial_spectral_distance": round(initial, 6),
        "final_spectral_distance": round(final, 6),
        "stock_spectral_distance": round(stock_loss, 6),
        "final_timbre_distance": round(final_timbre, 6),
        "stock_timbre_distance": round(stock_timbre, 6),
        "improved": bool(final < initial),
        "beat_stock_voice": beat,
        "steps": cfg.steps,
        "skipped_steps": skipped,
        "samples": len(usable),
        "rows_touched": len(touched),
        "loss_history_tail": [round(v, 6) for v in history[-10:]],
        "elapsed_s": round(time.time() - t0, 2),
    }
    manifest = {
        "voice_id": voice_id,
        "family": "kokoro",
        "backend": "kokoro-voice-pack",
        "objective": "spsa-audio-spectrum",
        "base_model": model_name,
        "base_voice": cfg.base_voice,
        "lang_code": cfg.lang_code,
        "tenant_id": tenant_id,
        "sample_count": len(usable),
        "audio_seconds": round(sum(s.duration for s, _ in usable), 2),
        "config": asdict(cfg),
        "created_at": time.time(),
        "quality": "improved" if beat else "regressed",
        "format": f"voice.safetensors (key='voice', shape={list(stock.shape)})",
    }
    voice_path = write_voice_package(out_dir, mx.array(pack), manifest, metrics)
    return {
        "voice_path": voice_path,
        "voice_id": voice_id,
        "manifest": manifest,
        "metrics": metrics,
    }
