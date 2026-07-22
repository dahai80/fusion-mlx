"""Sequential Temperature Scaling (STS) — paper §3.2.1 post-hoc calibration.

The confidence head emits per-position logits ``z_k``; raw conditional
confidence is ``c_k = sigmoid(z_k)`` and the prefix survival probability is
the cumulative product ``a_j = prod_{k<=j} c_k``. Neural confidence estimates
are overconfident, so STS fits one temperature scalar per block position,
left to right: at position ``k`` a 1-D grid search picks the temperature
``t_k`` (``c_k^cal = sigmoid(z_k / t_k)``) that minimizes the Expected
Calibration Error of the *cumulative* product ``a_k``, with the
already-fitted temperatures of positions ``< k`` frozen. Temperature scaling
is order-preserving per position (a monotone transform of the logit), so it
rectifies absolute magnitudes without disturbing the confidence head's
rankings.

Labels come from generation traces collected at ``confidence_threshold=0``
(a nonzero threshold censors acceptance outcomes past the prune point): the
empirical label for ``a_k`` is 1 iff the first ``k`` drafted tokens of the
round were all accepted, which is exactly the reference
``accept_prefix_mask.cumprod`` semantics (refs/deepspec ``verify_draft_tokens``).

ECE / AUROC / Brier definitions mirror the reference metrics code at
refs/deepspec/deepspec/eval/dspark/confidence_head.py
(``PerPositionConfidenceMetrics``): equal-width probability bins with
count-weighted |avg_pred - avg_target| for ECE, a fine-histogram rank
statistic (ties counted half) for AUROC, mean squared error for Brier, and
probabilities clamped to [1e-8, 1 - 1e-8]. Bin counts follow the reference
evaluator defaults (20 coarse / 1000 fine).

Fitted temperatures ship inside the converted model directory's
``config.json`` under ``dspark_config.sts_temperatures``;
``dspark_metal.draft.load_draft_model`` exposes them as
``draft.sts_temperatures`` and the runtime divides the confidence logits by
them before sigmoid + threshold (``runtime.confident_prefix_length``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

EPS_PROB = 1e-8
NUM_COARSE_BINS = 20  # reference CONFIDENCE_NUM_BINS
NUM_FINE_BINS = 1000  # reference CONFIDENCE_NUM_FINE_BINS

# Default 1-D search grid: log-spaced over [0.2, 10] (both over- and
# under-confidence correctable), with 1.0 included exactly so calibration can
# never be worse than raw on the fit split.
DEFAULT_GRID = tuple(
    sorted(
        set(np.round(np.exp(np.linspace(np.log(0.2), np.log(10.0), 157)), 6)) | {1.0}
    )
)


# ---------------------------------------------------------------------------
# Reliability metrics (reference definitions)
# ---------------------------------------------------------------------------


def _clamp_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=np.float64), EPS_PROB, 1.0 - EPS_PROB)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = NUM_COARSE_BINS,
) -> float:
    """Count-weighted ECE over ``num_bins`` equal-width bins.

    Mirrors ``PerPositionConfidenceMetrics``: bin index = floor(p * bins)
    clipped to the last bin; ECE = sum_b w_b * |mean_pred_b - mean_label_b|.
    """
    probs = _clamp_probs(probs).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return float("nan")
    bin_idx = np.clip((probs * num_bins).astype(np.int64), 0, num_bins - 1)
    counts = np.bincount(bin_idx, minlength=num_bins).astype(np.float64)
    pred_sum = np.bincount(bin_idx, weights=probs, minlength=num_bins)
    label_sum = np.bincount(bin_idx, weights=labels, minlength=num_bins)
    total = counts.sum()
    denom = np.maximum(counts, 1e-12)
    bin_err = np.abs(pred_sum / denom - label_sum / denom)
    return float((bin_err * counts).sum() / total)


def auroc(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = NUM_FINE_BINS,
) -> float:
    """Histogram AUROC (reference ``_auroc_from_hist``): ties count half."""
    probs = _clamp_probs(probs).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    bin_idx = np.clip((probs * num_bins).astype(np.int64), 0, num_bins - 1)
    pos_hist = np.bincount(bin_idx, weights=labels, minlength=num_bins)
    neg_hist = np.bincount(bin_idx, weights=1.0 - labels, minlength=num_bins)
    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()
    if total_pos <= 0.0 or total_neg <= 0.0:
        return float("nan")
    cum_neg_before = np.cumsum(neg_hist) - neg_hist
    pairs = (pos_hist * cum_neg_before).sum() + 0.5 * (pos_hist * neg_hist).sum()
    return float(pairs / (total_pos * total_neg))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = _clamp_probs(probs).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


# ---------------------------------------------------------------------------
# STS fit / apply
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _as_masked_arrays(
    confidence_logits: Sequence[Sequence[float]] | np.ndarray,
    acceptance_labels: Sequence[Sequence[float]] | np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack possibly-ragged per-round rows into [N, B] arrays + a valid mask.

    Rows shorter than ``block_size`` (stop-token-truncated rounds; positions
    past an accepted EOS have no defined label, reference
    ``effective_proposal_length``) are masked out beyond their length.
    """
    num_rounds = len(confidence_logits)
    logits = np.zeros((num_rounds, block_size), dtype=np.float64)
    labels = np.zeros((num_rounds, block_size), dtype=np.float64)
    mask = np.zeros((num_rounds, block_size), dtype=bool)
    for row, (row_logits, row_labels) in enumerate(
        zip(confidence_logits, acceptance_labels)
    ):
        row_logits = np.asarray(row_logits, dtype=np.float64).reshape(-1)
        row_labels = np.asarray(row_labels, dtype=np.float64).reshape(-1)
        length = min(row_logits.size, row_labels.size, block_size)
        logits[row, :length] = row_logits[:length]
        labels[row, :length] = row_labels[:length]
        mask[row, :length] = True
    return logits, labels, mask


def fit_sts(
    confidence_logits: Sequence[Sequence[float]] | np.ndarray,
    acceptance_labels: Sequence[Sequence[float]] | np.ndarray,
    block_size: int,
    grid: Sequence[float] = DEFAULT_GRID,
    num_bins: int = NUM_COARSE_BINS,
) -> list[float]:
    """Fit one temperature per block position, left to right (paper §3.2.1).

    confidence_logits: per-round confidence-head logits, ragged rows allowed
        (row k truncated at the round's effective proposal length).
    acceptance_labels: matching prefix labels — label[k] = 1 iff the first
        k+1 drafted tokens were all accepted (already-cumulative form).
    Returns ``block_size`` temperatures; positions with no observations keep
    temperature 1.0.

    At position k the grid search minimizes
    ``ECE(a_{k-1} * sigmoid(z_k / t), label_k)`` where ``a_{k-1}`` is the
    survival product under the already-fitted (frozen) temperatures. Ties
    prefer the temperature closest to 1.0 (then the smaller one) so the fit
    is deterministic and never strays from identity without evidence.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    grid_values = [float(t) for t in grid]
    if not grid_values or any(t <= 0.0 for t in grid_values):
        raise ValueError("grid must be a non-empty sequence of positive temperatures")

    logits, labels, mask = _as_masked_arrays(
        confidence_logits, acceptance_labels, block_size
    )
    temps: list[float] = []
    survival = np.ones(logits.shape[0], dtype=np.float64)
    for pos in range(block_size):
        valid = mask[:, pos]
        if not valid.any():
            temps.append(1.0)
            continue
        pos_logits = logits[valid, pos]
        pos_labels = labels[valid, pos]
        pos_survival = survival[valid]
        best_temp = 1.0
        best_key: tuple[float, float, float] | None = None
        for temp in grid_values:
            cumulative = pos_survival * _sigmoid(pos_logits / temp)
            ece = expected_calibration_error(cumulative, pos_labels, num_bins=num_bins)
            key = (ece, abs(np.log(temp)), temp)
            if best_key is None or key < best_key:
                best_key = key
                best_temp = temp
        temps.append(float(best_temp))
        # Freeze this position's calibrated confidence into the survival
        # product for all rows (masked rows never contribute later anyway).
        survival = survival * _sigmoid(logits[:, pos] / best_temp)
    return temps


def apply_sts(logits: Any, temps: Sequence[float]) -> Any:
    """Calibrate confidence logits: ``logits[..., k] / temps[k]``.

    Accepts numpy arrays (fit/eval paths) or ``mx.array`` (runtime path);
    the last axis must have length ``len(temps)`` (or fewer, for truncated
    rows — the leading temperatures apply).
    """
    if hasattr(logits, "__module__") and type(logits).__module__.startswith("mlx"):
        import mlx.core as mx

        length = int(logits.shape[-1])
        _check_temps_length(length, len(temps))
        return logits / mx.array(list(temps[:length]), dtype=logits.dtype)
    array = np.asarray(logits, dtype=np.float64)
    length = int(array.shape[-1])
    _check_temps_length(length, len(temps))
    return array / np.asarray(list(temps[:length]), dtype=np.float64)


def _check_temps_length(logit_len: int, temps_len: int) -> None:
    if logit_len > temps_len:
        raise ValueError(
            f"confidence logits have {logit_len} positions but only "
            f"{temps_len} STS temperatures are available"
        )


# ---------------------------------------------------------------------------
# Cumulative-survival helpers shared by fit/eval and the calibrate CLI
# ---------------------------------------------------------------------------


def survival_products(
    confidence_logits: Sequence[Sequence[float]] | np.ndarray,
    block_size: int,
    temps: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative survival probabilities ``a_k`` per round -> ([N, B], mask).

    ``temps=None`` gives the raw (uncalibrated) products.
    """
    logits, _, mask = _as_masked_arrays(
        confidence_logits, [np.zeros(len(row)) for row in confidence_logits], block_size
    )
    if temps is not None:
        logits = apply_sts(logits, temps)
    probs = _sigmoid(logits)
    # Masked positions multiply as 1.0 so they never poison the prefix; they
    # are excluded from metrics via the mask anyway.
    probs = np.where(mask, probs, 1.0)
    return np.cumprod(probs, axis=1), mask


def per_position_metrics(
    confidence_logits: Sequence[Sequence[float]] | np.ndarray,
    acceptance_labels: Sequence[Sequence[float]] | np.ndarray,
    block_size: int,
    temps: Sequence[float] | None = None,
    num_bins: int = NUM_COARSE_BINS,
    num_fine_bins: int = NUM_FINE_BINS,
) -> list[dict[str, float]]:
    """Reference-style per-position rows (ECE/AUROC/Brier on cumprod a_k)."""
    _, labels, mask = _as_masked_arrays(
        confidence_logits, acceptance_labels, block_size
    )
    cumulative, _ = survival_products(confidence_logits, block_size, temps)
    rows: list[dict[str, float]] = []
    for pos in range(block_size):
        valid = mask[:, pos]
        count = int(valid.sum())
        if count == 0:
            rows.append(
                {
                    "position": pos,
                    "count": 0,
                    "ece": float("nan"),
                    "auroc": float("nan"),
                    "brier": float("nan"),
                    "pred_mean": float("nan"),
                    "target_mean": float("nan"),
                }
            )
            continue
        probs = cumulative[valid, pos]
        target = labels[valid, pos]
        rows.append(
            {
                "position": pos,
                "count": count,
                "ece": expected_calibration_error(probs, target, num_bins=num_bins),
                "auroc": auroc(probs, target, num_bins=num_fine_bins),
                "brier": brier_score(probs, target),
                "pred_mean": float(_clamp_probs(probs).mean()),
                "target_mean": float(target.mean()),
            }
        )
    return rows


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Count-weighted means across positions (reference summarize_confidence_row)."""
    total = 0.0
    sums = {"ece": 0.0, "brier": 0.0, "pred_mean": 0.0, "target_mean": 0.0}
    auc_weight = 0.0
    auc_sum = 0.0
    for row in rows:
        weight = float(row["count"])
        if weight <= 0.0:
            continue
        total += weight
        for key in sums:
            sums[key] += float(row[key]) * weight
        if not np.isnan(row["auroc"]):
            auc_weight += weight
            auc_sum += float(row["auroc"]) * weight
    if total <= 0.0:
        return {
            key: float("nan")
            for key in ("ece", "auroc", "brier", "pred_mean", "target_mean")
        }
    out = {key: value / total for key, value in sums.items()}
    out["auroc"] = auc_sum / auc_weight if auc_weight > 0.0 else float("nan")
    return out


def pooled_cumulative_metrics(
    confidence_logits: Sequence[Sequence[float]] | np.ndarray,
    acceptance_labels: Sequence[Sequence[float]] | np.ndarray,
    block_size: int,
    temps: Sequence[float] | None = None,
) -> dict[str, float]:
    """ECE/AUROC/Brier pooling every (a_k, label_k) pair across positions."""
    _, labels, mask = _as_masked_arrays(
        confidence_logits, acceptance_labels, block_size
    )
    cumulative, _ = survival_products(confidence_logits, block_size, temps)
    probs = cumulative[mask]
    target = labels[mask]
    return {
        "count": int(probs.size),
        "ece": expected_calibration_error(probs, target),
        "auroc": auroc(probs, target),
        "brier": brier_score(probs, target),
    }


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def save_sts_temperatures(
    model_dir: str | Path,
    temps: Sequence[float],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write temps into ``<model_dir>/config.json`` under
    ``dspark_config.sts_temperatures`` (+ optional ``sts_calibration`` meta).
    """
    config_path = Path(model_dir) / "config.json"
    config = json.loads(config_path.read_text())
    dspark_config = config.get("dspark_config")
    if not isinstance(dspark_config, dict):
        raise ValueError(
            f"{config_path} has no dspark_config section; is this a converted "
            "DSpark draft model directory?"
        )
    block_size = int(dspark_config.get("block_size", len(temps)))
    if len(temps) != block_size:
        raise ValueError(
            f"expected {block_size} temperatures (block_size), got {len(temps)}"
        )
    dspark_config["sts_temperatures"] = [float(t) for t in temps]
    if metadata is not None:
        dspark_config["sts_calibration"] = metadata
    config["dspark_config"] = dspark_config
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def load_sts_temperatures(model_dir: str | Path) -> list[float] | None:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    temps = (config.get("dspark_config") or {}).get("sts_temperatures")
    if temps is None:
        return None
    return [float(t) for t in temps]
