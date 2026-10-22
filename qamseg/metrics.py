"""CPU copies of the supervised log metrics for the SSL smoke path.

Ratios match ``engine.py`` (pixel-pooled accuracy, sensitivity,
specificity, Dice / ``f1_or_dsc``, mIoU). Empty-class batches return 0
instead of indexing a 1×1 confusion matrix. HD95 is the same helper the
supervised val/test logs already call (``utils.summarize_hd95``). Values
on synthetic blobs are wiring checks, not dataset scores.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def binary_pixel_metrics(pred, gt, threshold: float = 0.5, gt_threshold: float = 0.5) -> dict:
    y_pre = np.asarray(pred).reshape(-1) >= threshold
    y_true = np.asarray(gt).reshape(-1) >= gt_threshold
    tp = int(np.count_nonzero(y_pre & y_true))
    tn = int(np.count_nonzero(~y_pre & ~y_true))
    fp = int(np.count_nonzero(y_pre & ~y_true))
    fn = int(np.count_nonzero(~y_pre & y_true))
    total = tp + tn + fp + fn
    accuracy = (tn + tp) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    denom_f1 = 2 * tp + fp + fn
    f1_or_dsc = (2 * tp) / denom_f1 if denom_f1 else 0.0
    denom_iou = tp + fp + fn
    miou = tp / denom_iou if denom_iou else 0.0
    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_or_dsc": float(f1_or_dsc),
        "miou": float(miou),
    }


def hd95_for_log(preds: Sequence, gts: Sequence, threshold: float = 0.5) -> str:
    """Same text ``engine._hd95_for_log`` would append, without CUDA."""
    try:
        from utils import format_hd95, summarize_hd95
    except Exception as exc:
        return f"n/a ({type(exc).__name__}: {exc})"
    try:
        return format_hd95(summarize_hd95(preds, gts, threshold=threshold))
    except Exception as exc:
        return f"n/a ({type(exc).__name__}: {exc})"


def format_ssl_log(row: dict) -> str:
    return (
        f"ssl: step {row['step']}, loss: {row['loss']:.4f}, "
        f"sup: {row['loss_sup']:.4f}, unsup: {row['loss_unsup']:.4f}, "
        f"lambda_u: {row['lambda_u']:.4f}, "
        f"gate_keep_fraction: {row['gate_keep_fraction']:.4f}, "
        f"q_mean: {row['q_mean']:.4f}, tau: {row['tau']:.4f}"
    )


def format_eval_log(metrics: dict) -> str:
    return (
        f"ssl-eval: loss: {metrics['loss']:.4f}, miou: {metrics['miou']}, "
        f"f1_or_dsc: {metrics['f1_or_dsc']}, accuracy: {metrics['accuracy']}, "
        f"specificity: {metrics['specificity']}, sensitivity: {metrics['sensitivity']}, "
        f"hd95: {metrics['hd95']}"
    )
