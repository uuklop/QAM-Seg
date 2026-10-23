"""Quality-Gated Consistency (QGC).

Reweights unsupervised BCE + Dice against a stop-grad teacher (or weak-view)
pseudo-label. Hard, soft, and the doc's drop-then-weight gate are supported.
An empty gate returns a zero that stays on the student graph and does not
divide by zero.

``lambda_u`` and ``tau`` helpers are the curriculum placeholders from the
method note. They are not tuned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from qamseg.mqe import QualityEstimate, as_prob_maps


@dataclass(frozen=True)
class GateConfig:
    """Gating placeholders. ``tau`` is overridden by the curriculum when training."""

    mode: str = "gated"
    level: str = "both"
    tau: float = 0.5
    gamma: float = 1.0
    eps: float = 1e-6
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    hard_targets: bool = True

    def __post_init__(self):
        if self.mode not in ("hard", "soft", "gated", "ungated"):
            raise ValueError(f"Unknown gate mode: {self.mode}")
        if self.level not in ("image", "pixel", "both"):
            raise ValueError(f"Unknown gate level: {self.level}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")


@dataclass
class GateResult:
    loss: Tensor
    keep_fraction: Tensor
    mean_weight: Tensor
    weights: Tensor
    qw_alpha: float = 0.0
    sample_weight: Optional[Tensor] = None


def tau_at_step(step: int, tau0: float, tau1: float, warm_steps: int) -> float:
    """Linear threshold ramp. ``τ_t = τ0 + (τ1-τ0) * min(1, t / T_warm)``."""
    if warm_steps <= 0:
        return float(tau1)
    progress = min(1.0, float(step) / float(warm_steps))
    return float(tau0) + (float(tau1) - float(tau0)) * progress


def lambda_u_at_step(step: int, total_steps: int, lambda_max: float) -> float:
    """Gaussian ramp-up. ``λ_u(t) = λ_max * exp(-5 (1 - t/T)^2)``.

    ``step == 0`` is near zero. ``step >= total_steps`` equals ``lambda_max``.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    ratio = min(1.0, max(0.0, float(step) / float(total_steps)))
    return float(lambda_max) * math.exp(-5.0 * (1.0 - ratio) ** 2)


def _score_to_weight(score: Tensor, mode: str, tau: float, gamma: float) -> Tensor:
    score = score.clamp(0.0, 1.0)
    if mode == "hard":
        return (score >= tau).to(dtype=score.dtype)
    if mode == "soft":
        return score.pow(gamma)
    if mode == "gated":
        # Drop below tau, otherwise keep the soft weight (doc step 3).
        return (score >= tau).to(dtype=score.dtype) * score.pow(gamma)
    raise ValueError(f"Unknown gate mode: {mode}")


def gate_weights(quality: QualityEstimate, config: GateConfig) -> Tensor:
    """Pixel weights in ``[0, 1]`` with shape ``(B, 1, H, W)``.

    ``gated`` + ``both`` follows the plan: reject the whole image when
    ``q < tau``, and otherwise multiply by ``q^gamma`` and the pixel map.
    The pixel map is not thresholded a second time in that mode.
    """
    pixel = as_prob_maps(quality.pixel_weight).detach()
    q = quality.q.detach().float().view(-1)
    if q.shape[0] != pixel.shape[0]:
        raise ValueError(
            f"Gate batch mismatch: q {tuple(q.shape)} vs pixels {tuple(pixel.shape)}"
        )
    q = q.to(device=pixel.device)

    if config.mode == "ungated":
        return torch.ones_like(pixel)

    use_image = config.level in ("image", "both")
    use_pixel = config.level in ("pixel", "both")

    if use_image:
        image_w = _score_to_weight(q, config.mode, config.tau, config.gamma)
        image_w = image_w.view(-1, 1, 1, 1)
    else:
        image_w = torch.ones(pixel.shape[0], 1, 1, 1, device=pixel.device, dtype=pixel.dtype)

    if use_pixel:
        if config.mode == "gated" and config.level == "both":
            pix_w = pixel.clamp(0.0, 1.0)
        else:
            pix_w = _score_to_weight(pixel, config.mode, config.tau, config.gamma)
    else:
        pix_w = torch.ones_like(pixel)

    return (image_w * pix_w).clamp(0.0, 1.0)


def quality_gated_consistency_loss(
    student_probs: Tensor,
    teacher_probs: Tensor,
    quality: QualityEstimate,
    config: Optional[GateConfig] = None,
) -> GateResult:
    """Weighted BCE + Dice. Teacher targets and gate weights are stop-grad.

    If every weight is zero the loss is ``student.sum() * 0``, which is
    finite, backpropagates as zeros, and does not divide by the weight sum.
    """
    if config is None:
        config = GateConfig()
    student = as_prob_maps(student_probs).clamp(config.eps, 1.0 - config.eps)
    teacher = as_prob_maps(teacher_probs).detach()
    if student.shape != teacher.shape:
        raise ValueError(
            f"QGC shape mismatch: student {tuple(student.shape)} vs teacher {tuple(teacher.shape)}"
        )
    if config.hard_targets:
        target = (teacher >= 0.5).to(dtype=student.dtype)
    else:
        target = teacher.clamp(0.0, 1.0).to(dtype=student.dtype)

    weights = gate_weights(quality, config).to(device=student.device, dtype=student.dtype)
    if weights.shape != student.shape:
        raise ValueError(
            f"QGC weight shape {tuple(weights.shape)} != student {tuple(student.shape)}"
        )

    keep_fraction = (weights > 0).to(dtype=student.dtype).mean()
    mean_weight = weights.mean()
    if not bool((weights > 0).any()):
        zero = student.sum() * 0.0
        return GateResult(
            loss=zero,
            keep_fraction=keep_fraction,
            mean_weight=mean_weight,
            weights=weights,
        )

    bce = F.binary_cross_entropy(student, target, reduction="none")
    weight_sum = weights.sum()
    bce_loss = (weights * bce).sum() / (weight_sum + config.eps)

    batch = student.shape[0]
    flat_w = weights.view(batch, -1)
    flat_p = student.view(batch, -1)
    flat_t = target.view(batch, -1)
    image_mass = flat_w.sum(dim=1)
    intersection = (flat_w * flat_p * flat_t).sum(dim=1)
    denom = (flat_w * flat_p).sum(dim=1) + (flat_w * flat_t).sum(dim=1)
    dice = (2.0 * intersection + config.eps) / (denom + config.eps)
    alive = image_mass > 0
    dice_loss = (image_mass[alive] * (1.0 - dice[alive])).sum() / image_mass[alive].sum().clamp_min(config.eps)

    loss = config.bce_weight * bce_loss + config.dice_weight * dice_loss
    return GateResult(
        loss=loss,
        keep_fraction=keep_fraction,
        mean_weight=mean_weight,
        weights=weights,
    )
