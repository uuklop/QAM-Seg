"""MQE-guided reweighting of the unlabeled consistency loss.

QG-SSL's PL-QW multiplies each pseudolabel loss by a frozen external
quality score ``w_j = g_φ(x, ŷ)``. This module is that weighting rule
with the score taken from the existing MQE estimate. It does not build
a quality network.

When ``mode`` is ``off`` and no UA-MT / PL baseline is requested, the
call delegates to ``quality_gated_consistency_loss`` so the previous
QGC objective is unchanged.

When ``mode`` is ``sample`` or ``pixel``, a ramp ``α(t)`` moves the
weight from uniform to the detached MQE score:

    w(t) = (1 - α(t)) + α(t) * q

``α`` is the same Gaussian used by ``λ_u`` unless ``ramp`` is
``linear`` or ``none``. The outer coefficient stays ``λ_u(t) * λ_qw``.
``λ_qw`` defaults to 1 so an ablation compares weights rather than a
second scale. QG-SSL published ``λ_qw = 0.25``; pass that explicitly.

Composition with QGC, when reweighting is on:

- ``hard`` / ``gated`` contribute a binary keep (``q >= tau``), not a
  second factor of ``q``. The continuous score is this reweight.
- The pixel map (including an optional BAQ edit applied by the caller)
  stays a spatial weight for ``sample`` mode.
- ``pixel`` mode owns that spatial map, so it is not multiplied twice.
- ``soft`` still applies ``q^gamma`` and will stack with this reweight.
  The MedIA recipes do not combine ``soft`` with reweighting.

UA-MT and confidence-threshold PL are baseline weight masks. They do
not read MQE. UA-MT here is teacher predictive entropy, not Monte Carlo
dropout inside MambaLiteUNet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from qamseg.mqe import QualityEstimate, as_prob_maps, normalized_entropy
from qamseg.qgc import GateConfig, GateResult, gate_weights, quality_gated_consistency_loss


@dataclass(frozen=True)
class ReweightConfig:
    """Placeholders. Not a tuned MedIA schedule."""

    mode: str = "off"
    lambda_qw: float = 1.0
    ramp: str = "gaussian"
    uncertainty: str = "off"
    uncertainty_tau: float = 0.75
    pl_threshold: float = 0.0

    def __post_init__(self):
        if self.mode not in ("off", "sample", "pixel"):
            raise ValueError(f"Unknown quality reweight mode: {self.mode}")
        if self.ramp not in ("gaussian", "linear", "none"):
            raise ValueError(f"Unknown quality-reweight ramp: {self.ramp}")
        if self.uncertainty not in ("off", "entropy"):
            raise ValueError(f"Unknown uncertainty mode: {self.uncertainty}")
        if self.lambda_qw < 0:
            raise ValueError(f"lambda_qw must be non-negative, got {self.lambda_qw}")
        if not 0.0 <= float(self.uncertainty_tau) <= 1.0:
            raise ValueError(
                f"uncertainty_tau must lie in [0, 1], got {self.uncertainty_tau}"
            )
        if not 0.0 <= float(self.pl_threshold) <= 1.0:
            raise ValueError(f"pl_threshold must lie in [0, 1], got {self.pl_threshold}")


def qw_alpha(step: int, total_steps: int, ramp: str) -> float:
    """Ramp in ``[0, 1]``. ``step >= total_steps`` is 1 for gaussian and linear.

    Gaussian matches ``lambda_u_at_step`` without the ``lambda_max`` factor:
    ``exp(-5 (1 - t/T)^2)``. The training loop passes ``T = max(steps - 1, 1)``
    so the last optimizer step reaches 1.
    """
    if ramp == "none":
        return 1.0
    if ramp not in ("gaussian", "linear"):
        raise ValueError(f"Unknown quality-reweight ramp: {ramp}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    ratio = min(1.0, max(0.0, float(step) / float(total_steps)))
    if ramp == "linear":
        return ratio
    return math.exp(-5.0 * (1.0 - ratio) ** 2)


def _legacy_qgc(config: ReweightConfig, gate: GateConfig) -> bool:
    return (
        config.mode == "off"
        and config.uncertainty == "off"
        and config.pl_threshold <= 0.0
        and gate.mode in ("hard", "soft", "gated")
    )


def _gate_factors(
    quality: QualityEstimate,
    gate: GateConfig,
    reweight_mode: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Image keep ``(B,1,1,1)``, spatial weights, ``q`` ``(B,)``, pixel map.

    Continuous ``q`` is not folded into the image keep for ``hard`` / ``gated``.
    """
    pixel = as_prob_maps(quality.pixel_weight).detach().clamp(0.0, 1.0)
    q = quality.q.detach().float().flatten()
    if q.numel() != pixel.shape[0]:
        raise ValueError(
            f"Reweight batch mismatch: q {tuple(q.shape)} vs pixels {tuple(pixel.shape)}"
        )
    q = q.to(device=pixel.device, dtype=pixel.dtype).clamp(0.0, 1.0)
    ones_i = torch.ones(pixel.shape[0], 1, 1, 1, device=pixel.device, dtype=pixel.dtype)
    ones_p = torch.ones_like(pixel)
    if gate.mode == "ungated":
        return ones_i, ones_p, q, pixel

    use_image = gate.level in ("image", "both")
    use_pixel = gate.level in ("pixel", "both")
    if gate.mode == "soft":
        image = q.view(-1, 1, 1, 1).pow(gate.gamma) if use_image else ones_i
        spatial = pixel.pow(gate.gamma) if use_pixel else ones_p
        return image, spatial, q, pixel
    if gate.mode not in ("hard", "gated"):
        raise ValueError(f"Unknown gate mode: {gate.mode}")

    image = (q >= gate.tau).to(dtype=pixel.dtype).view(-1, 1, 1, 1) if use_image else ones_i
    if not use_pixel:
        spatial = ones_p
    elif reweight_mode == "pixel":
        spatial = (pixel >= gate.tau).to(dtype=pixel.dtype) if gate.mode == "hard" else ones_p
    elif gate.mode == "hard":
        spatial = (pixel >= gate.tau).to(dtype=pixel.dtype)
    else:
        spatial = pixel
    return image, spatial, q, pixel


def _apply_uncertainty(weights: Tensor, teacher: Tensor, config: ReweightConfig) -> Tensor:
    if config.uncertainty == "off":
        return weights
    entropy = normalized_entropy(teacher).detach().clamp(0.0, 1.0)
    factor = (1.0 - entropy).clamp(0.0, 1.0)
    if float(config.uncertainty_tau) < 1.0:
        factor = factor * (entropy <= float(config.uncertainty_tau)).to(dtype=factor.dtype)
    return weights * factor.to(dtype=weights.dtype)


def _apply_pl(weights: Tensor, teacher: Tensor, config: ReweightConfig) -> Tensor:
    if config.pl_threshold <= 0.0:
        return weights
    confidence = torch.maximum(teacher, 1.0 - teacher)
    keep = (confidence >= float(config.pl_threshold)).to(dtype=weights.dtype)
    return weights * keep


def _prepare_pair(student_probs: Tensor, teacher_probs: Tensor, gate: GateConfig):
    student = as_prob_maps(student_probs).clamp(gate.eps, 1.0 - gate.eps)
    teacher = as_prob_maps(teacher_probs).detach()
    if student.shape != teacher.shape:
        raise ValueError(
            f"Reweight shape mismatch: student {tuple(student.shape)} vs teacher {tuple(teacher.shape)}"
        )
    if gate.hard_targets:
        target = (teacher >= 0.5).to(dtype=student.dtype)
    else:
        target = teacher.clamp(0.0, 1.0).to(dtype=student.dtype)
    return student, teacher, target


def _zero_result(student: Tensor, weights: Tensor, alpha: float) -> GateResult:
    zero = student.sum() * 0.0
    keep = (weights > 0).to(dtype=student.dtype).mean()
    return GateResult(
        loss=zero,
        keep_fraction=keep,
        mean_weight=weights.mean(),
        weights=weights,
        qw_alpha=float(alpha),
    )


def _finish(
    student: Tensor,
    target: Tensor,
    weights: Tensor,
    gate: GateConfig,
    *,
    alpha: float,
    lambda_qw: float,
    reduction: str,
    sample_weight: Optional[Tensor] = None,
) -> GateResult:
    weights = weights.to(device=student.device, dtype=student.dtype)
    if weights.shape != student.shape:
        raise ValueError(
            f"Reweight weight shape {tuple(weights.shape)} != student {tuple(student.shape)}"
        )
    if not bool((weights > 0).any()) or (
        sample_weight is not None and not bool((sample_weight > 0).any())
    ):
        return _zero_result(student, weights, alpha)

    bce_map = F.binary_cross_entropy(student, target, reduction="none")
    batch = student.shape[0]
    flat_w = weights.view(batch, -1)
    flat_bce = bce_map.view(batch, -1)
    flat_p = student.view(batch, -1)
    flat_t = target.view(batch, -1)
    mass = flat_w.sum(dim=1)
    alive = mass > 0
    intersection = (flat_w * flat_p * flat_t).sum(dim=1)
    denom = (flat_w * flat_p).sum(dim=1) + (flat_w * flat_t).sum(dim=1)
    dice = (2.0 * intersection + gate.eps) / (denom + gate.eps)
    keep = (weights > 0).to(dtype=student.dtype).mean()
    mean_weight = weights.mean()

    if reduction == "normalized":
        bce_loss = (weights * bce_map).sum() / (weights.sum() + gate.eps)
        dice_loss = (mass[alive] * (1.0 - dice[alive])).sum() / mass[alive].sum().clamp_min(gate.eps)
        loss = gate.bce_weight * bce_loss + gate.dice_weight * dice_loss
    elif reduction == "sample":
        if sample_weight is None:
            raise ValueError("sample reduction requires sample_weight")
        sample_weight = sample_weight.to(device=student.device, dtype=student.dtype).view(-1)
        if sample_weight.numel() != batch:
            raise ValueError(
                f"sample_weight length {sample_weight.numel()} != batch {batch}"
            )
        bce_i = torch.zeros(batch, device=student.device, dtype=student.dtype)
        dice_i = torch.zeros(batch, device=student.device, dtype=student.dtype)
        bce_i[alive] = (flat_w[alive] * flat_bce[alive]).sum(dim=1) / mass[alive].clamp_min(gate.eps)
        dice_i[alive] = 1.0 - dice[alive]
        per = gate.bce_weight * bce_i + gate.dice_weight * dice_i
        loss = (sample_weight * per).mean() * float(lambda_qw)
        mean_weight = (weights * sample_weight.view(-1, 1, 1, 1)).mean()
    elif reduction == "pixel":
        bce_loss = (weights * bce_map).mean()
        mean_w = flat_w.mean(dim=1)
        dice_i = torch.zeros(batch, device=student.device, dtype=student.dtype)
        dice_i[alive] = 1.0 - dice[alive]
        dice_loss = (mean_w * dice_i).mean()
        loss = (gate.bce_weight * bce_loss + gate.dice_weight * dice_loss) * float(lambda_qw)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return GateResult(
        loss=loss,
        keep_fraction=keep,
        mean_weight=mean_weight,
        weights=weights,
        qw_alpha=float(alpha),
        sample_weight=None if sample_weight is None else sample_weight.detach(),
    )


def unlabeled_consistency_loss(
    student_probs: Tensor,
    teacher_probs: Tensor,
    quality: QualityEstimate,
    gate: Optional[GateConfig] = None,
    config: Optional[ReweightConfig] = None,
    *,
    step: int = 0,
    total_steps: int = 1,
) -> GateResult:
    """Unlabeled BCE + Dice with optional MQE reweight, entropy, or PL mask.

    Teacher targets and every quality weight are stop-grad. An all-zero
    weight returns ``student.sum() * 0``.
    """
    if gate is None:
        gate = GateConfig()
    if config is None:
        config = ReweightConfig()
    if _legacy_qgc(config, gate):
        result = quality_gated_consistency_loss(student_probs, teacher_probs, quality, gate)
        result.qw_alpha = 0.0
        return result

    student, teacher, target = _prepare_pair(student_probs, teacher_probs, gate)
    if config.mode == "off" and gate.mode in ("hard", "soft", "gated"):
        weights = gate_weights(quality, gate).to(device=student.device, dtype=student.dtype)
        q = quality.q.detach().float().flatten().to(device=student.device)
        pixel = as_prob_maps(quality.pixel_weight).detach()
    else:
        image, spatial, q, pixel = _gate_factors(quality, gate, config.mode)
        weights = (image * spatial).to(device=student.device, dtype=student.dtype)

    weights = _apply_uncertainty(weights, teacher, config)
    weights = _apply_pl(weights, teacher, config)

    if config.mode == "off":
        return _finish(
            student,
            target,
            weights,
            gate,
            alpha=0.0,
            lambda_qw=1.0,
            reduction="normalized",
        )

    alpha = qw_alpha(step, total_steps, config.ramp)
    if config.mode == "sample":
        sample_w = (1.0 - alpha) + alpha * q.to(dtype=student.dtype)
        return _finish(
            student,
            target,
            weights,
            gate,
            alpha=alpha,
            lambda_qw=float(config.lambda_qw),
            reduction="sample",
            sample_weight=sample_w,
        )

    mix = (1.0 - alpha) + alpha * pixel.to(device=student.device, dtype=student.dtype)
    return _finish(
        student,
        target,
        weights * mix,
        gate,
        alpha=alpha,
        lambda_qw=float(config.lambda_qw),
        reduction="pixel",
    )
