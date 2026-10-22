"""Weak and strong views for the SSL smoke path.

These are fixed tensor ops so a CPU test is deterministic. They are not
the FixMatch / RandAugment list in the method note; that list is still a
placeholder for a real training run.
"""

from __future__ import annotations

import torch
from torch import Tensor


def weak_augment(images: Tensor) -> Tensor:
    """Horizontal flip. ``images`` is ``(B, C, H, W)``."""
    _check_image_batch(images)
    return torch.flip(images, dims=(-1,))


def strong_augment(images: Tensor, noise_std: float = 0.0) -> Tensor:
    """Vertical flip, a corner cutout, and optional Gaussian noise."""
    _check_image_batch(images)
    out = torch.flip(images, dims=(-2,)).clone()
    height = max(int(out.shape[-2]) // 4, 1)
    width = max(int(out.shape[-1]) // 4, 1)
    out[..., :height, :width] = 0
    if noise_std > 0:
        out = (out + float(noise_std) * torch.randn_like(out)).clamp(0.0, 1.0)
    return out


def _check_image_batch(images: Tensor) -> None:
    if images.ndim != 4:
        raise ValueError(f"Expected NCHW images, got shape {tuple(images.shape)}")
