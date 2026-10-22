"""Quality-aware SSL loop (MQE + QGC).

``python train.py`` with no flags never imports this module. ``--ssl`` and
``python -m qamseg.train_ssl`` do. The default run is a few CPU steps on
synthetic blobs with a tiny conv student, so it does not need a GPU,
``mamba_ssm``, or ISIC/HAM/PH2. That run checks the wiring. It is not an
experiment and it does not report a Dice or HD95 result.

The step itself takes any student that maps ``(B, 3, H, W)`` to a sigmoid
mask ``(B, 1, H, W)``. ``--ssl-backbone mamba`` builds
``models.MambaLiteUNet`` when that import works. The default remains
``TinySegStudent``. A prepared split is optional and only indexes local
arrays; this module still does not download medical data or report a
dataset Dice or HD95.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import copy
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from qamseg.augment import strong_augment, weak_augment
from qamseg.backbones import BackboneUnavailable, build_mamba_student
from qamseg.cli import parse_train_args, refuse_reserved_flags
from qamseg.data import build_dual_loaders, build_ssl_datasets, iter_dual_batches
from qamseg.ema import ema_update
from qamseg.metrics import binary_pixel_metrics, format_eval_log, format_ssl_log, hd95_for_log
from qamseg.mqe import apply_boundary_aware_prior, estimate_quality, gradient_magnitude
from qamseg.qgc import GateConfig, lambda_u_at_step, quality_gated_consistency_loss, tau_at_step
from utils import BceDiceLoss


class TinySegStudent(nn.Module):
    """CPU stand-in. Not MambaLiteUNet and not a source of benchmark scores."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def ssl_training_step(
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    labeled_images: torch.Tensor,
    labeled_masks: torch.Tensor,
    unlabeled_images: torch.Tensor,
    *,
    step: int,
    total_steps: int,
    criterion_sup: nn.Module,
    gate: GateConfig,
    lambda_max: float = 1.0,
    tau0: float = 0.4,
    tau1: float = 0.8,
    warm_steps: int = 1,
    ema_alpha: float = 0.99,
    ema_warmup_steps: int = 0,
    noise_std: float = 0.0,
    use_baq: bool = False,
) -> dict:
    """One labeled step plus one quality-gated unlabeled step.

    The teacher and the weak-view student forward used by MQE are
    stop-grad. The strong-view student prediction is what QGC trains.
    """
    student.train()
    pred_labeled = student(labeled_images)
    loss_sup = criterion_sup(pred_labeled, labeled_masks)

    view_weak = weak_augment(unlabeled_images)
    view_strong = strong_augment(unlabeled_images, noise_std=noise_std)
    with torch.no_grad():
        teacher_weak = teacher(view_weak)
        student_weak = student(view_weak)
    pred_strong = student(view_strong)

    quality = estimate_quality(
        teacher_weak,
        student_probs=student_weak,
        view_probs=pred_strong.detach(),
    )
    if use_baq:
        grad = gradient_magnitude(view_weak)
        quality.pixel_weight = apply_boundary_aware_prior(
            quality.pixel_weight,
            grad,
            quality.pixel_weight,
        )

    tau = tau_at_step(step, tau0, tau1, warm_steps)
    # The last optimizer step is t = T, so a run of N steps reaches lambda_max
    # on step N-1. Step 0 stays at the foot of the Gaussian ramp.
    lam = lambda_u_at_step(step, max(int(total_steps) - 1, 1), lambda_max)
    gated = quality_gated_consistency_loss(
        pred_strong,
        teacher_weak,
        quality,
        replace(gate, tau=tau),
    )
    loss = loss_sup + lam * gated.loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step >= ema_warmup_steps:
        ema_update(teacher, student, alpha=ema_alpha)

    return {
        "step": int(step),
        "loss": float(loss.detach()),
        "loss_sup": float(loss_sup.detach()),
        "loss_unsup": float(gated.loss.detach()),
        "lambda_u": float(lam),
        "gate_keep_fraction": float(gated.keep_fraction.detach()),
        "mean_gate_weight": float(gated.mean_weight.detach()),
        "q_mean": float(quality.q.detach().mean()),
        "tau": float(tau),
    }


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, criterion: nn.Module, threshold: float = 0.5) -> dict:
    """Labeled eval with the same metric names as the supervised logs."""
    model.eval()
    losses = []
    pred_maps = []
    gt_maps = []
    for images, masks in loader:
        out = model(images)
        losses.append(float(criterion(out, masks)))
        pred_maps.append(out[:, 0].detach().cpu().numpy())
        gt_maps.append(masks[:, 0].detach().cpu().numpy())
    pred = np.concatenate(pred_maps, axis=0)
    gt = np.concatenate(gt_maps, axis=0)
    metrics = binary_pixel_metrics(pred, gt, threshold=threshold)
    metrics["hd95"] = hd95_for_log([pred], [gt], threshold=threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def run_ssl_loop(
    student: nn.Module,
    labeled_loader,
    unlabeled_loader,
    val_loader,
    *,
    steps: int,
    gate: GateConfig,
    lambda_max: float,
    tau0: float,
    tau1: float,
    warm_steps: int,
    lr: float = 1e-3,
    seed: int = 0,
    use_baq: bool = False,
) -> dict:
    torch.manual_seed(seed)
    teacher = copy.deepcopy(student).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    criterion = BceDiceLoss()
    history = []
    step = 0
    while step < steps:
        for images_l, masks_l, images_u in iter_dual_batches(labeled_loader, unlabeled_loader):
            row = ssl_training_step(
                student,
                teacher,
                optimizer,
                images_l,
                masks_l,
                images_u,
                step=step,
                total_steps=steps,
                criterion_sup=criterion,
                gate=gate,
                lambda_max=lambda_max,
                tau0=tau0,
                tau1=tau1,
                warm_steps=warm_steps,
                use_baq=use_baq,
            )
            history.append(row)
            print(format_ssl_log(row))
            step += 1
            if step >= steps:
                break
    metrics = evaluate_loader(student, val_loader, criterion)
    print(format_eval_log(metrics))
    return {"history": history, "metrics": metrics}


def run_from_args(args) -> int:
    if int(args.ssl_steps) < 1:
        print("QAM-Seg: --ssl-steps must be >= 1.")
        return 2
    if int(args.ssl_batch_size) < 1:
        print("QAM-Seg: --ssl-batch-size must be >= 1.")
        return 2
    device = str(args.ssl_device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("QAM-Seg: --ssl-device cuda was requested but CUDA is not available.")
        print("The synthetic smoke path runs on CPU. Re-run with --ssl-device cpu.")
        return 2

    synthetic = not (args.ssl_unlabeled_npy or args.ssl_labeled_npy or args.ssl_mask_npy or args.ssl_split)
    print("QAM-Seg SSL path: MQE + QGC.")
    print("Implemented in code and exercised on this process only.")
    print("Not experimentally validated. No ISIC, HAM10000, or PH2 download.")
    print("No CVPR MambaLiteUNet metric is computed or claimed.")
    if args.use_learned_quality_head:
        print("Learned MQE head flag is set, but the head stays a stub (weight a3 = 0, beta = 0).")
    backbone = str(args.ssl_backbone)
    if backbone == "mamba":
        try:
            student = build_mamba_student()
        except BackboneUnavailable as exc:
            print(exc)
            return 2
        print("Student: MambaLiteUNet. Import succeeded.")
        print("This is not a CVPR reproduction. Logs from this process are not medical Dice or HD95 results.")
    else:
        student = TinySegStudent()
        print("Student: TinySegStudent.")
    if synthetic:
        print("Data: synthetic blobs.")
    elif args.ssl_split:
        print("Data: local train arrays sliced by --ssl-split. Unlabeled rows do not keep masks.")
        print("Validation and test images are not added. Eval on the labeled subset is a wiring check, not a test score.")
    else:
        print("Data: local npy arrays passed on the CLI.")

    length = max(int(args.ssl_batch_size) * 2, int(args.ssl_batch_size))
    try:
        labeled, unlabeled = build_ssl_datasets(
            length=length,
            size=int(args.ssl_image_size),
            seed=int(args.ssl_seed),
            unlabeled_npy=args.ssl_unlabeled_npy,
            labeled_npy=args.ssl_labeled_npy,
            mask_npy=args.ssl_mask_npy,
            split_json=args.ssl_split,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"QAM-Seg: {exc}")
        return 2
    labeled_loader, unlabeled_loader = build_dual_loaders(
        labeled,
        unlabeled,
        batch_size=int(args.ssl_batch_size),
        shuffle=False,
    )
    if len(labeled_loader) == 0 or len(unlabeled_loader) == 0:
        print("QAM-Seg: a loader is empty after drop_last.")
        print("Lower --ssl-batch-size, or pass a local array with at least one full batch.")
        print("This entry point does not download ISIC, HAM10000, or PH2.")
        return 2
    try:
        val_labeled, _val_unlabeled = build_ssl_datasets(
            length=length,
            size=int(args.ssl_image_size),
            seed=int(args.ssl_seed) + 17,
            unlabeled_npy="",
            labeled_npy=args.ssl_labeled_npy,
            mask_npy=args.ssl_mask_npy,
            split_json=args.ssl_split,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"QAM-Seg: {exc}")
        return 2
    val_loader, _ = build_dual_loaders(
        val_labeled,
        val_labeled,
        batch_size=int(args.ssl_batch_size),
        shuffle=False,
    )
    if device == "cuda":
        student = student.cuda()
        # Loaders yield CPU tensors. Move them inside the loop only if asked.
        # The default smoke path never takes this branch in CI without a GPU.
        labeled_loader = _DeviceLoader(labeled_loader, device)
        unlabeled_loader = _DeviceLoader(unlabeled_loader, device)
        val_loader = _DeviceLoader(val_loader, device)

    gate = GateConfig(mode=args.ssl_gate, level=args.ssl_level, gamma=float(args.ssl_gamma))
    result = run_ssl_loop(
        student,
        labeled_loader,
        unlabeled_loader,
        val_loader,
        steps=int(args.ssl_steps),
        gate=gate,
        lambda_max=float(args.ssl_lambda_max),
        tau0=float(args.ssl_tau0),
        tau1=float(args.ssl_tau1),
        warm_steps=int(args.ssl_warm_steps),
        seed=int(args.ssl_seed),
        use_baq=bool(args.ssl_baq),
    )
    if not result["history"]:
        print("QAM-Seg: SSL loop produced no steps.")
        return 1
    return 0


class _DeviceLoader:
    """Move loader batches to ``device``. Used only when CUDA was requested."""

    def __init__(self, loader, device: str):
        self.loader = loader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield _move(batch, self.device)


def _move(batch, device: str):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, (tuple, list)):
        return type(batch)(_move(item, device) for item in batch)
    return batch


def main(argv=None) -> int:
    args, unknown, reserved = parse_train_args(argv)
    if unknown:
        print("QAM-Seg: ignoring unrecognized arguments: " + " ".join(unknown))
    refuse_reserved_flags(reserved)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
