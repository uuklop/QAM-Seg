"""CPU smoke of the SSL step, EMA, and the metric names shared with engine.py."""

import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import torch
from torch import nn

from qamseg.data import build_dual_loaders, build_ssl_datasets
from qamseg.ema import ema_update
from qamseg.metrics import binary_pixel_metrics, format_eval_log, format_ssl_log, hd95_for_log
from qamseg.qgc import GateConfig
from qamseg.train_ssl import evaluate_loader, run_ssl_loop, ssl_training_step
from qamseg.train_ssl import TinySegStudent


class EmaTests(unittest.TestCase):
    def test_ema_moves_teacher_toward_student(self):
        class Toy(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(float(value)))

        teacher = Toy(0.0)
        student = Toy(1.0)
        ema_update(teacher, student, alpha=0.5)
        self.assertAlmostEqual(float(teacher.weight.detach()), 0.5, places=6)


class MetricTests(unittest.TestCase):
    def test_pixel_metrics_match_the_supervised_ratios(self):
        # One true positive, one false negative, one false positive, one true negative.
        pred = [1, 0, 1, 0]
        gt = [1, 1, 0, 0]
        scores = binary_pixel_metrics(pred, gt, threshold=0.5)
        self.assertAlmostEqual(scores["accuracy"], 0.5)
        self.assertAlmostEqual(scores["sensitivity"], 0.5)
        self.assertAlmostEqual(scores["specificity"], 0.5)
        self.assertAlmostEqual(scores["f1_or_dsc"], 0.5)
        self.assertAlmostEqual(scores["miou"], 1.0 / 3.0)

    def test_hd95_helper_on_identical_masks(self):
        mask = torch.zeros(1, 8, 8)
        mask[:, 2:6, 2:6] = 1
        array = mask.numpy()
        text = hd95_for_log([array], [array], threshold=0.5)
        self.assertTrue(text.startswith("0.0000") or text.startswith("0"))
        log = format_eval_log(
            {
                "loss": 0.1,
                "miou": 1.0,
                "f1_or_dsc": 1.0,
                "accuracy": 1.0,
                "specificity": 1.0,
                "sensitivity": 1.0,
                "hd95": text,
            }
        )
        self.assertIn("hd95:", log)
        self.assertIn("f1_or_dsc:", log)


class SmokeTests(unittest.TestCase):
    def test_two_cpu_steps_log_unsup_loss_and_keep_fraction(self):
        labeled, unlabeled = build_ssl_datasets(length=4, size=16, seed=1)
        labeled_loader, unlabeled_loader = build_dual_loaders(labeled, unlabeled, batch_size=2)
        val_labeled, _ = build_ssl_datasets(length=4, size=16, seed=2)
        val_loader, _ = build_dual_loaders(val_labeled, val_labeled, batch_size=2)
        result = run_ssl_loop(
            TinySegStudent(),
            labeled_loader,
            unlabeled_loader,
            val_loader,
            steps=2,
            gate=GateConfig(mode="soft", level="both"),
            lambda_max=1.0,
            tau0=0.4,
            tau1=0.8,
            warm_steps=1,
            seed=1,
        )
        self.assertEqual(len(result["history"]), 2)
        for row in result["history"]:
            self.assertTrue(torch.isfinite(torch.tensor(row["loss"])))
            self.assertTrue(torch.isfinite(torch.tensor(row["loss_unsup"])))
            self.assertGreaterEqual(row["gate_keep_fraction"], 0.0)
            self.assertLessEqual(row["gate_keep_fraction"], 1.0)
            text = format_ssl_log(row)
            self.assertIn("unsup:", text)
            self.assertIn("gate_keep_fraction:", text)
        metrics = result["metrics"]
        for key in ("loss", "miou", "f1_or_dsc", "accuracy", "specificity", "sensitivity", "hd95"):
            self.assertIn(key, metrics)
        self.assertIn("hd95:", format_eval_log(metrics))

    def test_single_step_backward_is_finite(self):
        labeled, unlabeled = build_ssl_datasets(length=2, size=16, seed=3)
        labeled_loader, unlabeled_loader = build_dual_loaders(labeled, unlabeled, batch_size=2)
        images_l, masks_l, images_u = next(iter(iter_batches(labeled_loader, unlabeled_loader)))
        student = TinySegStudent()
        teacher = TinySegStudent()
        for param in teacher.parameters():
            param.requires_grad_(False)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        from utils import BceDiceLoss

        row = ssl_training_step(
            student,
            teacher,
            optimizer,
            images_l,
            masks_l,
            images_u,
            step=0,
            total_steps=4,
            criterion_sup=BceDiceLoss(),
            gate=GateConfig(mode="gated", level="both", tau=0.0),
        )
        self.assertTrue(torch.isfinite(torch.tensor(row["loss_unsup"])))
        self.assertGreater(row["lambda_u"], 0.0)
        self.assertLess(row["lambda_u"], 0.1)

    def test_evaluate_loader_runs_on_cpu(self):
        labeled, _ = build_ssl_datasets(length=2, size=16, seed=4)
        loader, _ = build_dual_loaders(labeled, labeled, batch_size=2)
        from utils import BceDiceLoss

        metrics = evaluate_loader(TinySegStudent(), loader, BceDiceLoss())
        self.assertIn("hd95", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["f1_or_dsc"])))


def iter_batches(labeled_loader, unlabeled_loader):
    from qamseg.data import iter_dual_batches

    return iter_dual_batches(labeled_loader, unlabeled_loader)


class SupervisedFilesUntouchedTests(unittest.TestCase):
    def test_engine_still_logs_the_existing_metrics(self):
        with open("engine.py", encoding="utf-8") as handle:
            text = handle.read()
        for key in ("f1_or_dsc", "miou", "hd95", "sensitivity", "specificity"):
            self.assertIn(key, text)

    def test_train_py_keeps_the_supervised_entry(self):
        with open("train.py", encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("isic_loader", text)
        self.assertIn("train_one_epoch", text)
        self.assertIn("if _args.ssl:", text)
        self.assertIn("refuse_reserved_flags(_reserved)", text)


if __name__ == "__main__":
    unittest.main()
