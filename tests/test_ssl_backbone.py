"""Optional MambaLiteUNet hook and the ConvLiteUNet stand-in."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from models.ConvLiteUNet import ConvLiteUNet
from qamseg.backbones import (
    BackboneUnavailable,
    ConvLiteUnavailable,
    build_conv_student,
    build_mamba_student,
    read_c_list,
)
from qamseg.cli import parse_train_args
from qamseg.train_ssl import run_from_args


class _FakeMamba(nn.Module):
    built = 0

    def __init__(self, num_classes=1, input_channels=3, c_list=None):
        super().__init__()
        type(self).built += 1
        self.c_list = list(c_list or [])
        self.net = nn.Sequential(nn.Conv2d(input_channels, num_classes, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class BackboneTests(unittest.TestCase):
    def test_default_parser_stays_on_tiny(self):
        args, _unknown, reserved = parse_train_args(["--ssl"])
        self.assertEqual(args.ssl_backbone, "tiny")
        self.assertEqual(args.ssl_split, "")
        self.assertEqual(reserved, [])

    def test_c_list_matches_the_supervised_config(self):
        self.assertEqual(read_c_list(), [16, 32, 48, 64, 96, 128])

    def test_default_smoke_does_not_import_mamba(self):
        args, _unknown, _reserved = parse_train_args(
            ["--ssl", "--ssl-steps", "1", "--ssl-image-size", "16", "--ssl-batch-size", "2"]
        )
        with mock.patch("qamseg.backbones.import_mamba_lite_unet") as importer, mock.patch(
            "qamseg.backbones.import_conv_lite_unet"
        ) as conv_importer:
            code = run_from_args(args)
        self.assertEqual(code, 0)
        importer.assert_not_called()
        conv_importer.assert_not_called()

    def test_requested_mamba_uses_the_imported_class(self):
        _FakeMamba.built = 0
        args, _unknown, _reserved = parse_train_args(
            [
                "--ssl",
                "--ssl-backbone",
                "mamba",
                "--ssl-steps",
                "1",
                "--ssl-image-size",
                "16",
                "--ssl-batch-size",
                "2",
            ]
        )
        with mock.patch("qamseg.backbones.import_mamba_lite_unet", return_value=_FakeMamba):
            code = run_from_args(args)
        self.assertEqual(code, 0)
        self.assertGreaterEqual(_FakeMamba.built, 1)
        student = build_mamba_student(importer=lambda: _FakeMamba)
        self.assertIsInstance(student, _FakeMamba)
        self.assertEqual(student.c_list, [16, 32, 48, 64, 96, 128])
        self.assertEqual(tuple(student(torch.zeros(1, 3, 8, 8)).shape), (1, 1, 8, 8))

    def test_requested_mamba_import_failure_exits_nonzero(self):
        def explode():
            raise ImportError("No module named 'mamba_ssm'")

        args, _unknown, _reserved = parse_train_args(
            ["--ssl", "--ssl-backbone", "mamba", "--ssl-steps", "1"]
        )
        with mock.patch("qamseg.backbones.import_mamba_lite_unet", side_effect=explode):
            code = run_from_args(args)
        self.assertEqual(code, 2)
        with self.assertRaises(BackboneUnavailable) as ctx:
            build_mamba_student(importer=explode)
        self.assertIn("explicitly requested", str(ctx.exception))
        self.assertIn("mamba_ssm", str(ctx.exception))

    def test_train_py_mamba_flag_does_not_fall_back_to_tiny(self):
        completed = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--ssl",
                "--ssl-backbone",
                "mamba",
                "--ssl-steps",
                "1",
                "--ssl-image-size",
                "16",
                "--ssl-batch-size",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        blob = completed.stdout + completed.stderr
        self.assertNotIn("Student: TinySegStudent", completed.stdout)
        if completed.returncode == 0:
            self.assertIn("Student: MambaLiteUNet", completed.stdout)
        else:
            self.assertEqual(completed.returncode, 2, blob)
            self.assertIn("explicitly requested", blob)
            self.assertNotIn("gate_keep_fraction:", completed.stdout)

    def test_tiny_student_is_still_the_default_process(self):
        completed = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--ssl",
                "--ssl-steps",
                "1",
                "--ssl-image-size",
                "16",
                "--ssl-batch-size",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Student: TinySegStudent", completed.stdout)

    def test_conv_student_constructs_and_forwards(self):
        student = build_conv_student()
        self.assertIsInstance(student, ConvLiteUNet)
        self.assertEqual(student.c_list, [16, 32, 48, 64, 96, 128])
        y = student(torch.zeros(1, 3, 32, 32))
        self.assertEqual(tuple(y.shape), (1, 1, 32, 32))
        self.assertTrue(torch.all(y >= 0))
        self.assertTrue(torch.all(y <= 1))

    def test_conv_smoke_does_not_import_mamba_or_label_itself_mamba(self):
        args, _unknown, _reserved = parse_train_args(
            [
                "--ssl",
                "--ssl-backbone",
                "conv",
                "--ssl-steps",
                "1",
                "--ssl-image-size",
                "32",
                "--ssl-batch-size",
                "1",
            ]
        )
        self.assertEqual(args.ssl_backbone, "conv")
        with tempfile.TemporaryDirectory() as tmp:
            args.ssl_out_dir = tmp
            buffer = io.StringIO()
            with mock.patch("qamseg.backbones.import_mamba_lite_unet") as importer, redirect_stdout(buffer):
                code = run_from_args(args)
            text = buffer.getvalue()
            self.assertEqual(code, 0, text)
            importer.assert_not_called()
            self.assertIn("Student: ConvLiteUNet.", text)
            self.assertNotIn("Student: MambaLiteUNet", text)
            self.assertNotIn("Student: TinySegStudent", text)
            self.assertIn("not MambaLiteUNet", text)
            payload = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["backbone"], "conv")
        self.assertEqual(payload["student"], "ConvLiteUNet")
        self.assertNotEqual(payload["backbone"], "mamba")
        self.assertFalse(payload["experimentally_validated"])
        self.assertEqual(payload["qar"], "skipped")
        self.assertEqual(payload["data"], "synthetic")

    def test_conv_import_failure_does_not_fall_back(self):
        def explode():
            raise ImportError("ConvLite missing")

        args, _unknown, _reserved = parse_train_args(
            ["--ssl", "--ssl-backbone", "conv", "--ssl-steps", "1", "--ssl-image-size", "32"]
        )
        with mock.patch("qamseg.backbones.import_conv_lite_unet", side_effect=explode):
            code = run_from_args(args)
        self.assertEqual(code, 2)
        with self.assertRaises(ConvLiteUnavailable) as ctx:
            build_conv_student(importer=explode)
        message = str(ctx.exception)
        self.assertIn("explicitly requested", message)
        self.assertIn("not Mamba", message)

    def test_conv_synthetic_below_32_exits_before_training(self):
        args, _unknown, _reserved = parse_train_args(
            [
                "--ssl",
                "--ssl-backbone",
                "conv",
                "--ssl-steps",
                "1",
                "--ssl-image-size",
                "16",
                "--ssl-batch-size",
                "1",
            ]
        )
        with mock.patch("qamseg.backbones.import_conv_lite_unet") as importer:
            code = run_from_args(args)
        self.assertEqual(code, 2)
        importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
