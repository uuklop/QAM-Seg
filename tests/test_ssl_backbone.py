"""Optional MambaLiteUNet hook. The real class is not required."""

import contextlib
import io
import subprocess
import sys
import unittest
from unittest import mock

import torch
from torch import nn

from qamseg.backbones import BackboneUnavailable, build_mamba_student, read_c_list
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


class _CpuDeviceLoader:
    """Stand-in for ``_DeviceLoader`` so a mocked CUDA device stays on CPU."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        yield from self.loader


@contextlib.contextmanager
def _cuda_reported_without_a_gpu():
    """Treat CUDA as available and keep weights and batches on CPU."""

    def _stay_on_cpu(self, *args, **kwargs):
        return self

    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.nn.Module.cuda", _stay_on_cpu
    ), mock.patch("qamseg.train_ssl._DeviceLoader", _CpuDeviceLoader):
        yield


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
        with mock.patch("qamseg.backbones.import_mamba_lite_unet") as importer:
            code = run_from_args(args)
        self.assertEqual(code, 0)
        importer.assert_not_called()

    def test_requested_mamba_uses_the_imported_class(self):
        _FakeMamba.built = 0
        args, _unknown, _reserved = parse_train_args(
            [
                "--ssl",
                "--ssl-backbone",
                "mamba",
                "--ssl-device",
                "cuda",
                "--ssl-steps",
                "1",
                "--ssl-image-size",
                "16",
                "--ssl-batch-size",
                "2",
            ]
        )
        buf = io.StringIO()
        with _cuda_reported_without_a_gpu(), mock.patch(
            "qamseg.backbones.import_mamba_lite_unet", return_value=_FakeMamba
        ), contextlib.redirect_stdout(buf):
            code = run_from_args(args)
        text = buf.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("Student: MambaLiteUNet", text)
        self.assertNotIn("Student: TinySegStudent", text)
        self.assertGreaterEqual(_FakeMamba.built, 1)
        student = build_mamba_student(importer=lambda: _FakeMamba)
        self.assertIsInstance(student, _FakeMamba)
        self.assertEqual(student.c_list, [16, 32, 48, 64, 96, 128])
        self.assertEqual(tuple(student(torch.zeros(1, 3, 8, 8)).shape), (1, 1, 8, 8))

    def test_requested_mamba_import_failure_exits_nonzero(self):
        def explode():
            raise ImportError("No module named 'mamba_ssm'")

        args, _unknown, _reserved = parse_train_args(
            ["--ssl", "--ssl-backbone", "mamba", "--ssl-device", "cuda", "--ssl-steps", "1"]
        )
        buf = io.StringIO()
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "qamseg.backbones.import_mamba_lite_unet", side_effect=explode
        ), contextlib.redirect_stdout(buf):
            code = run_from_args(args)
        text = buf.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("explicitly requested", text)
        self.assertNotIn("Student: TinySegStudent", text)
        with self.assertRaises(BackboneUnavailable) as ctx:
            build_mamba_student(importer=explode)
        self.assertIn("explicitly requested", str(ctx.exception))
        self.assertIn("mamba_ssm", str(ctx.exception))

    def test_mamba_on_cpu_exits_before_forward(self):
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
        self.assertEqual(args.ssl_device, "cpu")
        buf = io.StringIO()
        with mock.patch("qamseg.backbones.import_mamba_lite_unet") as importer, contextlib.redirect_stdout(
            buf
        ):
            code = run_from_args(args)
        text = buf.getvalue()
        self.assertEqual(code, 2)
        importer.assert_not_called()
        self.assertIn("mamba_ssm selective scan is CUDA-only", text)
        self.assertIn("Re-run with --ssl-device cuda.", text)
        self.assertIn("explicitly requested", text)
        self.assertNotIn("Student: TinySegStudent", text)
        self.assertNotIn("Student: MambaLiteUNet", text)
        self.assertNotIn("gate_keep_fraction:", text)
        self.assertNotIn("Re-run with --ssl-device cpu.", text)

    def test_mamba_cuda_requested_but_unavailable_exits(self):
        args, _unknown, _reserved = parse_train_args(
            ["--ssl", "--ssl-backbone", "mamba", "--ssl-device", "cuda", "--ssl-steps", "1"]
        )
        buf = io.StringIO()
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "qamseg.backbones.import_mamba_lite_unet"
        ) as importer, contextlib.redirect_stdout(buf):
            code = run_from_args(args)
        text = buf.getvalue()
        self.assertEqual(code, 2)
        importer.assert_not_called()
        self.assertIn("mamba_ssm selective scan is CUDA-only", text)
        self.assertIn("Re-run with --ssl-device cuda.", text)
        self.assertIn("--ssl-device cuda was requested but CUDA is not available.", text)
        self.assertNotIn("Student: TinySegStudent", text)
        self.assertNotIn("Re-run with --ssl-device cpu.", text)

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
        self.assertEqual(completed.returncode, 2, blob)
        self.assertNotIn("Student: TinySegStudent", completed.stdout)
        self.assertNotIn("Student: MambaLiteUNet", completed.stdout)
        self.assertIn("mamba_ssm selective scan is CUDA-only", blob)
        self.assertIn("Re-run with --ssl-device cuda.", blob)
        self.assertIn("explicitly requested", blob)
        self.assertNotIn("gate_keep_fraction:", completed.stdout)
        self.assertNotIn("Re-run with --ssl-device cpu.", blob)

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


if __name__ == "__main__":
    unittest.main()
