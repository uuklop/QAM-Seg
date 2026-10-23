"""SSL student factory.

``tiny`` is ``TinySegStudent``, the CPU convolutional fallback used by the
default smoke path. There is no separate ConvLiteUNet module. ``mamba``
constructs ``models.MambaLiteUNet`` when that import succeeds. A failed import
is reported and is fatal only when ``mamba`` was requested. The supervised
``python train.py`` path does not use this module. MambaLiteUNet is the
backbone, not a second quality model.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C_LIST = [16, 32, 48, 64, 96, 128]


class BackboneUnavailable(Exception):
    """``--ssl-backbone mamba`` was requested and the class did not construct."""

    def __init__(self, cause: BaseException):
        self.cause = cause
        super().__init__(self.format_message())

    def format_message(self) -> str:
        return (
            "QAM-Seg: --ssl-backbone mamba was requested, but models.MambaLiteUNet "
            "could not be imported or constructed.\n"
            f"Reason: {type(self.cause).__name__}: {self.cause}\n"
            "The tiny conv student was not substituted, because this backbone was explicitly requested.\n"
            "Re-run with the default (omit --ssl-backbone, or pass --ssl-backbone tiny).\n"
            "This process does not install packages and does not download datasets.\n"
            "A successful import is still not a medical training run and does not report CVPR Dice or HD95."
        )


def read_c_list(config_path: Path | None = None) -> list[int]:
    """Read ``c_list`` without importing ``configs/config_setting.py``.

    That module builds a loss and a timestamped work directory on import.
    """
    path = config_path or (ROOT / "configs" / "config_setting.py")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return list(DEFAULT_C_LIST)
    match = re.search(r"""['"]c_list['"]\s*:\s*\[([0-9,\s]+)\]""", text)
    if not match:
        return list(DEFAULT_C_LIST)
    values = [int(part) for part in match.group(1).split(",") if part.strip()]
    return values or list(DEFAULT_C_LIST)


def import_mamba_lite_unet():
    """Import the released class. Missing ``mamba_ssm`` or ``timm`` raises."""
    from models.MambaLiteUNet import MambaLiteUNet

    return MambaLiteUNet


def build_mamba_student(importer=None):
    """Construct MambaLiteUNet with the config ``c_list``.

    The forward in this tree already applies ``sigmoid``. Callers that asked
    for this backbone should let ``BackboneUnavailable`` stop the process.
    """
    load = importer or import_mamba_lite_unet
    try:
        cls = load()
        return cls(num_classes=1, input_channels=3, c_list=read_c_list())
    except BackboneUnavailable:
        raise
    except Exception as exc:
        raise BackboneUnavailable(exc) from exc
