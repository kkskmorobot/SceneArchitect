"""Read-only environment diagnostics. This module never downloads model weights."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

OPTIONAL_PACKAGES = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "cv2",
    "gradio",
    "open_clip",
)


def _package_version(import_name: str) -> str | None:
    distribution_names = {"cv2": "opencv-python-headless", "open_clip": "open-clip-torch"}
    try:
        return metadata.version(distribution_names.get(import_name, import_name))
    except metadata.PackageNotFoundError:
        return None


def collect_environment(config_path: str | Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "config_exists": Path(config_path).expanduser().is_file() if config_path else None,
        "packages": {},
        "torch": {},
        "warnings": [],
    }
    for package in OPTIONAL_PACKAGES:
        installed = importlib.util.find_spec(package) is not None
        report["packages"][package] = {
            "installed": installed,
            "version": _package_version(package) if installed else None,
        }

    if not report["packages"]["torch"]["installed"]:
        report["warnings"].append(
            "PyTorch is not installed. Install a Blackwell-compatible CUDA wheel before AI stages."
        )
        return report

    import torch

    torch_report = report["torch"]
    torch_report["version"] = torch.__version__
    torch_report["cuda_available"] = torch.cuda.is_available()
    torch_report["compiled_cuda"] = torch.version.cuda
    try:
        if Version(torch.__version__.split("+")[0]) < Version("2.7"):
            report["warnings"].append(
                "PyTorch <2.7 predates official Blackwell wheel support; upgrade for RTX 5080."
            )
    except InvalidVersion:
        report["warnings"].append(f"Could not parse PyTorch version: {torch.__version__}")

    if not torch.cuda.is_available():
        report["warnings"].append(
            "CUDA is unavailable; SDXL training/inference will not use the RTX 5080."
        )
        return report

    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "vram_gib": round(properties.total_memory / (1024**3), 2),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    torch_report["devices"] = devices
    if devices and devices[0]["vram_gib"] < 15:
        report["warnings"].append(
            "Detected VRAM is below the 16GB baseline; lower resolution first."
        )
    if devices and not devices[0]["bf16_supported"]:
        report["warnings"].append("bf16 is unavailable; change mixed_precision to fp16.")
    return report


def format_environment(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
