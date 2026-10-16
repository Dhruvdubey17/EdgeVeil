import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "seqeval",
    "onnxruntime",
    "coremltools",
    "onnx",
    "onnxscript",
)
NOT_INSTALLED = "not installed"
NOT_AVAILABLE = "not available"


def chip_name() -> str:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def sysctl_value(name: str) -> str:
    if sys.platform != "darwin":
        return NOT_AVAILABLE
    result = subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or NOT_AVAILABLE


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return NOT_INSTALLED


def environment_record() -> dict[str, Any]:
    return {
        "chip": chip_name(),
        "machine_model": sysctl_value("hw.model"),
        "memory_bytes": sysctl_value("hw.memsize"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in TRACKED_PACKAGES},
    }
