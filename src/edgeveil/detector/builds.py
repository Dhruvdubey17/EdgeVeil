import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from edgeveil.detector.decoding import Thresholds
from edgeveil.detector.protocol import BatchDetector

ONNX_DIR_NAME = "onnx"
COREML_DIR_NAME = "coreml"
TORCH_WEIGHTS_NAME = "model.safetensors"
REFERENCE_RUN_SUFFIX = "fp32-torch"
INT4_BLOCK_SIZE = 32
ANE_SEQUENCE_LENGTHS = (128, 256, 512)


class UnsupportedBuildError(ValueError):
    pass


class Backend(StrEnum):
    TORCH = "torch"
    ONNXRUNTIME = "onnxruntime"
    COREML = "coreml"
    QUALCOMM = "qualcomm"


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT8_DYNAMIC = "int8-dynamic"
    INT8_STATIC = "int8-static"
    INT4 = "int4"


class ComputeUnits(StrEnum):
    ALL = "all"
    CPU_ONLY = "cpu-only"


QUANTIZATION_SCHEMES: dict[Backend, dict[Precision, str]] = {
    Backend.TORCH: {Precision.FP32: "FP32 weights and activations"},
    Backend.ONNXRUNTIME: {
        Precision.FP32: "FP32 weights and activations",
        Precision.FP16: "FP16 weights and activations, FP32 inputs and outputs",
        Precision.INT8_DYNAMIC: (
            "INT8 weights, activations quantized per batch at run time"
        ),
        Precision.INT8_STATIC: (
            "INT8 weights and activations in QDQ format, ranges calibrated "
            "on validation windows"
        ),
        Precision.INT4: "INT4 weights in blocks (MatMulNBits), FP32 activations",
    },
    Backend.COREML: {
        Precision.FP16: "FP16 weights and activations",
        Precision.INT8: "INT8 weights per channel, FP16 activations",
        Precision.INT4: "INT4 weights in blocks, FP16 activations",
    },
    Backend.QUALCOMM: {},
}


def check_supported(backend: Backend, precision: Precision) -> None:
    supported = QUANTIZATION_SCHEMES[backend]
    if precision not in supported:
        names = ", ".join(item.value for item in supported) or "none yet"
        raise UnsupportedBuildError(
            f"{backend.value} supports precisions: {names}; got {precision.value}"
        )


def build_path(model_dir: Path, backend: Backend, precision: Precision) -> Path:
    check_supported(backend, precision)
    if backend is Backend.TORCH:
        return model_dir / TORCH_WEIGHTS_NAME
    if backend is Backend.ONNXRUNTIME:
        return model_dir / ONNX_DIR_NAME / f"{precision.value}.onnx"
    return model_dir / COREML_DIR_NAME / f"{precision.value}.mlpackage"


def run_name(
    preset: str,
    backend: Backend,
    precision: Precision,
    compute_units: ComputeUnits | None = None,
) -> str:
    name = f"{preset}-{precision.value}-{backend.value}"
    units = effective_compute_units(backend, compute_units)
    if units is None:
        return name
    return f"{name}-{units.value}"


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_sha256(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(f"{item.relative_to(path).as_posix()}\0".encode())
            digest.update(f"{file_sha256(item)}\n".encode())
    return digest.hexdigest()


def build_size_bytes(path: Path) -> int:
    files = [path] if path.is_file() else list(path.rglob("*"))
    return sum(item.stat().st_size for item in files if item.is_file())


@dataclass(frozen=True, slots=True)
class BuildRecord:
    model_dir: str
    backend: str
    precision: str
    compute_units: str | None
    path: str
    sha256: str
    size_bytes: int
    quantization: str


def effective_compute_units(
    backend: Backend, compute_units: ComputeUnits | None
) -> ComputeUnits | None:
    if backend is not Backend.COREML:
        return None
    return compute_units or ComputeUnits.ALL


def build_record(
    model_dir: Path,
    backend: Backend,
    precision: Precision,
    compute_units: ComputeUnits | None = None,
) -> BuildRecord:
    path = build_path(model_dir, backend, precision)
    units = effective_compute_units(backend, compute_units)
    return BuildRecord(
        model_dir=str(model_dir),
        backend=backend.value,
        precision=precision.value,
        compute_units=None if units is None else units.value,
        path=str(path),
        sha256=build_sha256(path),
        size_bytes=build_size_bytes(path),
        quantization=QUANTIZATION_SCHEMES[backend][precision],
    )


def reference_thresholds_name(preset: str) -> str:
    return f"{preset}-{REFERENCE_RUN_SUFFIX}.json"


def load_detector(
    model_dir: Path,
    backend: Backend,
    precision: Precision,
    thresholds: Thresholds | None = None,
    compute_units: ComputeUnits = ComputeUnits.ALL,
) -> BatchDetector:
    path = build_path(model_dir, backend, precision)
    if backend is Backend.TORCH:
        from edgeveil.detector.torch_backend import TorchDetector

        return TorchDetector(model_dir, thresholds)
    if backend is Backend.ONNXRUNTIME:
        from edgeveil.detector.onnx_backend import OnnxRuntimeDetector

        return OnnxRuntimeDetector(path, model_dir, thresholds)
    raise UnsupportedBuildError(
        f"no {backend.value} detector yet for {compute_units.value} compute units"
    )
