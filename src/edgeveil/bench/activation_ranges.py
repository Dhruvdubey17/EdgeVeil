import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
from numpy.typing import NDArray
from onnx import numpy_helper

from edgeveil.bench.environment import environment_record
from edgeveil.detector.onnx_backend import CPU_PROVIDER
from edgeveil.detector.windows import Window
from edgeveil.export.calibration import window_feeds
from edgeveil.export.onnx_builds import preprocessed_for_quantization

INT8_MIN = -128
INT8_MAX = 127
TYPICAL_PERCENTILE = 50
TAIL_PERCENTILE = 99.9
LAYER_NORM_OP = "LayerNormalization"
LAYER_NORMS_PER_BLOCK = 2
EMBEDDINGS_LABEL = "embeddings"
CLASSIFIER_LABEL = "classifier"
RESULT_DECIMALS = 6


@dataclass(frozen=True, slots=True)
class QuantizedRange:
    scale: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class ActivationRange:
    tensor: str
    producer: str
    layer: str
    calibrated_low: float
    calibrated_high: float
    step: float
    max_abs: float
    tail_abs: float
    typical_abs: float

    @property
    def outlier_ratio(self) -> float:
        return self.max_abs / self.tail_abs if self.tail_abs else 0.0

    @property
    def typical_in_steps(self) -> float:
        return self.typical_abs / self.step if self.step else 0.0


def scalar(model: onnx.ModelProto, name: str) -> float:
    initializers = {item.name: item for item in model.graph.initializer}
    return float(numpy_helper.to_array(initializers[name]).reshape(-1)[0])


def quantized_ranges(static_model: onnx.ModelProto) -> dict[str, QuantizedRange]:
    ranges: dict[str, QuantizedRange] = {}
    for node in static_model.graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        scale = scalar(static_model, node.input[1])
        zero_point = scalar(static_model, node.input[2])
        ranges[node.input[0]] = QuantizedRange(
            scale=scale,
            low=scale * (INT8_MIN - zero_point),
            high=scale * (INT8_MAX - zero_point),
        )
    return ranges


def layer_label(layer_norms_seen: int, block_count: int) -> str:
    block = (layer_norms_seen - 1) // LAYER_NORMS_PER_BLOCK
    if block < 0:
        return EMBEDDINGS_LABEL
    if block >= block_count:
        return CLASSIFIER_LABEL
    return f"block {block}"


def producers_by_layer(model: onnx.ModelProto) -> dict[str, tuple[str, str]]:
    layer_norms = sum(node.op_type == LAYER_NORM_OP for node in model.graph.node)
    block_count = (layer_norms - 1) // LAYER_NORMS_PER_BLOCK
    located: dict[str, tuple[str, str]] = {}
    layer_norms_seen = 0
    for node in model.graph.node:
        layer = layer_label(layer_norms_seen, block_count)
        for output in node.output:
            located[output] = (node.op_type, layer)
        if node.op_type == LAYER_NORM_OP:
            layer_norms_seen += 1
    return located


def with_outputs(model: onnx.ModelProto, names: Sequence[str]) -> bytes:
    existing = {output.name for output in model.graph.output}
    model.graph.output.extend(
        onnx.helper.make_empty_tensor_value_info(name)
        for name in names
        if name not in existing
    )
    return model.SerializeToString()


def window_magnitudes(
    session: onnxruntime.InferenceSession, names: Sequence[str], window: Window
) -> dict[str, NDArray[np.float64]]:
    outputs = session.run(list(names), window_feeds(window))
    return {
        name: np.abs(np.asarray(value, dtype=np.float64)).reshape(-1)
        for name, value in zip(names, outputs, strict=True)
    }


def measured_magnitudes(
    preprocessed: onnx.ModelProto, names: Sequence[str], windows: Sequence[Window]
) -> dict[str, dict[str, float]]:
    session = onnxruntime.InferenceSession(
        with_outputs(preprocessed, names), providers=[CPU_PROVIDER]
    )
    maxima: dict[str, list[float]] = {name: [] for name in names}
    tails: dict[str, list[float]] = {name: [] for name in names}
    typical: dict[str, list[float]] = {name: [] for name in names}
    for window in windows:
        for name, values in window_magnitudes(session, names, window).items():
            maxima[name].append(float(values.max()))
            tails[name].append(float(np.percentile(values, TAIL_PERCENTILE)))
            typical[name].append(float(np.percentile(values, TYPICAL_PERCENTILE)))
    return {
        name: {
            "max_abs": max(maxima[name]),
            "tail_abs": float(np.median(tails[name])),
            "typical_abs": float(np.median(typical[name])),
        }
        for name in names
    }


def activation_ranges(
    fp32_path: Path, static_path: Path, windows: Sequence[Window]
) -> list[ActivationRange]:
    ranges = quantized_ranges(onnx.load(static_path))
    with tempfile.TemporaryDirectory() as scratch:
        preprocessed = onnx.load(
            preprocessed_for_quantization(fp32_path, Path(scratch))
        )
    located = producers_by_layer(preprocessed)
    names = [name for name in ranges if name in located]
    magnitudes = measured_magnitudes(preprocessed, names, windows)
    return [
        ActivationRange(
            tensor=name,
            producer=located[name][0],
            layer=located[name][1],
            calibrated_low=ranges[name].low,
            calibrated_high=ranges[name].high,
            step=ranges[name].scale,
            **magnitudes[name],
        )
        for name in names
    ]


def rounded(record: ActivationRange) -> dict[str, Any]:
    values = {
        **asdict(record),
        "outlier_ratio": record.outlier_ratio,
        "typical_in_steps": record.typical_in_steps,
    }
    return {
        key: round(value, RESULT_DECIMALS) if isinstance(value, float) else value
        for key, value in values.items()
    }


def write_activation_ranges(
    records: Sequence[ActivationRange], static_path: Path, windows: int, path: Path
) -> None:
    payload = {
        "build": static_path.as_posix(),
        "windows": windows,
        "sample": "the 300 validation calibration windows, batch size 1",
        "measured": (
            "calibrated range and step from each activation QuantizeLinear in the "
            "static INT8 build; max_abs is the largest FP32 magnitude over the "
            "windows; tail_abs and typical_abs are the median over windows of the "
            f"{TAIL_PERCENTILE} and {TYPICAL_PERCENTILE} percentile magnitudes; "
            "outlier_ratio is max_abs / tail_abs; typical_in_steps is "
            "typical_abs / step"
        ),
        "activations": [rounded(record) for record in records],
        "environment": environment_record(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
