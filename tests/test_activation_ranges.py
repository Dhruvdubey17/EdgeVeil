import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from edgeveil.bench.activation_ranges import (
    CLASSIFIER_LABEL,
    EMBEDDINGS_LABEL,
    ActivationRange,
    layer_label,
    quantized_ranges,
)


def quantize_model(scale: float, zero_point: int) -> onnx.ModelProto:
    node = helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["q"])
    graph = helper.make_graph(
        [node],
        "quantize",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("q", TensorProto.INT8, [2])],
        initializer=[
            numpy_helper.from_array(np.array(scale, dtype=np.float32), "scale"),
            numpy_helper.from_array(np.array(zero_point, dtype=np.int8), "zero"),
        ],
    )
    return helper.make_model(graph)


def test_calibrated_range_follows_scale_and_zero_point() -> None:
    [(name, quantized)] = quantized_ranges(quantize_model(0.5, -28)).items()
    assert name == "x"
    assert quantized.low == pytest.approx(0.5 * (-128 + 28))
    assert quantized.high == pytest.approx(0.5 * (127 + 28))


@pytest.mark.parametrize(
    ("seen", "label"),
    [
        (0, EMBEDDINGS_LABEL),
        (1, "block 0"),
        (2, "block 0"),
        (3, "block 1"),
        (13, CLASSIFIER_LABEL),
    ],
)
def test_layers_are_counted_by_layer_norms(seen: int, label: str) -> None:
    assert layer_label(seen, block_count=6) == label


def test_outlier_ratio_and_resolution() -> None:
    record = ActivationRange("t", "Add", "block 4", -500.0, 80.0, 2.0, 500.0, 2.5, 0.5)
    assert record.outlier_ratio == pytest.approx(200.0)
    assert record.typical_in_steps == pytest.approx(0.25)
