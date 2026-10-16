from pathlib import Path
from typing import cast

import coremltools
import numpy as np
import torch
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

from edgeveil.detector.builds import (
    ANE_SEQUENCE_LENGTHS,
    INT4_BLOCK_SIZE,
    Backend,
    Precision,
    build_path,
)
from edgeveil.detector.inference import (
    ATTENTION_BIAS_NAME,
    ATTENTION_MASK_NAME,
    INPUT_IDS_NAME,
    LOGITS_NAME,
    POSITION_IDS_NAME,
)
from edgeveil.export.coreml_ops import register_missing_torch_ops
from edgeveil.export.neural_engine import load_neural_engine_model
from edgeveil.export.torch_model import load_logits_model

COREML_DEPLOYMENT_TARGET = coremltools.target.macOS15
TRACE_SEQUENCE_LENGTH = max(ANE_SEQUENCE_LENGTHS)
WEIGHT_QUANTIZATION = {
    Precision.INT8: OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"
    ),
    Precision.INT4: OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=INT4_BLOCK_SIZE,
    ),
}


def traced_model(model_dir: Path) -> torch.jit.ScriptModule:
    input_ids = torch.ones((1, TRACE_SEQUENCE_LENGTH), dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        traced = torch.jit.trace(
            load_logits_model(model_dir), (input_ids, attention_mask)
        )
    return cast("torch.jit.ScriptModule", traced)


def traced_neural_engine_model(model_dir: Path) -> torch.jit.ScriptModule:
    shape = (1, TRACE_SEQUENCE_LENGTH)
    input_ids = torch.ones(shape, dtype=torch.int64)
    position_ids = torch.arange(TRACE_SEQUENCE_LENGTH)[None, :]
    attention_bias = torch.zeros((*shape, 1, 1))
    with torch.inference_mode():
        traced = torch.jit.trace(
            load_neural_engine_model(model_dir),
            (input_ids, position_ids, attention_bias),
        )
    return cast("torch.jit.ScriptModule", traced)


def enumerated_shape(trailing: tuple[int, ...] = ()) -> coremltools.EnumeratedShapes:
    return coremltools.EnumeratedShapes(
        shapes=[(1, length, *trailing) for length in ANE_SEQUENCE_LENGTHS],
        default=(1, TRACE_SEQUENCE_LENGTH, *trailing),
    )


def neural_engine_inputs() -> list[coremltools.TensorType]:
    return [
        coremltools.TensorType(
            name=INPUT_IDS_NAME, shape=enumerated_shape(), dtype=np.int32
        ),
        coremltools.TensorType(
            name=POSITION_IDS_NAME, shape=enumerated_shape(), dtype=np.int32
        ),
        coremltools.TensorType(
            name=ATTENTION_BIAS_NAME, shape=enumerated_shape((1, 1)), dtype=np.float32
        ),
    ]


def enumerated_inputs() -> list[coremltools.TensorType]:
    shapes = [(1, length) for length in ANE_SEQUENCE_LENGTHS]
    default = (1, TRACE_SEQUENCE_LENGTH)
    return [
        coremltools.TensorType(
            name=name,
            shape=coremltools.EnumeratedShapes(shapes=shapes, default=default),
            dtype=np.int32,
        )
        for name in (INPUT_IDS_NAME, ATTENTION_MASK_NAME)
    ]


def convert_fp16(model_dir: Path, backend: Backend) -> coremltools.models.MLModel:
    register_missing_torch_ops()
    is_neural_engine = backend is Backend.COREML_ANE
    converted = coremltools.convert(
        traced_neural_engine_model(model_dir)
        if is_neural_engine
        else traced_model(model_dir),
        inputs=neural_engine_inputs() if is_neural_engine else enumerated_inputs(),
        outputs=[coremltools.TensorType(name=LOGITS_NAME)],
        convert_to="mlprogram",
        compute_precision=coremltools.precision.FLOAT16,
        minimum_deployment_target=COREML_DEPLOYMENT_TARGET,
    )
    return cast("coremltools.models.MLModel", converted)


def quantize_weights(
    fp16_path: Path, precision: Precision
) -> coremltools.models.MLModel:
    source = coremltools.models.MLModel(str(fp16_path), skip_model_load=True)
    config = OptimizationConfig(global_config=WEIGHT_QUANTIZATION[precision])
    quantized = linear_quantize_weights(source, config)
    return cast("coremltools.models.MLModel", quantized)


def build_coreml(model_dir: Path, backend: Backend, precision: Precision) -> Path:
    output_path = build_path(model_dir, backend, precision)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fp16_path = build_path(model_dir, backend, Precision.FP16)
    if precision is Precision.FP16:
        convert_fp16(model_dir, backend).save(str(output_path))
        return output_path
    if not fp16_path.exists():
        convert_fp16(model_dir, backend).save(str(fp16_path))
    quantize_weights(fp16_path, precision).save(str(output_path))
    return output_path
