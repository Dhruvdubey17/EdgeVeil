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
from edgeveil.detector.inference import ATTENTION_MASK_NAME, INPUT_IDS_NAME, LOGITS_NAME
from edgeveil.export.coreml_ops import register_missing_torch_ops
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


def convert_fp16(model_dir: Path) -> coremltools.models.MLModel:
    register_missing_torch_ops()
    converted = coremltools.convert(
        traced_model(model_dir),
        inputs=enumerated_inputs(),
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


def build_coreml(model_dir: Path, precision: Precision) -> Path:
    output_path = build_path(model_dir, Backend.COREML, precision)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fp16_path = build_path(model_dir, Backend.COREML, Precision.FP16)
    if precision is Precision.FP16:
        convert_fp16(model_dir).save(str(output_path))
        return output_path
    if not fp16_path.exists():
        convert_fp16(model_dir).save(str(fp16_path))
    quantize_weights(fp16_path, precision).save(str(output_path))
    return output_path
