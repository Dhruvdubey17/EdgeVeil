import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import onnx
import torch
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.transformers import float16
from torch import nn
from transformers import AutoModelForTokenClassification

from edgeveil.detector.builds import Backend, Precision, build_path
from edgeveil.detector.inference import ATTENTION_MASK_NAME, INPUT_IDS_NAME, LOGITS_NAME
from edgeveil.detector.windows import MAX_WINDOW_TOKENS, Window, load_tokenizer
from edgeveil.export.calibration import WindowCalibrationReader, validation_sample

ONNX_OPSET_VERSION = 18
EXAMPLE_BATCH_SIZE = 2
EXAMPLE_SEQUENCE_LENGTH = 16
INT4_BITS = 4
INT4_BLOCK_SIZE = 32


class MissingBuildError(FileNotFoundError):
    pass


class LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def load_logits_model(model_dir: Path) -> LogitsOnly:
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return LogitsOnly(model).eval()


def example_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    shape = (EXAMPLE_BATCH_SIZE, EXAMPLE_SEQUENCE_LENGTH)
    input_ids = torch.ones(shape, dtype=torch.int64)
    return input_ids, torch.ones_like(input_ids)


def export_fp32(model_dir: Path, output_path: Path) -> None:
    batch = torch.export.Dim("batch")
    sequence = torch.export.Dim("sequence", max=MAX_WINDOW_TOKENS)
    dynamic_axes = {0: batch, 1: sequence}
    torch.onnx.export(
        load_logits_model(model_dir),
        example_inputs(),
        output_path,
        input_names=[INPUT_IDS_NAME, ATTENTION_MASK_NAME],
        output_names=[LOGITS_NAME],
        dynamic_shapes={
            INPUT_IDS_NAME: dynamic_axes,
            ATTENTION_MASK_NAME: dynamic_axes,
        },
        opset_version=ONNX_OPSET_VERSION,
        dynamo=True,
        external_data=False,
    )


def convert_fp16(fp32_path: Path, output_path: Path) -> None:
    model = float16.convert_float_to_float16(onnx.load(fp32_path), keep_io_types=True)
    onnx.save(model, output_path)


def preprocessed_for_quantization(fp32_path: Path, scratch_dir: Path) -> Path:
    preprocessed = scratch_dir / "preprocessed.onnx"
    quant_pre_process(fp32_path, preprocessed, skip_symbolic_shape=True)
    return preprocessed


def quantize_int8_dynamic(fp32_path: Path, output_path: Path) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        source = preprocessed_for_quantization(fp32_path, Path(scratch))
        quantize_dynamic(source, output_path, weight_type=QuantType.QInt8)


def quantize_int8_static(
    fp32_path: Path, output_path: Path, calibration: Sequence[Window]
) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        source = preprocessed_for_quantization(fp32_path, Path(scratch))
        quantize_static(
            source,
            output_path,
            WindowCalibrationReader(calibration),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )


def quantize_int4(fp32_path: Path, output_path: Path) -> None:
    quantizer = MatMulNBitsQuantizer(
        onnx.load(fp32_path),
        bits=INT4_BITS,
        block_size=INT4_BLOCK_SIZE,
        is_symmetric=True,
    )
    quantizer.process()
    cast("ONNXModel", quantizer.model).save_model_to_file(str(output_path))


def build_onnx(model_dir: Path, precision: Precision, data_dir: Path) -> Path:
    output_path = build_path(model_dir, Backend.ONNXRUNTIME, precision)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if precision is Precision.FP32:
        export_fp32(model_dir, output_path)
        return output_path
    fp32_path = build_path(model_dir, Backend.ONNXRUNTIME, Precision.FP32)
    if not fp32_path.exists():
        raise MissingBuildError(f"export the fp32 build first: {fp32_path}")
    if precision is Precision.INT8_STATIC:
        sample = validation_sample(data_dir, load_tokenizer(model_dir))
        quantize_int8_static(fp32_path, output_path, sample.calibration)
        return output_path
    converters = {
        Precision.FP16: convert_fp16,
        Precision.INT8_DYNAMIC: quantize_int8_dynamic,
        Precision.INT4: quantize_int4,
    }
    converters[precision](fp32_path, output_path)
    return output_path
