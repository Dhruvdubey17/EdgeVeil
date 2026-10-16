from typing import Any

import numpy as np
import torch
from coremltools.converters.mil.frontend.torch.ops import _get_inputs, _make_fill_op
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    _TORCH_OPS_REGISTRY,
    register_torch_op,
)
from coremltools.converters.mil.frontend.torch.utils import (
    NUM_TO_NUMPY_DTYPE,
    TORCH_DTYPE_TO_NUM,
)
from coremltools.converters.mil.mil.types import nptype_from_builtin

NEW_ONES_OP = "new_ones"
NEW_ONES_DTYPE_INPUT = 2


def numpy_dtype(dtype_value: Any) -> type:
    if isinstance(dtype_value, torch.dtype):
        dtype_value = TORCH_DTYPE_TO_NUM[dtype_value]
    return NUM_TO_NUMPY_DTYPE[int(dtype_value)]


def fill_dtype(source: Any, dtype_input: Any) -> type:
    requested = None if dtype_input is None else dtype_input.val
    dtype = (
        nptype_from_builtin(source.dtype)
        if requested is None
        else numpy_dtype(requested)
    )
    if dtype is not np.int32 and np.issubdtype(dtype, np.integer):
        return np.int32
    return dtype


def new_ones(context: Any, node: Any) -> None:
    inputs = _get_inputs(context, node, min_expected=2)
    source, size = inputs[0], inputs[1]
    dtype_input = (
        inputs[NEW_ONES_DTYPE_INPUT] if len(inputs) > NEW_ONES_DTYPE_INPUT else None
    )
    dtype = fill_dtype(source, dtype_input)
    context.add(_make_fill_op(size, dtype(1), node.name))


def register_missing_torch_ops() -> None:
    if NEW_ONES_OP not in _TORCH_OPS_REGISTRY:
        register_torch_op(new_ones)
