import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import coremltools
from coremltools.models.compute_device import (
    MLCPUComputeDevice,
    MLGPUComputeDevice,
    MLNeuralEngineComputeDevice,
)
from coremltools.models.compute_plan import (
    MLComputePlan,
    MLModelStructureProgramBlock,
    MLModelStructureProgramOperation,
)

from edgeveil.bench.environment import environment_record
from edgeveil.detector.builds import BuildRecord, ComputeUnits
from edgeveil.detector.coreml_backend import COREML_COMPUTE_UNITS

MAIN_FUNCTION = "main"
NEURAL_ENGINE = "neural_engine"
GPU = "gpu"
CPU = "cpu"
UNASSIGNED = "unassigned"
RESULT_DECIMALS = 4
DEVICE_NAMES = (
    (MLNeuralEngineComputeDevice, NEURAL_ENGINE),
    (MLGPUComputeDevice, GPU),
    (MLCPUComputeDevice, CPU),
)


class UnexpectedModelTypeError(TypeError):
    pass


@dataclass(frozen=True, slots=True)
class PlacementReport:
    operations: int
    unassigned_operations: int
    by_device: dict[str, int]
    cost_by_device: dict[str, float]
    neural_engine_share: float
    neural_engine_cost_share: float
    operators_off_neural_engine: dict[str, dict[str, int]]


def all_operations(
    block: MLModelStructureProgramBlock,
) -> Iterator[MLModelStructureProgramOperation]:
    for operation in block.operations:
        yield operation
        for nested in operation.blocks:
            yield from all_operations(nested)


def device_name(device: object) -> str:
    for device_type, name in DEVICE_NAMES:
        if isinstance(device, device_type):
            return name
    return type(device).__name__


def compute_plan(package_path: Path, compute_units: ComputeUnits) -> MLComputePlan:
    units = COREML_COMPUTE_UNITS[compute_units]
    model = coremltools.models.MLModel(str(package_path), compute_units=units)
    return MLComputePlan.load_from_path(
        model.get_compiled_model_path(), compute_units=units
    )


def share(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


@dataclass(frozen=True, slots=True)
class OperationPlacement:
    operator: str
    device: str | None
    cost: float


def operation_placements(
    package_path: Path, compute_units: ComputeUnits
) -> Iterator[OperationPlacement]:
    plan = compute_plan(package_path, compute_units)
    program = plan.model_structure.program
    if program is None:
        raise UnexpectedModelTypeError("compute plan has no ML Program")
    for operation in all_operations(program.functions[MAIN_FUNCTION].block):
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
        yield OperationPlacement(
            operator=operation.operator_name,
            device=None
            if usage is None
            else device_name(usage.preferred_compute_device),
            cost=0.0 if cost is None else cost.weight,
        )


def tally_placements(placements: Iterable[OperationPlacement]) -> PlacementReport:
    by_device: Counter[str] = Counter()
    cost_by_device: defaultdict[str, float] = defaultdict(float)
    off_neural_engine: dict[str, Counter[str]] = {}
    for placement in placements:
        if placement.device is None:
            by_device[UNASSIGNED] += 1
            continue
        by_device[placement.device] += 1
        cost_by_device[placement.device] += placement.cost
        if placement.device != NEURAL_ENGINE:
            operators = off_neural_engine.setdefault(placement.device, Counter())
            operators[placement.operator] += 1
    assigned = by_device.total() - by_device[UNASSIGNED]
    return PlacementReport(
        operations=assigned,
        unassigned_operations=by_device[UNASSIGNED],
        by_device=dict(by_device),
        cost_by_device={
            key: round(value, RESULT_DECIMALS) for key, value in cost_by_device.items()
        },
        neural_engine_share=share(by_device[NEURAL_ENGINE], assigned),
        neural_engine_cost_share=share(
            cost_by_device[NEURAL_ENGINE], sum(cost_by_device.values())
        ),
        operators_off_neural_engine={
            device: dict(counts.most_common())
            for device, counts in off_neural_engine.items()
        },
    )


def measure_placement(
    package_path: Path, compute_units: ComputeUnits
) -> PlacementReport:
    return tally_placements(operation_placements(package_path, compute_units))


def write_placement_results(
    report: PlacementReport, build: BuildRecord, path: Path
) -> None:
    payload = {
        "precision": build.precision,
        "backend": build.backend,
        "compute_units": build.compute_units,
        "build": asdict(build),
        "source": (
            "coremltools MLComputePlan: preferred compute device per ML Program "
            "operation in the main function; operations without a device usage "
            "are counted as unassigned and left out of the shares"
        ),
        **asdict(report),
        "environment": environment_record(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
