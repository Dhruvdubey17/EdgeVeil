import plistlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from edgeveil.bench.power import (
    NoSamplesInPhaseError,
    Phase,
    PowerLogFormatError,
    PowerLoop,
    energy_report,
    read_powermetrics_plist,
)

START = 1_800_000_000.0
HALF_SECOND_NS = 500_000_000


SAMPLES_PER_PHASE = 20


def sample(end: float, cpu: float, gpu: float, ane: float) -> bytes:
    document = {
        "timestamp": datetime.fromtimestamp(end, tz=UTC),
        "elapsed_ns": HALF_SECOND_NS,
        "processor": {"cpu_power": cpu, "gpu_power": gpu, "ane_power": ane},
    }
    return plistlib.dumps(document)


def write_log(path: Path, documents: list[bytes]) -> Path:
    path.write_bytes(b"\x00".join(documents) + b"\x00")
    return path


def test_energy_per_window_integrates_power_over_the_loop(tmp_path: Path) -> None:
    idle = [sample(START + 0.5, 100.0, 10.0, 0.0)] + [
        sample(0.0, 100.0, 10.0, 0.0) for _ in range(SAMPLES_PER_PHASE - 1)
    ]
    busy = [sample(0.0, 300.0, 10.0, 900.0) for _ in range(SAMPLES_PER_PHASE)]
    log = write_log(tmp_path / "power.plist", idle + busy)
    loop = PowerLoop(
        idle=Phase(START, START + 10),
        inference=Phase(START + 10, START + 20),
        windows=500,
        window_tokens=512,
    )
    report = energy_report(read_powermetrics_plist(log), loop)
    assert report.samples == 12
    assert report.windows_per_second == pytest.approx(50.0)
    assert report.millijoules_per_window["ane"] == pytest.approx(900.0 * 10 / 500)
    above_idle = report.millijoules_per_window_above_idle
    assert above_idle["cpu"] == pytest.approx(200.0 * 10 / 500)
    assert above_idle["gpu"] == pytest.approx(0.0)
    assert above_idle["total"] == pytest.approx((200.0 + 900.0) * 10 / 500)


def test_a_loop_outside_the_log_is_reported(tmp_path: Path) -> None:
    log = write_log(tmp_path / "power.plist", [sample(START + 0.5, 1.0, 1.0, 1.0)])
    loop = PowerLoop(
        Phase(START + 10, START + 12), Phase(START + 12, START + 14), 1, 512
    )
    with pytest.raises(NoSamplesInPhaseError):
        energy_report(read_powermetrics_plist(log), loop)


def test_missing_power_keys_name_what_the_log_has(tmp_path: Path) -> None:
    document = plistlib.dumps(
        {
            "timestamp": datetime.fromtimestamp(START, tz=UTC),
            "elapsed_ns": 1,
            "processor": {"cpu_power": 1.0},
        }
    )
    with pytest.raises(PowerLogFormatError, match="gpu_power"):
        read_powermetrics_plist(write_log(tmp_path / "power.plist", [document]))
