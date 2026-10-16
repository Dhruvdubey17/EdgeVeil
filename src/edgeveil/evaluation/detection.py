from enum import StrEnum


class Detection(StrEnum):
    MODEL = "model"
    VALIDATORS = "validators"
    COMBINED = "combined"


def results_stem(detection: Detection, preset: str, run_name: str) -> str:
    if detection is Detection.VALIDATORS:
        return f"{preset}-{Detection.VALIDATORS.value}"
    if detection is Detection.COMBINED:
        return f"{run_name}-{Detection.COMBINED.value}"
    return run_name
