from dataclasses import dataclass
from enum import StrEnum

MIN_SCORE = 0.0
MAX_SCORE = 1.0


class EntityType(StrEnum):
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    CREDENTIAL = "CREDENTIAL"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    GOV_ID = "GOV_ID"


class SpanSource(StrEnum):
    MODEL = "model"
    VALIDATOR = "validator"


class InvalidSpanError(ValueError):
    pass


def check_offsets(start: int, end: int) -> None:
    if start < 0:
        raise InvalidSpanError(f"span start {start} is negative")
    if end <= start:
        raise InvalidSpanError(f"span end {end} must be greater than start {start}")


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    label: EntityType
    score: float
    source: SpanSource

    def __post_init__(self) -> None:
        check_offsets(self.start, self.end)
        if not MIN_SCORE <= self.score <= MAX_SCORE:
            raise InvalidSpanError(
                f"span score {self.score} is outside [{MIN_SCORE}, {MAX_SCORE}]"
            )

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True, slots=True)
class GoldSpan:
    start: int
    end: int
    label: EntityType

    def __post_init__(self) -> None:
        check_offsets(self.start, self.end)
