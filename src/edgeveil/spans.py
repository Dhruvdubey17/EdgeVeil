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


class SpanSource(StrEnum):
    MODEL = "model"
    VALIDATOR = "validator"


class InvalidSpanError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    label: EntityType
    score: float
    source: SpanSource

    def __post_init__(self) -> None:
        if self.start < 0:
            raise InvalidSpanError(f"span start {self.start} is negative")
        if self.end <= self.start:
            raise InvalidSpanError(
                f"span end {self.end} must be greater than start {self.start}"
            )
        if not MIN_SCORE <= self.score <= MAX_SCORE:
            raise InvalidSpanError(
                f"span score {self.score} is outside [{MIN_SCORE}, {MAX_SCORE}]"
            )

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end
