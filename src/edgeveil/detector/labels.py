from edgeveil.spans import EntityType

OUTSIDE_LABEL = "O"
BEGIN_PREFIX = "B-"
INSIDE_PREFIX = "I-"
IGNORED_LABEL_ID = -100

BIO_LABELS: tuple[str, ...] = (
    OUTSIDE_LABEL,
    *(
        f"{prefix}{entity.value}"
        for entity in EntityType
        for prefix in (BEGIN_PREFIX, INSIDE_PREFIX)
    ),
)
LABEL_TO_ID = {label: index for index, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = dict(enumerate(BIO_LABELS))
OUTSIDE_ID = LABEL_TO_ID[OUTSIDE_LABEL]


def begin_id(entity: EntityType) -> int:
    return LABEL_TO_ID[f"{BEGIN_PREFIX}{entity.value}"]


def inside_id(entity: EntityType) -> int:
    return LABEL_TO_ID[f"{INSIDE_PREFIX}{entity.value}"]


def entity_of(label_id: int) -> EntityType | None:
    label = ID_TO_LABEL[label_id]
    if label == OUTSIDE_LABEL:
        return None
    return EntityType(label.removeprefix(BEGIN_PREFIX).removeprefix(INSIDE_PREFIX))


def is_begin(label_id: int) -> bool:
    return ID_TO_LABEL[label_id].startswith(BEGIN_PREFIX)
