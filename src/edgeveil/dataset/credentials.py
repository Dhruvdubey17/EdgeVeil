import base64
import json
import random
import re
import string
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from edgeveil.dataset.openpii import DatasetError
from edgeveil.spans import EntityType, GoldSpan

CREDENTIAL_ROW_FRACTION = 0.04
STANDALONE_CONTEXT_SHARE = 0.25
HARD_NEGATIVE_ROW_FRACTION = 0.02
AUGMENTATION_SEED = 20261006

PASSWORD_LENGTH_RANGE = (10, 20)
PASSPHRASE_WORD_COUNT_RANGE = (3, 4)
OPAQUE_TOKEN_LENGTH_RANGE = (32, 64)
BASE64_BLOB_BYTES_RANGE = (24, 96)
JWT_SIGNATURE_BYTES = 32
JWT_SUBJECT_RANGE = (100_000, 999_999)
JWT_ISSUED_AT_RANGE = (1_700_000_000, 1_800_000_000)
SHORT_SHA_LENGTH = 7
COMMIT_SHA_LENGTH = 40
MD5_HEX_LENGTH = 32
SHA256_HEX_LENGTH = 64

ALPHANUMERIC = string.ascii_letters + string.digits
UPPERCASE_ALPHANUMERIC = string.ascii_uppercase + string.digits
URL_SAFE = ALPHANUMERIC + "-_"
HEX_DIGITS = "0123456789abcdef"
PASSWORD_SYMBOLS = "!@#$%^&*-_=+?"
PASSPHRASE_SEPARATORS = "-._"
PASSPHRASE_WORDS = (
    "lantern",
    "orbit",
    "pickle",
    "harbor",
    "violet",
    "cactus",
    "meadow",
    "copper",
    "falcon",
    "pebble",
    "thunder",
    "marble",
)

MARKER = re.compile(r"\{(\w+)\}")
SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+|\n")

CREDENTIAL_SENTENCES = (
    "My password is {password}.",
    "The temporary password is {password}, please change it after you log in.",
    "Password: {password}",
    "I changed the Wi-Fi password to {password} last night.",
    "Here is the API key: {api_key}",
    "Our production API key is {api_key}, do not share it.",
    "Set the key to {api_key} in the dashboard settings.",
    "Use this access token: {token}",
    "The session token {token} expires in an hour.",
    "Authorization: Bearer {token}",
)
STANDALONE_TEMPLATES = (
    "DATABASE_HOST=localhost\nDATABASE_PASSWORD={password}\nAPI_KEY={api_key}\nDEBUG=false",
    'curl -H "Authorization: Bearer {token}" https://api.example.com/v1/orders',
    '{\n  "client_id": "app-412",\n  "client_secret": "{api_key}"\n}',
    'client = Client(api_key="{api_key}", timeout=30)',
    "auth:\n  username: deploy\n  password: {password}\n  token: {token}",
    "export SERVICE_API_KEY={api_key}",
)
HARD_NEGATIVE_TEMPLATES = (
    "Fixed in commit {commit_sha}.",
    "See {short_sha} for the revert.",
    "Request ID: {uuid}",
    "Trace {uuid} completed without errors.",
    "The SHA-256 checksum of the installer is {sha256}.",
    "MD5: {md5}",
    "Thumbnail: data:image/png;base64,{base64}",
)


class TemplateError(DatasetError):
    pass


class Origin(StrEnum):
    OPENPII = "openpii"
    OPENPII_WITH_CREDENTIAL = "openpii_with_credential"
    OPENPII_WITH_HARD_NEGATIVE = "openpii_with_hard_negative"
    CREDENTIAL_TEMPLATE = "credential_template"


@dataclass(frozen=True, slots=True)
class LabeledText:
    text: str = field(repr=False)
    spans: tuple[GoldSpan, ...]


@dataclass(frozen=True, slots=True)
class KeyShape:
    prefix_parts: tuple[str, ...]
    segments: tuple[tuple[str, int], ...]
    joiner: str = ""


API_KEY_SHAPES = (
    KeyShape(("AK", "IA"), ((UPPERCASE_ALPHANUMERIC, 16),)),
    KeyShape(("gh", "p_"), ((ALPHANUMERIC, 36),)),
    KeyShape(("s", "k-"), ((ALPHANUMERIC, 48),)),
    KeyShape(("s", "k-", "pr", "oj-"), ((URL_SAFE, 64),)),
    KeyShape(("s", "k-", "an", "t-", "api", "03-"), ((URL_SAFE, 93),)),
    KeyShape(("s", "k_", "li", "ve_"), ((ALPHANUMERIC, 24),)),
    KeyShape(("AI", "za"), ((URL_SAFE, 35),)),
    KeyShape(
        ("xo", "xb-"),
        ((string.digits, 11), (string.digits, 12), (ALPHANUMERIC, 24)),
        joiner="-",
    ),
    KeyShape((), ((HEX_DIGITS, 32),)),
    KeyShape((), ((ALPHANUMERIC, 40),)),
)


@dataclass(frozen=True, slots=True)
class AugmentedSplit:
    rows: list[LabeledText]
    origins: list[Origin]


def random_string(rng: random.Random, alphabet: str, length: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def random_password(rng: random.Random) -> str:
    length = rng.randint(*PASSWORD_LENGTH_RANGE)
    return random_string(rng, ALPHANUMERIC + PASSWORD_SYMBOLS, length)


def passphrase(rng: random.Random) -> str:
    words = rng.sample(PASSPHRASE_WORDS, rng.randint(*PASSPHRASE_WORD_COUNT_RANGE))
    separator = rng.choice(PASSPHRASE_SEPARATORS)
    return separator.join(words) + separator + random_string(rng, string.digits, 2)


def word_with_suffix(rng: random.Random) -> str:
    word = rng.choice(PASSPHRASE_WORDS).capitalize()
    return word + random_string(rng, string.digits, 4) + rng.choice(PASSWORD_SYMBOLS)


PASSWORD_STYLES: tuple[Callable[[random.Random], str], ...] = (
    random_password,
    passphrase,
    word_with_suffix,
)


def generate_password(rng: random.Random) -> str:
    return rng.choice(PASSWORD_STYLES)(rng)


def generate_api_key(rng: random.Random, shape: KeyShape) -> str:
    body = shape.joiner.join(
        random_string(rng, alphabet, length) for alphabet, length in shape.segments
    )
    return "".join(shape.prefix_parts) + body


def base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def json_web_token(rng: random.Random) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(rng.randint(*JWT_SUBJECT_RANGE)),
        "iat": rng.randint(*JWT_ISSUED_AT_RANGE),
    }
    parts = [
        base64url(json.dumps(header, separators=(",", ":")).encode()),
        base64url(json.dumps(payload, separators=(",", ":")).encode()),
        base64url(rng.randbytes(JWT_SIGNATURE_BYTES)),
    ]
    return ".".join(parts)


def opaque_token(rng: random.Random) -> str:
    return random_string(rng, URL_SAFE, rng.randint(*OPAQUE_TOKEN_LENGTH_RANGE))


def generate_token(rng: random.Random) -> str:
    return rng.choice((json_web_token, opaque_token))(rng)


def random_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def base64_blob(rng: random.Random) -> str:
    size = rng.randint(*BASE64_BLOB_BYTES_RANGE)
    return base64.b64encode(rng.randbytes(size)).decode("ascii")


CREDENTIAL_MARKERS: dict[str, Callable[[random.Random], str]] = {
    "password": generate_password,
    "api_key": lambda rng: generate_api_key(rng, rng.choice(API_KEY_SHAPES)),
    "token": generate_token,
}
HARD_NEGATIVE_MARKERS: dict[str, Callable[[random.Random], str]] = {
    "commit_sha": lambda rng: random_string(rng, HEX_DIGITS, COMMIT_SHA_LENGTH),
    "short_sha": lambda rng: random_string(rng, HEX_DIGITS, SHORT_SHA_LENGTH),
    "uuid": random_uuid,
    "sha256": lambda rng: random_string(rng, HEX_DIGITS, SHA256_HEX_LENGTH),
    "md5": lambda rng: random_string(rng, HEX_DIGITS, MD5_HEX_LENGTH),
    "base64": base64_blob,
}


def marker_value(name: str, rng: random.Random) -> tuple[str, bool]:
    if name in CREDENTIAL_MARKERS:
        return CREDENTIAL_MARKERS[name](rng), True
    if name in HARD_NEGATIVE_MARKERS:
        return HARD_NEGATIVE_MARKERS[name](rng), False
    raise TemplateError(f"unknown template marker {name}")


def render_template(template: str, rng: random.Random) -> LabeledText:
    text = ""
    spans: list[GoldSpan] = []
    cursor = 0
    for match in MARKER.finditer(template):
        text += template[cursor : match.start()]
        value, is_secret = marker_value(match.group(1), rng)
        if is_secret:
            spans.append(
                GoldSpan(len(text), len(text) + len(value), EntityType.CREDENTIAL)
            )
        text += value
        cursor = match.end()
    return LabeledText(text=text + template[cursor:], spans=tuple(spans))


def generate_hard_negative(rng: random.Random, template: str) -> LabeledText:
    return render_template(template, rng)


def is_inside_a_span(position: int, spans: Sequence[GoldSpan]) -> bool:
    return any(span.start < position < span.end for span in spans)


def insertion_points(row: LabeledText) -> list[int]:
    boundaries = {0, len(row.text)}
    boundaries.update(match.end() for match in SENTENCE_BOUNDARY.finditer(row.text))
    return sorted(
        point for point in boundaries if not is_inside_a_span(point, row.spans)
    )


def shifted(span: GoldSpan, offset: int) -> GoldSpan:
    return GoldSpan(span.start + offset, span.end + offset, span.label)


def insert_snippet(
    row: LabeledText, snippet: LabeledText, position: int
) -> LabeledText:
    before, after = row.text[:position], row.text[position:]
    lead = " " if before and not before[-1].isspace() else ""
    trail = " " if after and not after[0].isspace() else ""
    inserted = lead + snippet.text + trail
    kept = [
        shifted(span, len(inserted)) if span.start >= position else span
        for span in row.spans
    ]
    added = [shifted(span, position + len(lead)) for span in snippet.spans]
    spans = sorted(kept + added, key=lambda span: span.start)
    return LabeledText(text=before + inserted + after, spans=tuple(spans))


def insert_at_random_point(
    row: LabeledText, snippet: LabeledText, rng: random.Random
) -> LabeledText:
    return insert_snippet(row, snippet, rng.choice(insertion_points(row)))


@dataclass(frozen=True, slots=True)
class AugmentationPlan:
    credential_targets: frozenset[int]
    hard_negative_targets: frozenset[int]
    standalone_rows: int


def plan_augmentation(row_count: int, rng: random.Random) -> AugmentationPlan:
    credential_rows = round(row_count * CREDENTIAL_ROW_FRACTION)
    standalone_rows = round(credential_rows * STANDALONE_CONTEXT_SHARE)
    inserted_rows = credential_rows - standalone_rows
    hard_negative_rows = round(row_count * HARD_NEGATIVE_ROW_FRACTION)
    order = list(range(row_count))
    rng.shuffle(order)
    negatives_end = inserted_rows + hard_negative_rows
    return AugmentationPlan(
        credential_targets=frozenset(order[:inserted_rows]),
        hard_negative_targets=frozenset(order[inserted_rows:negatives_end]),
        standalone_rows=standalone_rows,
    )


def augment_row(
    index: int, row: LabeledText, plan: AugmentationPlan, rng: random.Random
) -> tuple[LabeledText, Origin]:
    if index in plan.credential_targets:
        snippet = render_template(rng.choice(CREDENTIAL_SENTENCES), rng)
        return insert_at_random_point(row, snippet, rng), Origin.OPENPII_WITH_CREDENTIAL
    if index in plan.hard_negative_targets:
        snippet = render_template(rng.choice(HARD_NEGATIVE_TEMPLATES), rng)
        return insert_at_random_point(
            row, snippet, rng
        ), Origin.OPENPII_WITH_HARD_NEGATIVE
    return row, Origin.OPENPII


def augment_split(rows: Sequence[LabeledText], rng: random.Random) -> AugmentedSplit:
    plan = plan_augmentation(len(rows), rng)
    augmented = [augment_row(index, row, plan, rng) for index, row in enumerate(rows)]
    for _ in range(plan.standalone_rows):
        template = render_template(rng.choice(STANDALONE_TEMPLATES), rng)
        augmented.append((template, Origin.CREDENTIAL_TEMPLATE))
    return AugmentedSplit(
        rows=[row for row, _ in augmented],
        origins=[origin for _, origin in augmented],
    )
