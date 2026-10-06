# Design notes

## Python version

`requires-python` is `>=3.13,<3.14`. Checked on PyPI on 2026-10-06:

| Package | Latest stable | Newest CPython wheel (macOS arm64 and Linux x86_64) |
|---|---|---|
| torch | 2.14.1 | 3.14 |
| onnxruntime | 1.30.0 | 3.14 |
| qai-hub | 0.56.0 | pure Python, `>=3.10` |
| coremltools | 9.0 | 3.13 |

coremltools sets the ceiling. Only the 9.1.dev1 pre-release ships cp314 wheels. Move to 3.14 once a stable coremltools release does.

## Tests and sockets

pytest runs with `--disable-socket --allow-unix-socket`. With `--disable-socket` alone, every anyio test on the asyncio backend fails during teardown with `AttributeError: '_UnixSelectorEventLoop' object has no attribute '_ssock'`, because the selector event loop builds its self-pipe with `socket.socketpair()`, which pytest-socket blocks. `--allow-unix-socket` permits only `AF_UNIX`, so TCP and UDP stay blocked.

## CI dependencies

CI runs `uv sync --locked --all-extras` so pyright can resolve the training and data modules. On Linux, torch comes from the PyTorch CPU index (`tool.uv.sources` in `pyproject.toml`). From PyPI, the Linux x86_64 torch wheel pulls 15 `nvidia-*` CUDA packages, about 3.0 GB in all; CI has no GPU. macOS keeps the PyPI wheel, which includes MPS.

The test step sets `HF_HUB_OFFLINE=1` in addition to pytest-socket, so a library that tries to reach the Hub fails fast instead of timing out.

## Training data source

The training and validation data is `ai4privacy/pii-masking-openpii-1.5m`, English rows only, pinned to revision `a785eb528e28be2693c3718a27e066970de5dadb` in `edgeveil.dataset.openpii`.

- License. It is CC-BY-4.0, which allows a public repo and a published model with attribution. The 300k and 400k sets are academic and non-commercial only and treat models as derivative works. The 500k set carries Llama Community License terms that would force a "Llama" prefix on the model name.
- No mixing with OpenPII 1M. 1.5M extends 1M, so taking rows from both would duplicate text. The English rows of 1.5M split into `source_dataset` values `base` (179,419 rows, the same count as the English rows in 1M's `distribution.json`) and `asia-pacific` (25,230 rows, the en-SG and en-IN additions).
- Streaming. `edgeveil data download` streams each split and keeps rows whose `language` is `en`. Filtering happens on the client, so the whole JSONL (5.3 GB) still crosses the network once, but only the English subset is written to disk.
- Pinned revision. ai4privacy updates its dataset repos after release (on 2026-10-06 the Hub API showed a last-modified date of 2026-06-03 for 38 of their 40 datasets). A pinned SHA keeps a rerun on the same rows.
- Unknown labels raise `UnknownLabelError` in `map_entities` rather than becoming O, so a label the mapping has never seen fails loudly instead of silently teaching the model that a new PII type is O.
- Writing the subset. `IterableDataset.to_parquet` in datasets 5.1.0 concatenates the streamed Arrow batches with `pa.concat_tables` and does not cast them to the declared features. A batch where every `privacy_mask` is empty is inferred as `list<null>`, and the concatenation fails with `ArrowInvalid: Schema at index 633 was different` after the whole train split has streamed. `Dataset.from_generator` with the stream's features casts each row, so the download writes through it instead.

## Cleaning before training

`edgeveil data prepare` runs four checks, in this order, and writes only counts to `data/openpii/clean/report.json`.

1. Offsets. For every `privacy_mask` entry, `source_text[start:end]` must equal the stored value. A row with any mismatch is dropped whole. Dropping only the bad entry would leave that PII in the text labeled O, which teaches the model to miss it.
2. Overlaps. A row whose entries overlap is dropped, because BIO tags cannot represent overlapping spans.
3. Off-card labels. The card lists 19 labels, but 825 English rows (655 train, 170 validation, all from `asia-pacific`) also carry others: TIME (2,491 entities), USERNAME (29), ORGANISATION (24), CURRENCY, AMOUNT, COUNTRY, URL, BANKNAME, IPV4, ACCOUNTNUM, and a few more. These rows are dropped. Mapping the extra labels to O would teach the model that usernames, IP addresses, and account numbers are not sensitive, and the rows are 0.4% of the English data.
4. Duplicates. Exact `source_text` duplicates are removed across both splits. When a text is in both, the validation copy is kept and the train copy dropped, so validation stays a fixed set and per-type thresholds tuned on it never see training text. Within a split the first occurrence is kept.

## Label mapping

The mapping lives in `edgeveil.dataset.openpii`, and `docs/labeling.md` states the same rules for hand labeling, so training labels and test labels agree.

- PERSON merges GIVENNAME and SURNAME across whitespace only. A comma is ambiguous between an inverted name (`Quill, Ada`) and a list of people, and splitting is safe for masking because both parts are still masked. A lone given name or surname is PERSON because a first name alone can identify someone in context.
- TITLE stays outside PERSON. It identifies no one, and keeping `Dr.` in the prompt helps the LLM write a natural reply.
- ADDRESS merges STREET, BUILDINGNUM, CITY, and ZIPCODE across separators and an optional two-letter region code, which covers the US `City, ST ZIP` layout. A group needs a STREET, BUILDINGNUM, or ZIPCODE to count. A city on its own is common in ordinary text, rarely identifying, and masking it degrades replies.
- GOV_ID collects IDCARDNUM, PASSPORTNUM, DRIVERLICENSENUM, and TAXNUM. They are identifying, but optional like SSN and CREDIT_CARD, so a deployment can turn them off.
- Whitespace at either edge of an entity is trimmed before grouping. In the English validation split, 878 TELEPHONENUM spans end in a space, which would otherwise disagree with the hand-labeling rule that spans carry no surrounding whitespace.
- Address parts separated by a dash do not merge, because a dash is not one of the agreed separators. In the English validation split, CITY is followed by ZIPCODE across ` – ` 853 times and across ` - ` 253 times. Each of those drops the city and keeps the ZIP code as its own ADDRESS span. Unit words such as `Apt.` between a street and a building number split the address in the same way.
- DATE, AGE, GENDER, SEX, and TITLE map to O. They are quasi-identifiers, outside the v1 entity set.

## Synthetic credentials

OpenPII has no password, key, or token labels, so CREDENTIAL training data is synthetic.

- A seeded generator in the package writes only under `data/`. It inserts fake passwords, API keys, and tokens into OpenPII English sentences and a few templated contexts, in about 3 to 5% of training rows.
- Hard negatives labeled O (commit SHAs, UUIDs, checksums, base64 blobs) teach the model that high-entropy text alone is not a secret.
- Provider key formats are assembled at runtime from parts, and no literal in a real provider format is stored in the repo, because GitHub secret scanning flags such strings.
- The test set's credentials are written by hand. If the generator also produced the test credentials, the test would measure how well the model learned the generator rather than real secrets.
