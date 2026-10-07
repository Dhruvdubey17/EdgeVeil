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

- PERSON merges GIVENNAME and SURNAME across whitespace, middle initials, the lowercase particles van, von, de, da, del, la, le, bin, binti, and al, and s/o or d/o, never across commas. The connector lists are closed so that ordinary words between two names (`and`, `with`) never join two people into one span. A comma is ambiguous between an inverted name (`Quill, Ada`) and a list of people, and splitting is safe for masking because both parts are still masked. A lone given name or surname is PERSON because a first name alone can identify someone in context.
- TITLE stays outside PERSON. It identifies no one, and keeping `Dr.` in the prompt helps the LLM write a natural reply.
- ADDRESS merges STREET, BUILDINGNUM, CITY, and ZIPCODE across separators (whitespace, commas, periods, `#`, hyphens, en and em dashes), a two-letter uppercase region code, and a closed, case-insensitive list of unit words (Apt, Apartment, Suite, Ste, Unit, Flat, Floor, Fl, Room, Rm, Bldg, Block, Blk, Level, No). `MAX_ADDRESS_GAP_LENGTH` (16 characters) caps the gap so that two separate addresses joined by punctuation in a long list do not become one span. A group needs a STREET, BUILDINGNUM, or ZIPCODE to count. A city on its own is common in ordinary text, rarely identifying, and masking it degrades replies.
- GOV_ID collects IDCARDNUM, PASSPORTNUM, DRIVERLICENSENUM, and TAXNUM. They are identifying, but optional like SSN and CREDIT_CARD, so a deployment can turn them off.
- Whitespace at either edge of an entity is trimmed before grouping. In the English validation split, 878 TELEPHONENUM spans end in a space, which would otherwise disagree with the hand-labeling rule that spans carry no surrounding whitespace.
- Effect of widening the gap rules, measured with `edgeveil data prepare` on 2026-10-06. ADDRESS spans went from 101,146 to 94,972 in train and from 25,130 to 23,562 in validation, because pieces that used to be split now merge. PERSON spans did not change (161,229 train, 40,368 validation): no GIVENNAME and SURNAME pair in the data is separated by an initial or particle. After the change, 2,566 of 31,254 adjacent name-part pairs and 9,581 of 39,239 adjacent address-part pairs in validation still split. The most common address splits are words such as ` in ` and ZIP-code phrases such as ` (ZIP `. Most name splits are separate people joined by ` and `, commas, or form labels such as `Last Name: `.
- DATE, AGE, GENDER, SEX, and TITLE map to O. They are quasi-identifiers, outside the v1 entity set.

## Name connectors

Widening the PERSON gap rule changed no training span, because no OpenPII row has an initial or particle between a given name and a surname. The test set will have them, so `edgeveil data augment` adds them. `prepare` records each whitespace gap between a GIVENNAME and the SURNAME that follows it (`name_joints` in the clean parquet), and `augment` replaces about 3% of those gaps (`NAME_CONNECTOR_PAIR_FRACTION`) with `gap + connector + gap`, half capital initials and half listed particles. The surrounding PERSON span grows by the inserted length and later spans shift. This runs before credential insertion, because the joint offsets refer to the clean text. On the 2026-10-06 run, 3,443 of 114,781 train pairs and 860 of 28,651 validation pairs got a connector.

## Gap report

`edgeveil data gaps` lists the gaps that keep adjacent name or address parts from merging. Gaps longer than `MAX_ADDRESS_GAP_LENGTH` (16) are shown as one bucket. For addresses that is the merge limit, so any longer gap splits by length alone. Person gaps have no length limit, and the same bucket is used there only to keep long stretches of text out of the report.

## Whitespace characters

U+00A0 (no-break space) and U+202F (narrow no-break space) count as whitespace everywhere: `str.isspace` and `str.strip` treat them as whitespace, and `\s` in Python's `re` matches them in str patterns. OpenPII uses U+202F between address parts (for example 243 CITY to ZIPCODE gaps in validation). Tests pin this for gap matching and edge trimming. The validators do not exist yet; their tests will cover the same characters.

## Evaluation statistics

Per-type precision, recall, and F1 each get a 95% bootstrap confidence interval. Resampling is over test snippets, not spans, because spans in one snippet are not independent. The seed and the number of resamples are named constants, so a rerun reproduces the same interval. With about 200 snippets and 40 instances per type, a single F1 number would hide a wide interval.

## Synthetic credentials

OpenPII has no password, key, or token labels, so CREDENTIAL training data is synthetic.

- A seeded generator in the package writes only under `data/`. It inserts fake passwords, API keys, and tokens into OpenPII English sentences and a few templated contexts, in about 3 to 5% of training rows.
- Hard negatives labeled O (commit SHAs, UUIDs, checksums, base64 blobs) teach the model that high-entropy text alone is not a secret.
- Provider key formats are assembled at runtime from parts, and no literal in a real provider format is stored in the repo, because GitHub secret scanning flags such strings.
- Rates, set as constants in `edgeveil.dataset.credentials`: `CREDENTIAL_ROW_FRACTION = 0.04` of rows end up holding a credential. A quarter of those (`STANDALONE_CONTEXT_SHARE`) are standalone templated rows (a `.env` file, a curl command, JSON and YAML config, client code), and the rest are OpenPII rows with a credential sentence inserted. `HARD_NEGATIVE_ROW_FRACTION = 0.02` of OpenPII rows get a hard-negative sentence. On the 2026-10-06 run this gave 3.96% of rows with a CREDENTIAL span in each split, 7,054 CREDENTIAL spans in train and 1,760 in validation.
- Validation is augmented too, with its own random stream (`seed:split`). Per-type thresholds are tuned on validation, and without credentials there the CREDENTIAL threshold could not be tuned. The test set stays hand-written, so the generator never touches it.
- Train and validation draw from disjoint template sets (`TRAIN_TEMPLATES`, `VALIDATION_TEMPLATES`) for credential sentences, standalone contexts, and hard negatives, so validation measures generalization to contexts the model has not seen. The secret generators and key shapes are shared, so validation does not test unseen key formats. CREDENTIAL validation scores are still scores on synthetic data. Only the hand-written test set measures real-looking credentials.
- Insertions go at the start, the end, or after a sentence boundary (`.`, `!`, or `?` followed by whitespace, or a newline), never inside an existing span. Spans after the insertion point shift by the inserted length.
- A CREDENTIAL span covers only the secret. Field names, quotes, and the `Bearer` scheme word stay outside, matching `docs/labeling.md`.
- `edgeveil data augment` refuses an output path with no `data` directory in it. The `data/` pattern in `.gitignore` matches a directory of that name at any depth, so this keeps generated provider-shaped keys out of anything that could be committed.
- The test set's credentials are written by hand. If the generator also produced the test credentials, the test would measure how well the model learned the generator rather than real secrets.
