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

## Test set import

- `edgeveil data import-labels` converts a Label Studio JSON export into `data/test/test_set.jsonl`. Each region must carry exactly one known label, its stored text must match its offsets, and it must not start or end with whitespace. Spans in one task must not overlap, and each task needs exactly one annotation that is not cancelled. Problems are listed by task id and offsets, never by text, and nothing is written until the export is clean.
- Label Studio computes offsets in the browser, where a JavaScript string index counts UTF-16 code units. A character outside the Basic Multilingual Plane (an emoji, for example) counts as two units there and one code point in Python. The converter tries the offsets as code points first, then converts them from UTF-16 and checks the text again.
- Label Studio stores a region's text with each line break as the two characters `\` and `n`, while the offsets count the real newline as one character. The converter accepts stored text that matches the covered text after replacing that escape with a newline. Found on 2026-10-15 in three multi-line address regions.
- The loader lives in `edgeveil.evaluation.test_set`. A test parses every module under `src/edgeveil` and fails if anything outside `evaluation/` imports it, so training code cannot read the test set.
- Pydantic models at these boundaries set `hide_input_in_errors=True`, so a validation error never echoes snippet text.

## Windows and alignment

- Texts are cut into windows of `MAX_WINDOW_TOKENS = 512` tokens that overlap by `WINDOW_OVERLAP_TOKENS = 128`, in `edgeveil.detector.windows`, not by the tokenizer. With tokenizers 0.23.2 and transformers 5.19.0, `return_overflowing_tokens=True` returns at most one overflow window. Reproduced with the real `distilbert/distilbert-base-cased` fast tokenizer and the standard call (`truncation=True, max_length=512, stride=128, return_overflowing_tokens=True`): a text of 6,577 tokens came back as two windows of 512 and 134 tokens, and `overflow_to_sample_mapping` listed sample 0 twice. Calling the Rust backend directly (`backend_tokenizer.enable_truncation(max_length=512, stride=128)`) gave one overflow encoding of 134 tokens. The same happens with a test WordLevel tokenizer, so the behavior is in the library, not in one tokenizer. Relying on it would leave the end of long inputs unscanned at inference. Custom windowing also gives every backend the same window layout, which the parity tests depend on. The text is tokenized once without special tokens, sliced into windows, and each window gets the special tokens the tokenizer adds, learned by encoding a probe string with and without them (transformers 5 removed `build_inputs_with_special_tokens`).
- Token offsets are trimmed of whitespace, because byte-level BPE tokenizers such as ModernBERT's can include a leading space in a token's offsets.
- A token that overlaps a span gets that span's label: B on the first such token, I after. A span boundary inside a token widens to the token. A span cut by the start of a window gets B on its first token in that window. The 128-token overlap gives spans near a boundary a second, uncut view in the neighboring window.
- The decoder treats an I label with no open span of the same type as the start of a new span, which matches seqeval's default mode.

## Training

- Presets in `edgeveil.detector.presets` pin `distilbert/distilbert-base-cased` at `6ea81172465e8b0ad3fddeed32b986cdcdcffcf0` and `answerdotai/ModernBERT-base` at `8949b909ec900327062f0ebf497f51aef5e6f0c8`. Both are Apache-2.0. Hyperparameters (learning rate 5e-5, 2 epochs, batch 32, 6% warmup, weight decay 0.01) are starting values, not tuned.
- Class weights are `(N / (K * n_c)) ** 0.5`, capped at 10, where `n_c` is the count of BIO class `c` among training tokens and `K` is the number of classes. The square root keeps rare classes from dominating the loss, and the cap covers classes with very few tokens.
- Batches: 8 windows per device step with 4 accumulation steps, so the effective batch is 32. The machine is an Apple M3 with 8 GiB of unified memory. A device batch of 32 ran out of MPS memory on its first step (8.99 GiB allocated against a 9.07 GiB limit), because length grouping puts the longest batch first. A device batch of 16 did not crash but thrashed: after 9 minutes no step had finished and swap was at 5.6 of 6 GB. Measured on MPS, fp32, one forward and backward pass plus an AdamW step: 16 x 96 tokens took 276 ms with 2.06 GiB of driver memory, 16 x 256 took 739 ms with 4.07 GiB, and 8 x 512 took 2.5 to 4.4 s with 5.06 GiB. Train windows have a median of 81 tokens, a 99th percentile of 313, and 81 windows at the 512 limit, so 8 x 512 is the worst case and occurs in about ten batches per epoch. `torch_empty_cache_steps=50` returns cached allocator memory every 50 steps so one long batch does not hold 5 GiB for the rest of the run.
- Throughput on the 50-step smoke run: 0.70 optimizer steps per second (22.5 windows per second), and 322 s for a full validation pass over 41,162 windows.
- Padding is dynamic per batch (`DataCollatorForTokenClassification`), and batches are grouped by length (`train_sampling_strategy="group_by_length"` in transformers 5, which replaced `group_by_length=True`). A precomputed `length` column saves the sampler from measuring 165,000 windows at startup. Fixed shapes matter only at export.
- The weighted loss is passed as `compute_loss_func` and is a weighted mean over one micro-batch. By default Trainer assumes such a loss is already normalized over the whole accumulated batch and does not divide it, which with 2 accumulation steps would double the gradient. `build_trainer` sets `trainer.loss_is_scaled_for_ga = False`, so Trainer divides by the accumulation steps.
- A checkpoint is saved at each epoch and the last two are kept. `edgeveil train --resume` continues from the latest one.
- transformers 5.19 removed `warmup_ratio`; `warmup_steps` takes a float in [0, 1) as a share of total steps.
- seqeval runs with `mode="strict"` and `scheme=IOB2` everywhere (`edgeveil.detector.metrics`). An I tag with no matching B is not an entity in strict mode. The product decoder is lenient and treats such a tag as the start of a span, which favors recall. During training the metric sees raw argmax tags, so strict mode penalizes orphan I tags there. At evaluation the decoder's spans are re-encoded as IOB2, which is always well formed.
- seqeval's `classification_report` has no type hints, so pyright reads `zero_division` as `str` and the return as `str`. It is called with the default `zero_division="warn"`, which also scores a class with no predictions as 0.0, with `UndefinedMetricWarning` silenced for that call, and the result is cast to a dict.
- Logits are reduced to argmax label ids before Trainer gathers them (`preprocess_logits_for_metrics`), so evaluating 41,000 validation windows holds integers rather than 17 floats per token.
- Every run writes `training_run.json` next to the model with the preset, base model revision, dataset revision, seed, device, platform, library versions, class weights, training time, and validation metrics.

## Where training runs

Fine-tuning runs on a Kaggle GPU, not on the M3. The full two-epoch run on the M3 (device batch 8, accumulation 4) started on 2026-10-12 and failed. The loss was 2.298 at step 100. At step 200 the gradient norm was NaN, the loss read 0 at steps 300 and 400, and every log after that was NaN. The log showed repeated Metal errors, `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`: GPU command buffers failed for lack of memory, PyTorch kept going on the results of work that never completed, and the weights filled with NaN. The run also slowed to about 0.03 steps per second (1,200 steps in about 10 hours) against 0.70 in the 50-step smoke run, which was too short to show the problem. Training needs about 5 GiB at the worst batch on an 8 GiB machine that also runs the OS and other apps.

- `NonFiniteLossGuard` now stops training with `TrainingDivergedError` the moment a logged loss or gradient norm is NaN or infinite, so a bad run fails in minutes instead of hours.
- Only training moves. Inference, threshold tuning, evaluation, export, and benchmarks run on the M3, and the on-device claims rest on those runs. The training data is public and synthetic, so moving training off the device sends no user data anywhere.
- `scripts/kaggle_train.sh` installs the locked environment with `uv sync --locked`, then replaces the CPU torch build that the lock pins for Linux with the CUDA 12.6 build of the same version (`uv pip install --torch-backend cu126`). torch 2.14.1 has cu126 and cu130 wheels for CPython 3.13 but no cu128 wheel, so the backend is pinned rather than detected. CUDA 12.6 runs on any recent NVIDIA driver through minor-version compatibility.
- The script rebuilds the data on Kaggle (`data download`, `prepare`, `augment`) from the pinned OpenPII revision and the fixed seed instead of uploading it. The prepare and augment reports are copied to the output, so they can be compared with the local reports to confirm the same rows were built.
- The CUDA torch install names the project interpreter (`uv pip install --python .venv/bin/python`). On Kaggle the first run replaced the notebook's own conda torch (2.11.0+cu128) instead of the one in `.venv`, because `uv pip` prefers an active conda environment (`CONDA_PREFIX`) over `.venv`. The project venv kept its CPU build and the CUDA check failed. The script also unsets `PYTHONPATH`, `PYTHONSTARTUP`, `CONDA_PREFIX`, and `VIRTUAL_ENV`, since Kaggle's `PYTHONPATH` loads a `sitecustomize` that imports packages the project venv does not have (`ModuleNotFoundError: No module named 'wrapt'`).
- `CUDA_VISIBLE_DEVICES=0`: on a two-GPU machine, Trainer would wrap the model in DataParallel and multiply the per-device batch by the GPU count, changing the effective batch from 32 to 64.
- `--device-batch-size` sets windows per device step and recomputes accumulation so the effective batch stays 32. The Kaggle default is 16, which at 512 tokens needs roughly twice the 5.06 GiB measured for 8 x 512 on the M3, well within a 16 GB GPU.
- `training_run.json` records the accelerator name (`torch.cuda.get_device_name` on CUDA).

## Thresholds and decoding

- `edgeveil.detector.decoding` is the one function from logits to spans that every backend uses. For each token, the entity mass of a type is P(B-type) + P(I-type). The token takes the type with the highest mass if that mass exceeds P(O); otherwise it is O. B or I follows whichever of the two has more probability. Tokens are decoded into spans, a span's score is the mean entity mass of its tokens, and the span is kept only if that score reaches its type's threshold. Until 2026-10-15 the threshold was applied to each token, which split a span wherever one inner token dipped below it; the test set exposed this (see the first test-set evaluation below).
- Spans from overlapping windows merge by character offsets. Overlapping spans of the same type join into one that covers both and keeps the higher score. Overlapping spans of different types keep the higher-scoring one. Spans that only touch stay separate.
- `edgeveil tune-thresholds` sweeps one threshold for all types over `THRESHOLD_GRID` (0.05 to 0.95 in steps of 0.05) on validation, scores the emitted spans per type, and picks the threshold with the highest F-beta, `THRESHOLD_BETA = 2.0`, per type. With beta 2, recall counts twice as much as precision (beta squared, 4, weights precision in the denominator), because a missed secret costs more than an extra placeholder. Ties go to the lower threshold. Sweeping one value for all types is equivalent to sweeping each type on its own: which tokens form a span, and of which type, does not depend on any threshold, so a type's threshold only decides whether that type's spans are kept.
- Scoring uses the final spans the product emits, after thresholds and merging, re-encoded as IOB2 over each text's tokens and scored with strict seqeval. The test-set evaluation will use the same function.
- Results go to `bench/results/thresholds/<preset>-fp32-torch.json` with the chosen threshold, precision, recall, F2, and support per type, the full curve, the model and dataset revisions, and the chip, OS, and library versions.

## First test-set evaluation and what it exposed

The first run of `edgeveil eval` on the hand-labeled test set (235 snippets, 2026-10-15) scored a strict overall F1 of 0.561 against 0.985 on validation, with EMAIL at 0.000. Results: `bench/results/accuracy/baseline/distilbert-fp32-torch-2026-10-15.json`.

- Emitted emails were fragmented (`k.obrien@` and `com` as two spans), so strict matching scored none of them. Every test email is on `example.com` or `example.org`, as the synthetic-data rule requires. None of the 91,238 training emails used those domains, and `example.com`/`example.org` appeared 413 times in training text labeled O, mostly in URLs from the credential templates (`https://api.example.com/...`). The model learned that those domain tokens are not PII.
- Thresholds were applied per token. With EMAIL at 0.80 and CREDENTIAL at 0.95, one inner token below the threshold cut a span in two (`P@ssw0rd-Linden` from a longer password).
- The test set exposed both problems, so the evaluation after the fixes is not fully independent of the test set. The fixes are tuned on validation only, and no threshold or rule is chosen by looking at test-set scores.

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
