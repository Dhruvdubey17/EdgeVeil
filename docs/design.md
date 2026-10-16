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
- Within a window, decoded spans of the same type with zero characters between them join into one span before the threshold check, so the type threshold sees the whole value (`rohan.` and `iyer@example.com` become one EMAIL). Until 2026-10-16 a B tag after a touching span of the same type started a new span.
- Spans from overlapping windows merge by character offsets. Overlapping or touching spans of the same type join into one that covers both and keeps the higher score. Where spans of different types overlap, the higher-scoring span wins the overlap and the rest of the other span stays masked under its own type, snapped to whole tokens. Until 2026-10-16 the lower-scoring span was dropped whole, which unmasked its characters outside the overlap. Spans of different types that only touch stay separate.
- `edgeveil tune-thresholds` sweeps one threshold for all types over `THRESHOLD_GRID` (0.05 to 0.95 in steps of 0.05) on validation, scores the emitted spans per type, and picks the threshold with the highest F-beta, `THRESHOLD_BETA = 2.0`, per type. With beta 2, recall counts twice as much as precision (beta squared, 4, weights precision in the denominator), because a missed secret costs more than an extra placeholder. Ties go to the lower threshold. Sweeping one value for all types is equivalent to sweeping each type on its own: which tokens form a span, and of which type, does not depend on any threshold, so a type's threshold only decides whether that type's spans are kept. One exception since 2026-10-16: where two windows disagree on the type of an overlap, whether the other type's span survives its threshold decides how much of this type's span it takes, so the equivalence holds everywhere except at such window overlaps.
- Scoring uses the final spans the product emits, after thresholds and merging, re-encoded as IOB2 over each text's tokens and scored with strict seqeval. The test-set evaluation will use the same function.
- Results go to `bench/results/thresholds/<preset>-fp32-torch.json` with the chosen threshold, precision, recall, F2, and support per type, the full curve, the model and dataset revisions, and the chip, OS, and library versions.

## First test-set evaluation and what it exposed

The first run of `edgeveil eval` on the hand-labeled test set (235 snippets, 2026-10-15) scored a strict overall F1 of 0.561 against 0.985 on validation, with EMAIL at 0.000. Results: `bench/results/accuracy/baseline/distilbert-fp32-torch-2026-10-15.json`.

- Emitted emails were fragmented (`k.obrien@` and `com` as two spans), so strict matching scored none of them. Every test email is on `example.com` or `example.org`, as the synthetic-data rule requires. None of the 91,238 training emails used those domains, and `example.com`/`example.org` appeared 413 times in training text labeled O, mostly in URLs from the credential templates (`https://api.example.com/...`). The model learned that those domain tokens are not PII.
- Thresholds were applied per token. With EMAIL at 0.80 and CREDENTIAL at 0.95, one inner token below the threshold cut a span in two (`P@ssw0rd-Linden` from a longer password).
- The test set exposed both problems, so the evaluation after the fixes is not fully independent of the test set. The fixes are tuned on validation only, and no threshold or rule is chosen by looking at test-set scores.

## Second test-set evaluation

After both fixes (span-level thresholds and email domain rewriting), the model was retrained on a Kaggle T4 (10,300 steps, 3,642 s), thresholds were retuned on validation, and the same 235-snippet test set was scored once (2026-10-16). Strict overall F1 went from 0.561 to 0.702 [0.652, 0.748], EMAIL from 0.000 to 0.805 [0.689, 0.918], and PERSON from 0.805 to 0.896. CREDIT_CARD recall stayed at 0.293 with precision 1.000, PHONE and ADDRESS precision stayed low (0.516 and 0.474), and CREDENTIAL recall was 0.500. The two fixes were applied together, so this run does not separate their effects. Results: `bench/results/accuracy/distilbert-fp32-torch.json`, table in `bench/results/tables/`.

## Email domain rewriting

`edgeveil data augment` rewrites the domain of `EMAIL_DOMAIN_REWRITE_SHARE` (25%) of email spans in both splits, after the name connectors and before credential insertion. Half the new domains are `example.com`, `example.org`, or `example.net`; the rest are generated names under `.test` (`mail.harbor.test`, `copper-orbit.test`). Every one is a reserved name, so the rewrite cannot produce a real address. The credential template URL moved from `api.example.com` to `api.orders.invalid`. On the 2026-10-15 rebuild, 23,161 of 91,238 train emails and 5,762 of 22,794 validation emails were rewritten, and `example.com`/`example.org` outside email spans fell from 413 to 136 occurrences in train, all from OpenPII's own text. The change came from the first test-set evaluation (see above).

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

## The dev-test set

From 2026-10-16 the 235 hand-labeled snippets in `data/test/test_set.jsonl` are called the dev-test set. They informed fixes: the first evaluation led to email domain rewriting and span-level thresholds, and the error analysis below shaped the validators, the combiner rules, the decoding change for touching spans, the never-unmask merge across windows, the threshold retune that followed it, and the label corrections. Scores on the dev-test set are therefore optimistic and are not the number the README reports. Before publication, about 100 fresh snippets are written under the same rules in `docs/labeling.md`, labeled, and scored once, and the README reports that run.

## Frozen v1 checkpoint

The DistilBERT model trained on Kaggle on 2026-10-16 is v1, frozen for milestones 2 and 3: no retraining and no ModernBERT run until then. Every export, quantized build, and backend comparison starts from this file. `models/distilbert/model.safetensors` has SHA-256 `c0eafef1bf82eec0c88765986160ff0874b2a74812e717fb4918a5549b502c06`, and every accuracy result file records `model_sha256`, so a result from another checkpoint is visible in its file. The thresholds stay at `bench/results/thresholds/distilbert-fp32-torch.json`.

## Error analysis

`edgeveil eval` classifies every strict error. Gold and emitted spans are first snapped to the token grid used for strict scoring, so an exact match here is a strict true positive. A gold span with no exact match is a miss, and an emitted span with no exact match is a false positive. Each falls in one group:

- Boundary: it overlaps a span of the same type on the other side. One boundary error counts once as a miss and once as a false positive, the way strict scoring counts it.
- Wrong type: it overlaps only spans of other types.
- Missed (for a miss) or spurious (for a false positive): it overlaps nothing.

Same-type overlap takes precedence over other-type overlap. Counts per type go into each accuracy result file under `errors` and into the rendered tables. `edgeveil eval --show-errors model` prints every error with 30 characters of context; that text goes to the terminal only.

On the dev-test set (model only, 2026-10-16) the 102 misses split into 25 missed, 31 boundary, and 46 wrong type, and the 131 false positives into 34 spurious, 45 boundary, and 52 wrong type.

- CREDIT_CARD: 27 of 29 misses are wrong type. The model tags processor test numbers as PHONE (21) or GOV_ID (6), so the digits are masked (char recall 0.912) under the wrong placeholder. Those same spans are 27 of the 30 PHONE false positives.
- PHONE: 7 of 10 misses are wrong type. Bare `555-01xx` numbers are tagged SSN or ADDRESS, often as a fragment such as `01` or `55`.
- ADDRESS: 17 spurious false positives on short numbers (ports, PINs, room numbers, order numbers, security codes) and 16 boundary false positives where one address splits at unit words or region codes (`Suite 300`, `Room 412`, `ZZ`).
- CREDENTIAL: 11 missed entirely, mostly human-chosen passwords and passphrases (dictionary words with digits or symbols) and one 32-character hex token. The synthetic training credentials are mostly machine-generated keys.
- GOV_ID: 7 wrong-type misses, mostly tax and national insurance numbers tagged SSN.
- EMAIL: 5 boundary misses, each splitting the local part at a period.

The listing also shows gold spans that break `docs/labeling.md`: two addresses labeled EMAIL (snippets 103 and 144), titles inside PERSON (113, 231), a comma-joined name (64), a name split at `d/o` (133), and a partial card number (87). They were later fixed through the label lint and corrections (see Label corrections below); the counts in this section are from before the fix.

## Secondary metrics

Strict entity F1 stays the headline. Two secondary measures sit next to it in `edgeveil eval` and the tables, each with a bootstrap interval over snippets using the same seed and resample count:

- Overlap precision and recall: an emitted span matches if it overlaps a gold span of the same type, and the reverse for recall. This separates "found, wrong edges" from "not found".
- Char recall: the share of gold span characters covered by any emitted span, of any type. A card number masked as PHONE still counts, because the value does not leave the device. This is the measure closest to leak rate.

On the dev-test set, model only, overall strict recall is 0.729 [0.679, 0.776], overlap recall 0.812 [0.768, 0.850], and char recall 0.911 [0.882, 0.937].

## Validators and the combiner

Pulled forward from week 3. `edgeveil.validators.finders` has four pure functions that return spans with source `validator` and score 1.0:

- Card numbers: digit runs with spaces, dashes, or no separators. Every contiguous run of digit groups with one separator style and 13 to 19 digits is tested with Luhn, and the longest passing run is kept, so a trailing expiry or a following number does not break the match.
- Email: a strict pattern, `local@domain` with dot-separated parts and a 2 to 63 letter top-level label. Consecutive dots fail.
- Phone: a fixed set of shapes (NANP with optional `+1` and parentheses, a 7-digit local number, `+` country code with groups, and national numbers with a leading 0), 7 to 15 digits. SSN, date, version, and price shapes do not match.
- Credentials: runs of 20 or more key characters with upper case, lower case, and digits, at least 3.5 bits of entropy per character, not hex only (commit SHAs, UUIDs, checksums), and not a path.

Validator spans that overlap each other are resolved by priority: email, card, phone, credential. `edgeveil.validators.combiner.combine_spans` holds every rule for merging model and validator spans. Two requirements come first:

- Never unmask. Every model span is emitted above its type threshold, and the combiner never removes the mask from any of its characters. A format check that fails (a veto) can only let a validator change the type of those characters. The spec puts recall first, so this is a requirement, not a choice tuned on the dev-test set. A property test checks that every non-whitespace character a model span covers is still covered after combining, and that the combiner covers nothing that neither side emitted.
- An accepted validator span owns its characters. Model spans that overlap it keep the parts outside it, trimmed of whitespace, under the model's type. A card inside a model ADDRESS span therefore splits it into ADDRESS, CREDIT_CARD, ADDRESS.

Format checks on model spans (the vetoes):

| Model type | Format check |
|---|---|
| CREDIT_CARD | Luhn-valid, 13 to 19 digits |
| PHONE | 7 to 15 digits |
| SSN | only digits, spaces, and dashes, with 9 digits (full) or 4 (partial) |
| EMAIL | matches the strict email pattern |
| PERSON, ADDRESS, CREDENTIAL, GOV_ID | none, always passes |

Precedence when a validator span overlaps model spans:

| Validator | Overlapping model span | Result |
|---|---|---|
| Card (Luhn) or email | none | validator span added |
| Card (Luhn) or email | any type | validator type and edges win; model parts outside stay masked |
| Phone or entropy | none | validator span added |
| Phone or entropy | same type | validator edges win; model parts outside stay masked |
| Phone or entropy | other type that fails its format check | validator relabels the overlap; model parts outside stay masked |
| Phone or entropy | other type that passes its format check | validator span dropped; model span unchanged |

A phone or entropy span is accepted only if every model span it overlaps allows it. A model span that fails its check with no validator over it keeps its type and its mask.

The validators were written after the error analysis above, and the precedence rules after the first ablation, so validator and combined scores on the dev-test set are optimistic like the model scores.

The first ablation (2026-10-16, before the never-unmask rules and the label corrections) gave overall strict F1 of 0.702 for the model, 0.519 for validators, and 0.810 for both. Its combiner dropped model spans that failed a format check, which unmasked one GOV_ID fragment, and let the phone validator replace two correct GOV_ID spans. Both are gone under the current rules. Current numbers are in the rerun section below.

## Label corrections

Gold labels are never edited by hand. `edgeveil data import-labels` converts the untouched Label Studio exports (copies in `data/test/exports/`), applies `data/test/corrections.jsonl`, and then runs the label lint described in `docs/labeling.md`. It writes nothing if a correction no longer matches or a check fails. The lint is derived only from the labeling rules, never from model output, and stays in the import so the fresh held-out set is checked the same way before it is scored.

The lint ran over all 235 dev-test snippets on 2026-10-16 and the user approved every change:

- Nine mechanical fixes: a scheme word (`Basic`) and two ID-type words (`DL`, `CDL`) moved out of spans (snippets 19, 131, 146), titles moved out of PERSON (113, 231), a comma-joined name split (64), a name split at `d/o` joined (133), a partial card number unlabeled (87), and parentheses around a whole phone number moved out (204).
- Four reviewer decisions on checks the rules cannot settle: two addresses labeled EMAIL relabeled ADDRESS over the whole span (103, 144) and two lone cities unlabeled (137, 154).
- One one-off reviewer decision outside the lint: the trailing `&` (a URL query separator) trimmed from a token (31). It is not a lint rule, because real passwords can end in punctuation.

The dev-test set went from 377 to 374 spans: CREDIT_CARD 41 to 40 and EMAIL 45 to 43; ADDRESS stays 47 (two added, two removed) and PERSON stays 75 (one split, one join). The error analysis prompted these corrections, so they are informed by the dev-test set.

## Threshold retune after the decoding change

Joining touching spans changes which spans reach the threshold check, so thresholds were retuned on validation on 2026-10-16 with the same grid and F2, keeping the v1 checkpoint. Every type kept its threshold:

| Type | Old | New | Validation recall, old to new |
|---|---|---|---|
| PERSON | 0.80 | 0.80 | 0.994 to 0.993 |
| EMAIL | 0.95 | 0.95 | 0.999 to 0.999 |
| PHONE | 0.75 | 0.75 | 0.996 to 0.996 |
| ADDRESS | 0.85 | 0.85 | 0.986 to 0.985 |
| CREDENTIAL | 0.95 | 0.95 | 0.831 to 0.833 |
| SSN | 0.55 | 0.55 | 0.971 to 0.971 |
| CREDIT_CARD | 0.65 | 0.65 | 0.998 to 0.998 |
| GOV_ID | 0.55 | 0.55 | 0.979 to 0.979 |

The old file is kept at `bench/results/thresholds/baseline/distilbert-fp32-torch-2026-10-16.json`. Every accuracy result file records `thresholds_sha256` next to `model_sha256`; the current thresholds file has SHA-256 `6d2804910f9e2c0c5c802aa33bc5da1f5f055924a2698bca31a4a5e27348a39a`. The decoding change and the retune were prompted by the dev-test error analysis, so they are informed by the dev-test set, although the retune itself used validation only.

## Rerun after corrections (2026-10-16)

v1 with the retuned thresholds, current decoding and combiner, on the corrected dev-test set (235 snippets, 374 spans). Every change in this rerun was informed by the dev-test set, so these scores are optimistic.

Ablation, strict F1 [95% CI]:

| Type | Model only | Validators only | Both |
|---|---|---|---|
| PERSON | 0.948 [0.904, 0.984] | 0.000 | 0.961 [0.925, 0.987] |
| EMAIL | 0.881 [0.773, 0.971] | 0.977 [0.940, 1.000] | 0.977 [0.940, 1.000] |
| PHONE | 0.635 [0.506, 0.736] | 0.941 [0.875, 0.988] | 0.897 [0.809, 0.966] |
| ADDRESS | 0.608 [0.481, 0.732] | 0.000 | 0.608 [0.481, 0.732] |
| CREDENTIAL | 0.588 [0.441, 0.727] | 0.509 [0.333, 0.654] | 0.657 [0.529, 0.763] |
| SSN | 0.812 [0.718, 0.886] | 0.000 | 0.845 [0.757, 0.915] |
| CREDIT_CARD | 0.462 [0.267, 0.618] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 0.659 [0.526, 0.769] | 0.000 | 0.714 [0.571, 0.828] |
| overall | 0.729 [0.682, 0.772] | 0.530 [0.481, 0.580] | 0.836 [0.801, 0.869] |

Overall recall is 0.759 for the model, 0.366 for validators, and 0.877 for both. Overall char recall is 0.918 for the model and 0.950 for both, and no type's char recall is lower with both than with the model alone.

Model only, strict errors:

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 1 | 0 | 5 | 1 | 0 |
| EMAIL | 3 | 3 | 0 | 0 | 4 | 0 |
| PHONE | 1 | 1 | 7 | 0 | 2 | 27 |
| ADDRESS | 1 | 8 | 0 | 18 | 18 | 4 |
| CREDENTIAL | 11 | 6 | 3 | 2 | 6 | 0 |
| SSN | 2 | 1 | 0 | 5 | 1 | 10 |
| CREDIT_CARD | 2 | 0 | 26 | 0 | 0 | 0 |
| GOV_ID | 2 | 4 | 7 | 6 | 5 | 7 |
| overall | 23 | 24 | 43 | 36 | 37 | 48 |

Model and validators, strict errors:

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 0 | 0 | 5 | 0 | 0 |
| EMAIL | 0 | 0 | 0 | 0 | 0 | 2 |
| PHONE | 0 | 2 | 1 | 0 | 2 | 4 |
| ADDRESS | 1 | 8 | 0 | 18 | 18 | 4 |
| CREDENTIAL | 10 | 2 | 5 | 5 | 2 | 0 |
| SSN | 2 | 1 | 0 | 5 | 1 | 6 |
| CREDIT_CARD | 0 | 0 | 0 | 0 | 0 | 0 |
| GOV_ID | 2 | 4 | 7 | 6 | 5 | 0 |
| overall | 16 | 17 | 13 | 39 | 28 | 16 |

- Model only moved from 0.702 to 0.729 overall. Scoring the current code on the uncorrected labels gives 0.708, so the decoding change accounts for about 0.006 (EMAIL 0.805 to 0.861, from local parts no longer split at a period) and the label corrections for the rest.
- The never-unmask rules cost strict precision in a few places, all still masked. The phone validator cannot relabel a model ADDRESS fragment (`55` in `555-0119`, snippets 207 and 214) because ADDRESS has no format check, so the number ends up as an ADDRESS fragment plus a PHONE piece. Model CREDENTIAL characters outside a validator token stay masked as their own spans (`-u ci-bot -p`, a trailing `&`), which adds spurious CREDENTIAL false positives.
- The entropy validator flags a base64 PNG in a data URI (snippet 181), a hard negative.
- PHONE recall with both (0.929) is below validators alone (0.952) for the same reason: a model ADDRESS span over part of a phone number blocks the phone validator there (snippets 207 and 214, and 231, where the ADDRESS span starts at the phone number).
- Full tables: `bench/results/tables/ablation-distilbert-fp32-torch.md` and the three `accuracy-*.md` files.

## Tokenizer without transformers

Every backend loads `tokenizer.json` through `tokenizers` (0.23.2) and windows text with the same code, so the ONNX Runtime and Core ML paths import neither torch nor transformers (a test checks `sys.modules`). Training still uses the transformers tokenizer object and passes its `backend_tokenizer` to the same windowing. `tests/test_tokenizer_parity.py` encodes all 40,738 validation rows both ways, through the transformers call the training run used and through `tokenizers` alone, and finds no row whose window ids or offsets differ. The test skips when the local model or data is absent, as in CI.

## ONNX export

- Exporter: `torch.onnx.export` with `dynamo=True`, opset 18, dynamic batch and sequence axes (sequence at most 512), weights in one 261.8 MB file. Optimum was the first choice; optimum-onnx 0.1.0 requires transformers below 4.58 and the lock has 5.19.
- The model is wrapped to return only the logits tensor, with inputs `input_ids` and `attention_mask` (DistilBERT has no token type ids).
- Fidelity: on 2,000 validation windows (192,275 tokens), the fp32 ONNX build and fp32 torch on CPU pick the same label for every token, with a largest logit difference of 4e-5. Dev-test scores are identical for every type.

## Quantized ONNX builds

| Precision | Method | Size (MB) |
|---|---|---|
| fp32 | export | 261.8 |
| fp16 | `onnxruntime.transformers.float16.convert_float_to_float16`, FP32 inputs and outputs kept | 131.3 |
| int8-dynamic | `quantize_dynamic`, INT8 weights | 66.2 |
| int8-static | `quantize_static`, QDQ, per-channel INT8 weights, INT8 activations, MinMax over 300 validation windows | 66.5 |
| int4 | `MatMulNBitsQuantizer`, 4 bits, block 32, symmetric | 118.3 |

- Calibration and parity windows come from one seeded shuffle of validation (seed 20261016). Rows go to calibration until it has 300 windows, then to parity until it has 2,000, so no row feeds both.
- `quant_pre_process` fails on the dynamo graph in its symbolic shape inference ("Incomplete symbolic shape inference"). On failure, onnxruntime 1.30 also writes `sym_shape_infer_temp.onnx` and a 260 MB external data file into the working directory (`onnxruntime/tools/symbolic_shape_infer.py`, line 3059). Both INT8 builds run it with `skip_symbolic_shape=True`, keeping ONNX shape inference and graph optimization. The dynamo graph already carries shapes.
- MinMax calibration adds a ReduceMin and ReduceMax output per tensor, so it keeps scalars, not activations, and 300 windows fit in memory without batching. Its `CalibMaxIntermediateOutputs` option is not used: in onnxruntime 1.30, `MinMaxCalibrater.collect_data` clears the collected outputs when the limit is hit without computing their ranges first.
- All three integer builds quantize the 37 weight MatMuls (six per layer plus the classifier). In int8-dynamic and int4 the 12 attention MatMuls, which multiply two activations, stay in floating point; int8-static puts QDQ pairs on those as well. Both INT8 builds also quantize the 89 MB token embedding to 22 MB. INT4 leaves it in FP32, which is why it is larger than either INT8 build.
- Per-channel quantization warns for LayerNorm weights (rank 1, no axis 1); those fall back to per-tensor.
- All builds are scored with the fp32 torch thresholds (`bench/results/thresholds/distilbert-fp32-torch.json`, SHA-256 6d28049...a39a). They are not retuned per precision, so a score change is the build's effect alone.

## ONNX parity and dev-test accuracy

Token agreement with fp32 torch on CPU, 2,000 validation windows, 192,275 tokens, 59,252 of them labeled as an entity by the reference. Files: `bench/results/parity/distilbert-*-onnxruntime.json`.

| Precision | Token agreement | Entity token agreement | Max logit difference | Mean logit difference |
|---|---|---|---|---|
| fp32 | 1.0000 | 1.0000 | 0.00004 | 0.000001 |
| fp16 | 1.0000 | 1.0000 | 0.0134 | 0.0004 |
| int8-dynamic | 0.9993 | 0.9979 | 3.80 | 0.130 |
| int8-static | 0.9960 | 0.9886 | 10.85 | 0.430 |
| int4 | 0.9996 | 0.9989 | 3.36 | 0.111 |

Dev-test strict F1, overall (235 snippets; this set informed earlier fixes, so scores are optimistic). Files: `bench/results/accuracy/distilbert-*-onnxruntime*.json`.

| Precision | Model only (95% CI) | Model and validators |
|---|---|---|
| torch fp32 | 0.729 [0.682, 0.772] | 0.836 |
| fp32 | 0.729 [0.682, 0.772] | 0.836 |
| fp16 | 0.728 [0.681, 0.771] | 0.836 |
| int8-dynamic | 0.716 [0.662, 0.762] | 0.827 |
| int8-static | 0.619 [0.568, 0.673] | 0.780 |
| int4 | 0.723 [0.675, 0.766] | 0.825 |

- fp16 emits one more PHONE span than fp32, with the wrong boundary, over a number fp32 typed as something else (PHONE precision 0.532 to 0.524, recall unchanged).
- int8-static is the one build whose interval does not overlap fp32's. Model-only EMAIL F1 falls from 0.881 to 0.560, PERSON from 0.948 to 0.861, PHONE from 0.635 to 0.447. Token agreement is still 99.6%, but 674 of its 778 disagreeing tokens are entity tokens, and strict matching fails a whole span when one of its tokens flips. With validators the EMAIL loss disappears (0.977) because the email validator sets those boundaries, and overall falls 0.836 to 0.780. The cause has not been isolated: this build puts QDQ pairs in front of every operator the quantizer supports (all 49 MatMuls, LayerNormalization, Softmax, Add, and the Mul and Div inside GELU), with per-tensor MinMax activation ranges. Per the milestone plan, it is reported as a finding and not tuned.
- int8-dynamic and int4 stay within 0.015 of fp32 overall.

## Core ML conversion

- Path: `torch.jit.trace` of the logits wrapper at (1, 512), then `coremltools.convert` to an ML Program with FP16 compute, minimum target macOS 15, and `EnumeratedShapes` of (1, 128), (1, 256), (1, 512) for both int32 inputs, default (1, 512). The backend pads each window to the smallest enumerated length that fits, with attention mask 0 on the padding.
- coremltools 9.0 has no converter for `Tensor.new_ones`, which transformers 5.19 calls in `masking_utils.and_masks` while building the bidirectional mask, so the first conversion stopped with "PyTorch convert function for op 'new_ones' not implemented". coremltools 9.0 lists torch 2.7 as its newest tested version; this project runs torch 2.14.1. `edgeveil.export.coreml_ops` registers a converter through coremltools' `register_torch_op`, the same pattern as its own `new_zeros` and `new_full`: a `fill` of ones whose dtype is the source tensor's unless one is passed, with integer types mapped to int32 as coremltools' `full` does. It registers only when coremltools has no `new_ones` of its own. No other operator was missing.
- The trace raises two TracerWarnings, both on shape arithmetic that is constant under enumerated shapes: the mask padding length in `masking_utils` (mask length always equals the input length) and `is_causal` in the SDPA integration (a mask is always passed, so it is False). The mask itself stays a traced input. A quick check before the full parity run: on 100 validation windows of 26 to 506 tokens, padded to all three lengths, the FP16 package agreed with torch on 99.99% of tokens on CPU_ONLY. A dropped mask would have shown up on the padded windows.
- Weight quantization uses `coremltools.optimize.coreml.linear_quantize_weights` on the FP16 package: INT8 symmetric per channel, and INT4 symmetric per block of 32 along the input axis. With the default `weight_threshold` of 2,048 elements, the token embedding is quantized too, so Core ML INT4 is 36.9 MB, against 118.3 MB for ONNX INT4, which leaves the embedding in FP32. FP16 is 130.5 MB and INT8 65.5 MB.

## Core ML parity and dev-test accuracy

Same 2,000 validation windows and reference as the ONNX parity. Files: `bench/results/parity/distilbert-*-coreml-*.json`, `bench/results/accuracy/distilbert-*-coreml-*.json`.

| Precision | Compute units | Token agreement | Entity token agreement | Max logit difference | F1, model (95% CI) | F1, model and validators |
|---|---|---|---|---|---|---|
| fp16 | all | 1.0000 | 1.0000 | 0.021 | 0.728 [0.681, 0.771] | 0.836 |
| fp16 | cpu-only | 1.0000 | 0.9999 | 0.226 | 0.729 [0.681, 0.773] | 0.837 |
| int8 | all | 1.0000 | 0.9999 | 0.434 | 0.725 [0.674, 0.768] | 0.831 |
| int8 | cpu-only | 0.9999 | 0.9999 | 0.376 | 0.725 [0.674, 0.768] | 0.831 |
| int4 | all | 0.9994 | 0.9987 | 2.632 | 0.696 [0.643, 0.744] | 0.812 |
| int4 | cpu-only | 0.9994 | 0.9987 | 2.567 | 0.696 [0.643, 0.744] | 0.812 |

- The parity run was the check that the `new_ones` converter is correct: every build agrees with fp32 torch on at least 99.94% of tokens, padded to all three enumerated lengths.
- Core ML int4 loses 0.033 overall F1 against fp32 (ONNX int4 loses 0.006). The two use the same block size of 32, but Core ML also quantizes the embedding and computes in FP16. Its interval still overlaps fp32's.

## Latency on the M3

Machine: Mac15,13 (MacBook Air, Apple M3, 8 GiB, no fan), macOS 26.1, other apps closed. One window at batch size 1, model call only, 20 warmup and 200 timed calls per length, 60 s idle before each build. `pmset -g therm` recorded no thermal or performance warning before or after any run. Files: `bench/results/latency/`. p50 / p95 in ms:

| Build | 128 tokens | 256 tokens | 512 tokens |
|---|---|---|---|
| onnxruntime fp32 | 30.70 / 31.84 | 67.67 / 70.00 | 156.36 / 162.63 |
| onnxruntime fp16 | 33.25 / 35.66 | 72.84 / 76.91 | 175.52 / 184.01 |
| onnxruntime int8-dynamic | 11.82 / 12.31 | 27.62 / 29.11 | 75.99 / 78.85 |
| onnxruntime int8-static | 14.25 / 14.78 | 32.79 / 33.69 | 87.07 / 89.85 |
| onnxruntime int4 | 33.87 / 35.71 | 77.38 / 86.86 | 184.97 / 228.17 |
| coreml fp16, all | 27.05 / 27.36 | 53.04 / 54.59 | 117.83 / 120.27 |
| coreml fp16, cpu-only | 5.82 / 6.10 | 11.40 / 11.68 | 25.70 / 25.94 |
| coreml int8, all | 27.16 / 27.49 | 52.72 / 53.29 | 117.63 / 118.18 |
| coreml int8, cpu-only | 6.05 / 6.24 | 11.37 / 11.48 | 25.69 / 25.97 |
| coreml int4, all | 5.26 / 5.42 | 9.37 / 9.50 | 18.13 / 18.31 |
| coreml int4, cpu-only | 6.03 / 6.19 | 11.37 / 11.55 | 25.69 / 25.92 |

- ONNX Runtime fp16 on the CPU is 12% slower than fp32 at 512 tokens (175.5 against 156.4 ms), as the plan expected. The converted graph adds Cast nodes to keep FP32 inputs and outputs; where the rest of the time goes was not profiled. Not optimized, per the plan.
- ONNX int4 is slower than fp32 (185 against 156 ms at 512) and has the widest p95 (228 ms). It was built with the default `accuracy_level`, so activations stay FP32; the INT8-compute setting (level 4) was not tried. The INT8 builds are the fastest ONNX builds, about 2x fp32.
- Every Core ML cpu-only build runs at about 25.7 ms at 512 tokens, whatever the weight precision, against 156 ms for ONNX Runtime fp32 on the same CPU.
- The latency target is under 50 ms p95 per 512-token window on the Neural Engine. No build ran on the Neural Engine (next section). The fastest build, Core ML int4 with compute units all, ran on the GPU at 18.31 ms p95.

## Neural Engine placement

`edgeveil bench placement` reads the preferred device of each operation from coremltools 9.0's `MLComputePlan` (files: `bench/results/placement/`). For every build and both settings, 0 of 230 to 235 operations prefer the Neural Engine. With compute units all, fp16 and int8 place every operation on the CPU, and int4 places every operation on the GPU. Between 317 and 321 further operations report no device usage and are left out of the shares.

A one-off diagnostic in the session scratchpad, not a results file, loaded the fp16 package with all four compute-unit settings and timed 50 calls (median, ms):

| Compute units | Preferred device per plan | 128 tokens | 512 tokens |
|---|---|---|---|
| CPU_ONLY | cpu, all 231 | 6.17 | 25.70 |
| CPU_AND_GPU | gpu, all 231 | 5.61 | 18.74 |
| CPU_AND_NE | cpu, 230 | 26.90 | 117.06 |
| ALL | cpu, 230 | 27.04 | 117.21 |

- Under CPU_AND_NE the plan lists the Neural Engine among the supported devices for all 37 `linear` operations, yet it prefers the CPU for every operation. Why has not been established. Candidates include the shape-dependent operations the trace keeps (`shape`, `range_1d`, `tile`, `gather_nd`) and the int32 and bool mask arithmetic (`greater_equal`, `logical_and`, `select`) inside the graph. The Xcode performance report in docs/benchmarks.md is the next check.
- When the Neural Engine is allowed, the CPU plan runs 4.5x slower than CPU_ONLY, and the fp16 build is closer to fp32 torch (max logit difference 0.021 against 0.226). One reading is that the two CPU paths compute at different precision. That has not been verified.
- The spec's Neural Engine claim is not supported by this conversion. Per the plan, this is recorded rather than worked around.
