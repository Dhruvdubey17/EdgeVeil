# EdgeVeil

EdgeVeil is a privacy layer that runs on the device, between a user or an app and a cloud LLM. A small fine-tuned token classifier finds personal data in the prompt locally, on the Apple Neural Engine, a Snapdragon NPU, or the CPU. Each value is swapped for a reversible placeholder such as `⟦PERSON_1⟧` before the request leaves the device, and the real values are put back into the reply. The cloud model sees the structure of the conversation but never the names, addresses, or secrets in it.

## How a request flows

```
app ──▶ EdgeVeil proxy (OpenAI-compatible /v1/chat/completions)
          │
          ├─ detect   token classifier on device + rule-based validators
          ├─ combine  one set of non-overlapping spans
          ├─ mask     values → ⟦TYPE_n⟧, mapping kept in an in-memory vault
          │
          ├────────▶ upstream LLM (OpenAI, Anthropic, or Ollama)
          │
          └─ restore  placeholders → values, also across streamed chunks
app ◀──
```

Raw values stay on the device. They never go to the upstream, to logs, to metrics, or to disk.

## What it detects

PERSON, EMAIL, PHONE, ADDRESS, and CREDENTIAL (passwords, API keys, access tokens), plus three optional types: SSN, CREDIT_CARD, and GOV_ID (ID card, passport, driver's license, and tax numbers). Configuration decides which types are on. The span rules for every type are in [docs/labeling.md](docs/labeling.md).

## Architecture

### Detector

- A DistilBERT model fine-tuned for token classification with BIO tags. Long text is cut into 512-token windows with an overlap, and spans from overlapping windows are merged by character offsets.
- Every backend shares one tokenizer, one label map, and one function from logits to spans, so results from different backends are comparable.
- Each entity type has its own threshold, tuned on the validation set with F2 so recall counts more than precision. A missed secret costs more than an extra placeholder.
- Backends: PyTorch, ONNX Runtime on the CPU, Core ML on the Apple Neural Engine, and Qualcomm AI Hub for Snapdragon devices. The Qualcomm backend is used for evaluation and benchmarks only, never by the proxy.

### Validators and the combiner

- Pure functions that find spans with rules: a Luhn check for card numbers, strict patterns for emails and phone numbers, and an entropy check for keys and tokens.
- One combiner merges model spans and validator spans, and all of its rules live in one place. It never removes the mask from text the model flagged. Card and email matches may override the model's type; phone and entropy matches may relabel a model span only when that span fails its own format check, and otherwise they only add spans where the model found nothing.

### Masker, vault, and restorer

- The masker is pure, with no I/O. Within a session, the same normalized value always gets the same placeholder.
- The vault holds the placeholder mapping in memory only, per session, with a time-to-live. It cannot be serialized, and its repr hides the values.
- The restorer tolerates placeholders the LLM has altered: possessive suffixes, ASCII or missing brackets, and changed case. Placeholders it does not recognize pass through unchanged.
- The streaming restorer holds back the shortest end of the buffer that could still turn into a placeholder, and releases it as soon as it cannot. This keeps a placeholder split across two stream chunks from reaching the app half-restored.

### Proxy

- FastAPI, uvicorn, and an httpx async client. It accepts OpenAI-compatible chat completion requests, with and without streaming, and answers in the same format.
- Every upstream implements one adapter interface. OpenAI and Ollama pass through; the Anthropic adapter translates requests and server-sent events in both directions.
- `/health`, and `/metrics` with the latency the proxy adds and entity counts by type, never values.
- Configuration comes from environment variables with the `EDGEVEIL_` prefix and an optional TOML file: enabled types, an allowlist, and per-type thresholds.

### Export and on-device builds

- ONNX export, with FP16, dynamic and static INT8 (QDQ format with a calibration set), and weight-only INT4 builds through ONNX Runtime quantization.
- Core ML in ML Program format, with fixed or enumerated input shapes so the work stays on the Neural Engine, and quantized variants through `coremltools.optimize`.
- Qualcomm AI Hub compiles, profiles, and runs the model on Snapdragon devices.
- Parity tests compare each build with the FP32 model token by token.

## Evaluation and benchmarks

- Accuracy is scored on a hand-labeled test set of short, realistic snippets that contain only synthetic values. Strict entity-level precision, recall, and F1 per type is the headline. Overlap precision and recall and character-level recall (the share of personal-data characters that end up masked) are reported next to it.
- Every per-type score has a bootstrap confidence interval, resampled by snippet with a fixed seed.
- Errors are grouped per type into missed entirely, boundary mismatch, and wrong type.
- A validator ablation compares the model alone, the validators alone, and both.
- The proxy is measured for leak rate (personal data that reaches a mock upstream unmasked), restore accuracy, and added latency.
- Benchmarks run warmup passes, then timed runs, and report the median and 95th percentile per 512-token window. Every result file records the chip, OS, and library versions, and every table in the repo is rendered from those files by `edgeveil bench`.

## Data

- Training and validation data come from the English rows of Ai4Privacy's OpenPII dataset (CC-BY-4.0), with labels mapped to EdgeVeil's types. OpenPII has no secrets, so credentials are generated synthetically and inserted into its text, along with hard negatives such as commit SHAs and UUIDs.
- Training runs on a cloud GPU on public synthetic data. Inference, export, and benchmarks run on the device.
- The test set is labeled in Label Studio and stored as JSONL character spans. `edgeveil data import-labels` checks every span against the labeling rules before writing it, and fixes go through a corrections file applied on top of the untouched export, never by hand-editing the labels.
- No real personal data is committed. Everything under `data/` and `models/` stays out of git.

Details are in [docs/data_card.md](docs/data_card.md), [docs/model_card.md](docs/model_card.md), and [docs/training.md](docs/training.md). Design decisions and the reasons behind them are in [docs/design.md](docs/design.md).

## Command line

| Command | What it does |
|---|---|
| `edgeveil scan <file>` | Print the masked text and the detected spans |
| `edgeveil proxy --port 8787 --upstream openai` | Start the masking proxy |
| `edgeveil eval --precision int8 --backend coreml` | Evaluate accuracy for one precision and backend |
| `edgeveil bench` | Run benchmarks and regenerate every results table |
| `edgeveil train`, `edgeveil tune-thresholds` | Fine-tune the detector and pick per-type thresholds |
| `edgeveil data ...` | Download, prepare, and augment training data; import and lint test-set labels |

## Scope

English text only. Images, PDFs, OCR, and voice are out of scope, and EdgeVeil makes no claim of HIPAA or GDPR compliance.

## Tech stack

Python 3.13 managed with uv. PyTorch, Transformers, and Datasets for training; ONNX Runtime, coremltools, and qai-hub for inference and on-device builds; FastAPI, uvicorn, httpx, and pydantic for the proxy; Typer and rich for the CLI. Quality checks use ruff and pyright, and tests use pytest, Hypothesis, and pytest-socket.

## Repository layout

```
src/edgeveil/
  detector/     model, windowing, decoding, backends
  validators/   rule-based finders and the combiner
  masker/       placeholders, vault, restorer
  proxy/        FastAPI app and upstream adapters
  cli/          Typer commands
  dataset/      training data download, cleaning, augmentation
  export/       ONNX, quantization, Core ML, AI Hub
  evaluation/   test set, scoring, label lint
  bench/        latency runs and table rendering
tests/
docs/
bench/results/  raw result files and rendered tables
```

## Development

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync --all-extras
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

The runtime core (masker, proxy, CLI, ONNX Runtime) installs without torch. Extras add the heavy dependencies: `train`, `coreml`, `qualcomm`, and `eval`. Tests run with sockets disabled, so they never reach the network; upstream LLMs are faked in-process.
