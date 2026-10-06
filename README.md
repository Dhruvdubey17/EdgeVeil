# EdgeVeil

EdgeVeil runs on the device between an app and a cloud LLM. A small fine-tuned token classifier finds personal data (names, emails, phone numbers, addresses, credentials) locally, swaps each value for a reversible placeholder such as `⟦PERSON_1⟧` before the prompt is sent, and puts the real values back into the reply.

## Status

Milestone 0: project scaffold only. The `scan`, `proxy`, `eval`, and `bench` commands exist and exit with "not implemented yet". There is no model and there are no results yet.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```
uv sync --all-extras
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

The runtime core installs without torch. Extras add the heavy dependencies: `train` (torch, transformers, datasets, seqeval), `coreml` (coremltools), `qualcomm` (qai-hub), and `eval` (seqeval).

Tests run with sockets disabled, so they cannot reach the network.
