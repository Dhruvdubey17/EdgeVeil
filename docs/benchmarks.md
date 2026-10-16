# Benchmarks

Every number in a results table comes from a JSON file under `bench/results/`, and `edgeveil bench` rebuilds every table from those files. The accuracy-versus-latency table is `bench/results/tables/accuracy-latency.md`. A cell reads `not measured` when its file does not exist.

## What each file measures

| Directory | Command | Measures |
|---|---|---|
| `accuracy/` | `edgeveil eval --backend B --precision P` | Strict P/R/F1 per type on the dev-test set, with bootstrap intervals, for the model alone, validators alone, and both |
| `parity/` | `edgeveil parity --backend B` | Token-level label agreement with fp32 torch on 2,000 validation windows that calibration never saw |
| `latency/` | `edgeveil bench latency --backend B` | Wall time of one model call on one window of 128, 256, and 512 tokens at batch size 1 |
| `placement/` | `edgeveil bench placement --backend B` | Share of Core ML operations whose preferred device is the Neural Engine, and the operators whose supported devices exclude it, from `MLComputePlan` |
| `power/` | `bash scripts/measure_power.sh` | CPU, GPU, and Neural Engine energy per 512-token window from `powermetrics` |

Latency runs 20 warmup calls and then 200 timed calls per length, and reports p50 and p95 (numpy linear interpolation) plus every raw timing. Tokenization and span decoding are not timed; they are the same code for every backend. Each file records the chip, machine model, memory, OS, Python and library versions, the load average, swap use, and `pmset -g therm` output before and after the run, and the build's path, size, and SHA-256. The command idles 60 seconds before timing each build so one build's heat does not carry into the next.

Close other apps before a latency run. Memory pressure and background load both move the numbers, and an 8 GiB machine swaps with a browser open.

## Order of a full run

```
uv run edgeveil export --backend onnxruntime --precision fp32
uv run edgeveil export --backend onnxruntime --precision fp16
uv run edgeveil export --backend onnxruntime --precision int8-dynamic
uv run edgeveil export --backend onnxruntime --precision int8-static
uv run edgeveil export --backend onnxruntime --precision int4
uv run edgeveil export --backend coreml --precision fp16
uv run edgeveil export --backend coreml --precision int8
uv run edgeveil export --backend coreml --precision int4
uv run edgeveil bench latency --backend onnxruntime
uv run edgeveil bench latency --backend coreml
uv run edgeveil export --backend coreml-ane --precision fp16
uv run edgeveil export --backend coreml-ane --precision int8
uv run edgeveil export --backend coreml-ane --precision int4
uv run edgeveil bench latency --backend coreml-ane --compute-units all --compute-units cpu-and-ne
uv run edgeveil bench placement
uv run edgeveil bench placement --backend coreml-ane
uv run edgeveil parity --backend onnxruntime
uv run edgeveil parity --backend coreml --compute-units all
uv run edgeveil parity --backend coreml --compute-units cpu-only
uv run edgeveil eval --backend coreml --precision fp16 --compute-units all
uv run edgeveil bench
```

Evaluate every ONNX and Core ML build the same way as the Core ML fp16 line, once per precision and, for Core ML, once per compute-unit setting.

## Power

`powermetrics` needs root, so this run is manual:

```
bash scripts/measure_power.sh
```

The script asks for the sudo password once, starts `powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -f plist`, and runs `edgeveil bench power-loop` four times: the Neural Engine layout fp16 build on cpu-only, cpu-and-gpu, and cpu-and-ne, then ONNX Runtime int8-dynamic on the CPU. Each loop idles 30 s, then runs one 512-token window in a loop for 60 s and saves its phase timestamps under `bench/results/power/loops/`. The script then stops `powermetrics` and runs `edgeveil bench power --log <log>`, which writes `bench/results/power/<run>.json` with mean power per rail in each phase, energy per window, energy per window above idle, and the samples it used. Close other apps first, as for latency. The raw plist stays in `$TMPDIR/edgeveil-power/`.

Sample times are the first plist timestamp (whole seconds) plus accumulated `elapsed_ns`, so samples within 2 s of a phase edge are left out. The parser reads `processor.cpu_power`, `gpu_power`, and `ane_power` in mW; these keys were checked against a real log on macOS 26.1.

## Xcode performance report (manual cross-check)

`edgeveil bench placement` reads the preferred device of each operation from `MLComputePlan`. The Xcode performance report measures placement on the device instead, so it is a one-time check that the two agree. It is not part of `edgeveil bench`, and no table cell comes from it.

1. Build the Core ML packages with `edgeveil export --backend coreml`.
2. Open one package in Xcode, for example `open models/distilbert/coreml/fp16.mlpackage`.
3. Open the Performance tab and create a report with the plus button. Pick this Mac as the device and All as the compute units.
4. Run the report. Note the shape Xcode uses for the enumerated inputs.
5. Compare the per-compute-unit operation counts in the report with `neural_engine_share`, `by_device`, and `operators_off_neural_engine` in `bench/results/placement/distilbert-fp16-coreml-all.json`.
6. Record the Xcode version, the date, the shape, and both sets of counts in `docs/design.md`, including any operator the two place differently.
