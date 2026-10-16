Accuracy against latency for every build. F1 is strict, overall, on the dev-test set with the fp32 torch thresholds. Token agreement is with fp32 torch on 2,000 validation windows. Latency is one window at batch size 1, model call only, after warmup. Energy is CPU, GPU, and Neural Engine power from powermetrics during a fixed loop, minus idle, divided by windows run. Latency machine: Apple M3, macOS-26.1-arm64-arm-64bit-Mach-O.

| Backend | Precision | Compute units | Size (MB) | F1, model (95% CI) | F1, model and validators | Token agreement | Entity token agreement | p50 / p95 ms, 128 tokens | p50 / p95 ms, 256 tokens | p50 / p95 ms, 512 tokens | Neural Engine ops | mJ per 512-token window, above idle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch | fp32 | n/a | 260.8 | 0.729 [0.682, 0.772] | 0.836 | reference | reference | not measured | not measured | not measured | n/a | not measured |
| onnxruntime | fp32 | n/a | 261.8 | 0.729 [0.682, 0.772] | 0.836 | 1.0000 | 1.0000 | 30.70 / 31.84 | 67.67 / 70.00 | 156.36 / 162.63 | n/a | not measured |
| onnxruntime | fp16 | n/a | 131.3 | 0.728 [0.681, 0.771] | 0.836 | 1.0000 | 1.0000 | 33.25 / 35.66 | 72.84 / 76.91 | 175.52 / 184.01 | n/a | not measured |
| onnxruntime | int8-dynamic | n/a | 66.2 | 0.716 [0.662, 0.762] | 0.827 | 0.9993 | 0.9979 | 11.82 / 12.31 | 27.62 / 29.11 | 75.99 / 78.85 | n/a | not measured |
| onnxruntime | int8-static | n/a | 66.5 | 0.619 [0.568, 0.673] | 0.780 | 0.9960 | 0.9886 | 14.25 / 14.78 | 32.79 / 33.69 | 87.07 / 89.85 | n/a | not measured |
| onnxruntime | int4 | n/a | 118.3 | 0.723 [0.675, 0.766] | 0.825 | 0.9996 | 0.9989 | 33.87 / 35.71 | 77.38 / 86.86 | 184.97 / 228.17 | n/a | not measured |
| coreml | fp16 | all | 130.5 | 0.728 [0.681, 0.771] | 0.836 | 1.0000 | 1.0000 | 27.05 / 27.36 | 53.04 / 54.59 | 117.83 / 120.27 | 0.0% of 230 | not measured |
| coreml | fp16 | cpu-and-ne | 130.5 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 230 | not measured |
| coreml | fp16 | cpu-and-gpu | 130.5 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 231 | not measured |
| coreml | fp16 | cpu-only | 130.5 | 0.729 [0.681, 0.773] | 0.837 | 1.0000 | 0.9999 | 5.82 / 6.10 | 11.40 / 11.68 | 25.70 / 25.94 | 0.0% of 231 | not measured |
| coreml | int8 | all | 65.5 | 0.725 [0.674, 0.768] | 0.831 | 1.0000 | 0.9999 | 27.16 / 27.49 | 52.72 / 53.29 | 117.63 / 118.18 | 0.0% of 234 | not measured |
| coreml | int8 | cpu-and-ne | 65.5 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 234 | not measured |
| coreml | int8 | cpu-and-gpu | 65.5 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 235 | not measured |
| coreml | int8 | cpu-only | 65.5 | 0.725 [0.674, 0.768] | 0.831 | 0.9999 | 0.9999 | 6.05 / 6.24 | 11.37 / 11.48 | 25.69 / 25.97 | 0.0% of 235 | not measured |
| coreml | int4 | all | 36.9 | 0.696 [0.643, 0.744] | 0.812 | 0.9994 | 0.9987 | 5.26 / 5.42 | 9.37 / 9.50 | 18.13 / 18.31 | 0.0% of 235 | not measured |
| coreml | int4 | cpu-and-ne | 36.9 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 2.1% of 235 | not measured |
| coreml | int4 | cpu-and-gpu | 36.9 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 235 | not measured |
| coreml | int4 | cpu-only | 36.9 | 0.696 [0.643, 0.744] | 0.812 | 0.9994 | 0.9987 | 6.03 / 6.19 | 11.37 / 11.55 | 25.69 / 25.92 | 0.0% of 235 | not measured |
| coreml-ane | fp16 | all | 130.6 | 0.727 [0.677, 0.769] | 0.833 | 1.0000 | 1.0000 | 1.74 / 2.20 | 2.37 / 2.44 | 6.33 / 6.38 | 98.1% of 412 | not measured |
| coreml-ane | fp16 | cpu-and-ne | 130.6 | 0.727 [0.677, 0.769] | 0.833 | 1.0000 | 1.0000 | 1.75 / 2.79 | 2.37 / 2.44 | 6.32 / 6.38 | 98.1% of 412 | not measured |
| coreml-ane | fp16 | cpu-and-gpu | 130.6 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 412 | not measured |
| coreml-ane | fp16 | cpu-only | 130.6 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 412 | not measured |
| coreml-ane | int8 | all | 65.6 | 0.724 [0.673, 0.767] | 0.830 | 1.0000 | 0.9999 | 3.34 / 4.54 | 3.81 / 4.70 | 5.45 / 5.50 | 97.1% of 416 | not measured |
| coreml-ane | int8 | cpu-and-ne | 65.6 | 0.724 [0.673, 0.767] | 0.830 | 1.0000 | 0.9999 | 1.28 / 3.84 | 2.43 / 2.50 | 5.46 / 5.56 | 97.1% of 416 | not measured |
| coreml-ane | int8 | cpu-and-gpu | 65.6 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 416 | not measured |
| coreml-ane | int8 | cpu-only | 65.6 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 416 | not measured |
| coreml-ane | int4 | all | 37.0 | 0.696 [0.643, 0.744] | 0.812 | 0.9994 | 0.9987 | 11.57 / 12.47 | 16.07 / 16.42 | 26.47 / 27.09 | 0.0% of 416 | not measured |
| coreml-ane | int4 | cpu-and-ne | 37.0 | 0.696 [0.642, 0.744] | 0.812 | 0.9994 | 0.9987 | 28.15 / 28.39 | 64.46 / 64.81 | 145.92 / 146.46 | 0.0% of 415 | not measured |
| coreml-ane | int4 | cpu-and-gpu | 37.0 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 416 | not measured |
| coreml-ane | int4 | cpu-only | 37.0 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | 0.0% of 416 | not measured |

Sources:

- `bench/results/accuracy/distilbert-fp16-coreml-all-combined.json`
- `bench/results/accuracy/distilbert-fp16-coreml-all.json`
- `bench/results/accuracy/distilbert-fp16-coreml-ane-all-combined.json`
- `bench/results/accuracy/distilbert-fp16-coreml-ane-all.json`
- `bench/results/accuracy/distilbert-fp16-coreml-ane-cpu-and-ne-combined.json`
- `bench/results/accuracy/distilbert-fp16-coreml-ane-cpu-and-ne.json`
- `bench/results/accuracy/distilbert-fp16-coreml-cpu-only-combined.json`
- `bench/results/accuracy/distilbert-fp16-coreml-cpu-only.json`
- `bench/results/accuracy/distilbert-fp16-onnxruntime-combined.json`
- `bench/results/accuracy/distilbert-fp16-onnxruntime.json`
- `bench/results/accuracy/distilbert-fp32-onnxruntime-combined.json`
- `bench/results/accuracy/distilbert-fp32-onnxruntime.json`
- `bench/results/accuracy/distilbert-fp32-torch-combined.json`
- `bench/results/accuracy/distilbert-fp32-torch.json`
- `bench/results/accuracy/distilbert-int4-coreml-all-combined.json`
- `bench/results/accuracy/distilbert-int4-coreml-all.json`
- `bench/results/accuracy/distilbert-int4-coreml-ane-all-combined.json`
- `bench/results/accuracy/distilbert-int4-coreml-ane-all.json`
- `bench/results/accuracy/distilbert-int4-coreml-ane-cpu-and-ne-combined.json`
- `bench/results/accuracy/distilbert-int4-coreml-ane-cpu-and-ne.json`
- `bench/results/accuracy/distilbert-int4-coreml-cpu-only-combined.json`
- `bench/results/accuracy/distilbert-int4-coreml-cpu-only.json`
- `bench/results/accuracy/distilbert-int4-onnxruntime-combined.json`
- `bench/results/accuracy/distilbert-int4-onnxruntime.json`
- `bench/results/accuracy/distilbert-int8-coreml-all-combined.json`
- `bench/results/accuracy/distilbert-int8-coreml-all.json`
- `bench/results/accuracy/distilbert-int8-coreml-ane-all-combined.json`
- `bench/results/accuracy/distilbert-int8-coreml-ane-all.json`
- `bench/results/accuracy/distilbert-int8-coreml-ane-cpu-and-ne-combined.json`
- `bench/results/accuracy/distilbert-int8-coreml-ane-cpu-and-ne.json`
- `bench/results/accuracy/distilbert-int8-coreml-cpu-only-combined.json`
- `bench/results/accuracy/distilbert-int8-coreml-cpu-only.json`
- `bench/results/accuracy/distilbert-int8-dynamic-onnxruntime-combined.json`
- `bench/results/accuracy/distilbert-int8-dynamic-onnxruntime.json`
- `bench/results/accuracy/distilbert-int8-static-onnxruntime-combined.json`
- `bench/results/accuracy/distilbert-int8-static-onnxruntime.json`
- `bench/results/latency/distilbert-fp16-coreml-all.json`
- `bench/results/latency/distilbert-fp16-coreml-ane-all.json`
- `bench/results/latency/distilbert-fp16-coreml-ane-cpu-and-ne.json`
- `bench/results/latency/distilbert-fp16-coreml-cpu-only.json`
- `bench/results/latency/distilbert-fp16-onnxruntime.json`
- `bench/results/latency/distilbert-fp32-onnxruntime.json`
- `bench/results/latency/distilbert-int4-coreml-all.json`
- `bench/results/latency/distilbert-int4-coreml-ane-all.json`
- `bench/results/latency/distilbert-int4-coreml-ane-cpu-and-ne.json`
- `bench/results/latency/distilbert-int4-coreml-cpu-only.json`
- `bench/results/latency/distilbert-int4-onnxruntime.json`
- `bench/results/latency/distilbert-int8-coreml-all.json`
- `bench/results/latency/distilbert-int8-coreml-ane-all.json`
- `bench/results/latency/distilbert-int8-coreml-ane-cpu-and-ne.json`
- `bench/results/latency/distilbert-int8-coreml-cpu-only.json`
- `bench/results/latency/distilbert-int8-dynamic-onnxruntime.json`
- `bench/results/latency/distilbert-int8-static-onnxruntime.json`
- `bench/results/parity/distilbert-fp16-coreml-all.json`
- `bench/results/parity/distilbert-fp16-coreml-ane-all.json`
- `bench/results/parity/distilbert-fp16-coreml-ane-cpu-and-ne.json`
- `bench/results/parity/distilbert-fp16-coreml-cpu-only.json`
- `bench/results/parity/distilbert-fp16-onnxruntime.json`
- `bench/results/parity/distilbert-fp32-onnxruntime.json`
- `bench/results/parity/distilbert-int4-coreml-all.json`
- `bench/results/parity/distilbert-int4-coreml-ane-all.json`
- `bench/results/parity/distilbert-int4-coreml-ane-cpu-and-ne.json`
- `bench/results/parity/distilbert-int4-coreml-cpu-only.json`
- `bench/results/parity/distilbert-int4-onnxruntime.json`
- `bench/results/parity/distilbert-int8-coreml-all.json`
- `bench/results/parity/distilbert-int8-coreml-ane-all.json`
- `bench/results/parity/distilbert-int8-coreml-ane-cpu-and-ne.json`
- `bench/results/parity/distilbert-int8-coreml-cpu-only.json`
- `bench/results/parity/distilbert-int8-dynamic-onnxruntime.json`
- `bench/results/parity/distilbert-int8-static-onnxruntime.json`
- `bench/results/placement/distilbert-fp16-coreml-all.json`
- `bench/results/placement/distilbert-fp16-coreml-ane-all.json`
- `bench/results/placement/distilbert-fp16-coreml-ane-cpu-and-gpu.json`
- `bench/results/placement/distilbert-fp16-coreml-ane-cpu-and-ne.json`
- `bench/results/placement/distilbert-fp16-coreml-ane-cpu-only.json`
- `bench/results/placement/distilbert-fp16-coreml-cpu-and-gpu.json`
- `bench/results/placement/distilbert-fp16-coreml-cpu-and-ne.json`
- `bench/results/placement/distilbert-fp16-coreml-cpu-only.json`
- `bench/results/placement/distilbert-int4-coreml-all.json`
- `bench/results/placement/distilbert-int4-coreml-ane-all.json`
- `bench/results/placement/distilbert-int4-coreml-ane-cpu-and-gpu.json`
- `bench/results/placement/distilbert-int4-coreml-ane-cpu-and-ne.json`
- `bench/results/placement/distilbert-int4-coreml-ane-cpu-only.json`
- `bench/results/placement/distilbert-int4-coreml-cpu-and-gpu.json`
- `bench/results/placement/distilbert-int4-coreml-cpu-and-ne.json`
- `bench/results/placement/distilbert-int4-coreml-cpu-only.json`
- `bench/results/placement/distilbert-int8-coreml-all.json`
- `bench/results/placement/distilbert-int8-coreml-ane-all.json`
- `bench/results/placement/distilbert-int8-coreml-ane-cpu-and-gpu.json`
- `bench/results/placement/distilbert-int8-coreml-ane-cpu-and-ne.json`
- `bench/results/placement/distilbert-int8-coreml-ane-cpu-only.json`
- `bench/results/placement/distilbert-int8-coreml-cpu-and-gpu.json`
- `bench/results/placement/distilbert-int8-coreml-cpu-and-ne.json`
- `bench/results/placement/distilbert-int8-coreml-cpu-only.json`
