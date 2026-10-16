Accuracy against latency for every build. F1 is strict, overall, on the dev-test set with the fp32 torch thresholds. Token agreement is with fp32 torch on 2,000 validation windows. Latency is one window at batch size 1, model call only, after warmup. Latency machine: not measured.

| Backend | Precision | Compute units | Size (MB) | F1, model (95% CI) | F1, model and validators | Token agreement | Entity token agreement | p50 / p95 ms, 128 tokens | p50 / p95 ms, 256 tokens | p50 / p95 ms, 512 tokens | Neural Engine ops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch | fp32 | n/a | 260.8 | 0.729 [0.682, 0.772] | 0.836 | reference | reference | not measured | not measured | not measured | n/a |
| onnxruntime | fp32 | n/a | 261.8 | 0.729 [0.682, 0.772] | 0.836 | 1.0000 | 1.0000 | not measured | not measured | not measured | n/a |
| onnxruntime | fp16 | n/a | 131.3 | 0.728 [0.681, 0.771] | 0.836 | 1.0000 | 1.0000 | not measured | not measured | not measured | n/a |
| onnxruntime | int8-dynamic | n/a | 66.2 | 0.716 [0.662, 0.762] | 0.827 | 0.9993 | 0.9979 | not measured | not measured | not measured | n/a |
| onnxruntime | int8-static | n/a | 66.5 | 0.619 [0.568, 0.673] | 0.780 | 0.9960 | 0.9886 | not measured | not measured | not measured | n/a |
| onnxruntime | int4 | n/a | 118.3 | 0.723 [0.675, 0.766] | 0.825 | 0.9996 | 0.9989 | not measured | not measured | not measured | n/a |

Sources:

- `bench/results/accuracy/distilbert-fp16-onnxruntime-combined.json`
- `bench/results/accuracy/distilbert-fp16-onnxruntime.json`
- `bench/results/accuracy/distilbert-fp32-onnxruntime-combined.json`
- `bench/results/accuracy/distilbert-fp32-onnxruntime.json`
- `bench/results/accuracy/distilbert-fp32-torch-combined.json`
- `bench/results/accuracy/distilbert-fp32-torch.json`
- `bench/results/accuracy/distilbert-int4-onnxruntime-combined.json`
- `bench/results/accuracy/distilbert-int4-onnxruntime.json`
- `bench/results/accuracy/distilbert-int8-dynamic-onnxruntime-combined.json`
- `bench/results/accuracy/distilbert-int8-dynamic-onnxruntime.json`
- `bench/results/accuracy/distilbert-int8-static-onnxruntime-combined.json`
- `bench/results/accuracy/distilbert-int8-static-onnxruntime.json`
- `bench/results/parity/distilbert-fp16-onnxruntime.json`
- `bench/results/parity/distilbert-fp32-onnxruntime.json`
- `bench/results/parity/distilbert-int4-onnxruntime.json`
- `bench/results/parity/distilbert-int8-dynamic-onnxruntime.json`
- `bench/results/parity/distilbert-int8-static-onnxruntime.json`
