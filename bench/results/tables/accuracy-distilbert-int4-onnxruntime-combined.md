Detection: model and validators. distilbert/distilbert-base-cased, int4, onnxruntime. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int4-onnxruntime-combined.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.937 [0.878, 0.987] | 0.987 [0.954, 1.000] | 0.961 [0.926, 0.988] | 0.937 [0.878, 0.987] | 0.987 [0.954, 1.000] | 0.985 [0.946, 1.000] |
| EMAIL | 43 | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 0.977 [0.940, 1.000] | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| PHONE | 42 | 0.844 [0.732, 0.942] | 0.905 [0.805, 0.978] | 0.874 [0.774, 0.954] | 0.911 [0.814, 0.980] | 0.976 [0.921, 1.000] | 1.000 [1.000, 1.000] |
| ADDRESS | 47 | 0.451 [0.337, 0.579] | 0.787 [0.674, 0.896] | 0.574 [0.455, 0.695] | 0.683 [0.571, 0.790] | 0.979 [0.930, 1.000] | 0.954 [0.915, 0.984] |
| CREDENTIAL | 40 | 0.710 [0.559, 0.870] | 0.550 [0.400, 0.700] | 0.620 [0.481, 0.754] | 0.839 [0.720, 0.952] | 0.625 [0.486, 0.771] | 0.832 [0.710, 0.924] |
| SSN | 44 | 0.808 [0.694, 0.905] | 0.955 [0.896, 1.000] | 0.875 [0.800, 0.935] | 0.808 [0.694, 0.905] | 0.955 [0.896, 1.000] | 0.983 [0.959, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 43 | 0.725 [0.562, 0.886] | 0.674 [0.521, 0.812] | 0.699 [0.551, 0.817] | 0.850 [0.714, 0.976] | 0.791 [0.658, 0.900] | 0.857 [0.746, 0.953] |
| overall | 374 | 0.785 [0.739, 0.829] | 0.869 [0.835, 0.901] | 0.825 [0.787, 0.861] | 0.860 [0.820, 0.896] | 0.922 [0.894, 0.949] | 0.950 [0.927, 0.969] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 0 | 0 | 3 | 0 | 2 |
| EMAIL | 0 | 0 | 0 | 0 | 0 | 2 |
| PHONE | 0 | 3 | 1 | 0 | 3 | 4 |
| ADDRESS | 1 | 9 | 0 | 20 | 19 | 6 |
| CREDENTIAL | 7 | 3 | 8 | 5 | 4 | 0 |
| SSN | 2 | 0 | 0 | 4 | 0 | 6 |
| CREDIT_CARD | 0 | 0 | 0 | 0 | 0 | 0 |
| GOV_ID | 3 | 5 | 6 | 6 | 5 | 0 |
| overall | 14 | 20 | 15 | 38 | 31 | 20 |
