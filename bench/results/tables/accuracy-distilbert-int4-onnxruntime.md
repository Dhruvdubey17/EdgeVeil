Detection: model only. distilbert/distilbert-base-cased, int4, onnxruntime. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int4-onnxruntime.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.924 [0.866, 0.977] | 0.973 [0.931, 1.000] | 0.948 [0.904, 0.983] | 0.937 [0.878, 0.987] | 0.987 [0.954, 1.000] | 0.985 [0.946, 1.000] |
| EMAIL | 43 | 0.902 [0.788, 1.000] | 0.861 [0.750, 0.957] | 0.881 [0.773, 0.971] | 1.000 [1.000, 1.000] | 0.930 [0.844, 1.000] | 0.917 [0.833, 0.981] |
| PHONE | 42 | 0.525 [0.397, 0.650] | 0.762 [0.622, 0.889] | 0.621 [0.495, 0.726] | 0.574 [0.452, 0.696] | 0.809 [0.690, 0.919] | 0.926 [0.854, 0.977] |
| ADDRESS | 47 | 0.451 [0.337, 0.579] | 0.787 [0.674, 0.896] | 0.574 [0.455, 0.695] | 0.683 [0.571, 0.790] | 0.979 [0.930, 1.000] | 0.954 [0.915, 0.984] |
| CREDENTIAL | 40 | 0.741 [0.583, 0.889] | 0.500 [0.350, 0.659] | 0.597 [0.452, 0.732] | 0.926 [0.833, 1.000] | 0.625 [0.477, 0.775] | 0.794 [0.666, 0.906] |
| SSN | 44 | 0.737 [0.619, 0.854] | 0.955 [0.896, 1.000] | 0.832 [0.750, 0.902] | 0.737 [0.619, 0.854] | 0.955 [0.896, 1.000] | 0.983 [0.959, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.462 [0.267, 0.618] | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.919 [0.843, 0.986] |
| GOV_ID | 43 | 0.617 [0.460, 0.774] | 0.674 [0.521, 0.812] | 0.644 [0.500, 0.761] | 0.723 [0.568, 0.872] | 0.791 [0.658, 0.900] | 0.857 [0.746, 0.953] |
| overall | 374 | 0.695 [0.642, 0.743] | 0.754 [0.704, 0.799] | 0.723 [0.675, 0.766] | 0.786 [0.736, 0.829] | 0.821 [0.780, 0.856] | 0.919 [0.892, 0.944] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 1 | 0 | 3 | 1 | 2 |
| EMAIL | 3 | 3 | 0 | 0 | 4 | 0 |
| PHONE | 0 | 2 | 8 | 0 | 3 | 26 |
| ADDRESS | 1 | 9 | 0 | 20 | 19 | 6 |
| CREDENTIAL | 8 | 5 | 7 | 2 | 5 | 0 |
| SSN | 2 | 0 | 0 | 4 | 0 | 11 |
| CREDIT_CARD | 2 | 0 | 26 | 0 | 0 | 0 |
| GOV_ID | 3 | 5 | 6 | 6 | 5 | 7 |
| overall | 20 | 25 | 47 | 35 | 37 | 52 |
