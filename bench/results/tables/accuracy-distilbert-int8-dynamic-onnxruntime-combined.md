Detection: model and validators. distilbert/distilbert-base-cased, int8-dynamic, onnxruntime. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int8-dynamic-onnxruntime-combined.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.924 [0.859, 0.976] | 0.973 [0.930, 1.000] | 0.948 [0.906, 0.980] | 0.924 [0.859, 0.976] | 0.973 [0.930, 1.000] | 0.970 [0.920, 1.000] |
| EMAIL | 43 | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 0.977 [0.940, 1.000] | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| PHONE | 42 | 0.889 [0.787, 0.976] | 0.952 [0.878, 1.000] | 0.919 [0.840, 0.978] | 0.911 [0.814, 0.980] | 0.976 [0.921, 1.000] | 1.000 [1.000, 1.000] |
| ADDRESS | 47 | 0.474 [0.346, 0.610] | 0.787 [0.667, 0.907] | 0.592 [0.460, 0.723] | 0.718 [0.609, 0.816] | 0.979 [0.930, 1.000] | 0.954 [0.910, 0.985] |
| CREDENTIAL | 40 | 0.733 [0.583, 0.889] | 0.550 [0.400, 0.700] | 0.629 [0.493, 0.754] | 0.833 [0.708, 0.952] | 0.600 [0.455, 0.739] | 0.830 [0.708, 0.924] |
| SSN | 44 | 0.792 [0.675, 0.891] | 0.955 [0.896, 1.000] | 0.866 [0.789, 0.926] | 0.792 [0.675, 0.891] | 0.955 [0.896, 1.000] | 0.983 [0.959, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 43 | 0.638 [0.475, 0.809] | 0.698 [0.543, 0.829] | 0.667 [0.519, 0.790] | 0.766 [0.604, 0.918] | 0.814 [0.690, 0.923] | 0.890 [0.797, 0.966] |
| overall | 374 | 0.784 [0.736, 0.831] | 0.874 [0.842, 0.907] | 0.827 [0.789, 0.865] | 0.854 [0.814, 0.892] | 0.920 [0.893, 0.945] | 0.950 [0.926, 0.969] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 2 | 0 | 0 | 5 | 0 | 1 |
| EMAIL | 0 | 0 | 0 | 0 | 0 | 2 |
| PHONE | 0 | 1 | 1 | 0 | 1 | 4 |
| ADDRESS | 1 | 9 | 0 | 17 | 19 | 5 |
| CREDENTIAL | 8 | 2 | 8 | 5 | 3 | 0 |
| SSN | 2 | 0 | 0 | 5 | 0 | 6 |
| CREDIT_CARD | 0 | 0 | 0 | 0 | 0 | 0 |
| GOV_ID | 2 | 5 | 6 | 10 | 6 | 1 |
| overall | 15 | 17 | 15 | 42 | 29 | 19 |
