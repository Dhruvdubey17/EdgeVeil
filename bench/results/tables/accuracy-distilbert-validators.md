Detection: validators only. distilbert/distilbert-base-cased, fp32, torch. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-validators.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | nan [nan, nan] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| EMAIL | 43 | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 0.977 [0.940, 1.000] | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| PHONE | 42 | 0.930 [0.846, 1.000] | 0.952 [0.879, 1.000] | 0.941 [0.875, 0.988] | 0.930 [0.846, 1.000] | 0.952 [0.879, 1.000] | 0.946 [0.864, 1.000] |
| ADDRESS | 47 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | nan [nan, nan] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CREDENTIAL | 40 | 0.933 [0.778, 1.000] | 0.350 [0.209, 0.500] | 0.509 [0.333, 0.654] | 0.933 [0.778, 1.000] | 0.350 [0.209, 0.500] | 0.569 [0.391, 0.727] |
| SSN | 44 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | nan [nan, nan] | 0.000 [0.000, 0.000] | 0.024 [0.000, 0.076] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 43 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | nan [nan, nan] | 0.000 [0.000, 0.000] | 0.057 [0.000, 0.146] |
| overall | 374 | 0.958 [0.918, 0.987] | 0.366 [0.323, 0.413] | 0.530 [0.481, 0.580] | 0.958 [0.918, 0.987] | 0.366 [0.323, 0.413] | 0.394 [0.352, 0.441] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 75 | 0 | 0 | 0 | 0 | 0 |
| EMAIL | 0 | 0 | 0 | 0 | 0 | 2 |
| PHONE | 2 | 0 | 0 | 0 | 0 | 3 |
| ADDRESS | 47 | 0 | 0 | 0 | 0 | 0 |
| CREDENTIAL | 24 | 0 | 2 | 1 | 0 | 0 |
| SSN | 43 | 0 | 1 | 0 | 0 | 0 |
| CREDIT_CARD | 0 | 0 | 0 | 0 | 0 | 0 |
| GOV_ID | 41 | 0 | 2 | 0 | 0 | 0 |
| overall | 232 | 0 | 5 | 1 | 0 | 5 |
