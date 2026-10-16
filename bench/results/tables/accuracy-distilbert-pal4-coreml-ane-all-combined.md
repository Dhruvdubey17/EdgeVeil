Detection: model and validators. distilbert/distilbert-base-cased, pal4, coreml-ane, compute units all. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-pal4-coreml-ane-all-combined.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.938 [0.883, 0.987] | 1.000 [1.000, 1.000] | 0.968 [0.938, 0.994] | 0.938 [0.883, 0.987] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| EMAIL | 43 | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 0.977 [0.940, 1.000] | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| PHONE | 42 | 0.911 [0.818, 0.979] | 0.976 [0.921, 1.000] | 0.943 [0.886, 0.988] | 0.911 [0.818, 0.979] | 0.976 [0.921, 1.000] | 1.000 [1.000, 1.000] |
| ADDRESS | 47 | 0.542 [0.409, 0.672] | 0.830 [0.718, 0.933] | 0.655 [0.526, 0.774] | 0.736 [0.621, 0.838] | 0.979 [0.930, 1.000] | 0.967 [0.942, 0.990] |
| CREDENTIAL | 40 | 0.786 [0.654, 0.923] | 0.550 [0.400, 0.698] | 0.647 [0.510, 0.761] | 0.857 [0.731, 0.967] | 0.600 [0.455, 0.739] | 0.810 [0.680, 0.912] |
| SSN | 44 | 0.759 [0.630, 0.869] | 0.932 [0.854, 1.000] | 0.837 [0.742, 0.909] | 0.759 [0.630, 0.869] | 0.932 [0.854, 1.000] | 0.983 [0.959, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 43 | 0.540 [0.370, 0.708] | 0.628 [0.468, 0.780] | 0.581 [0.422, 0.712] | 0.760 [0.609, 0.898] | 0.791 [0.656, 0.902] | 0.900 [0.814, 0.972] |
| overall | 374 | 0.792 [0.747, 0.837] | 0.877 [0.845, 0.909] | 0.833 [0.796, 0.868] | 0.858 [0.818, 0.892] | 0.920 [0.892, 0.944] | 0.955 [0.935, 0.970] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 0 | 0 | 0 | 5 | 0 | 0 |
| EMAIL | 0 | 0 | 0 | 0 | 0 | 2 |
| PHONE | 0 | 0 | 1 | 0 | 0 | 4 |
| ADDRESS | 1 | 7 | 0 | 17 | 14 | 2 |
| CREDENTIAL | 10 | 2 | 6 | 4 | 2 | 0 |
| SSN | 2 | 0 | 1 | 6 | 0 | 7 |
| CREDIT_CARD | 0 | 0 | 0 | 0 | 0 | 0 |
| GOV_ID | 2 | 7 | 7 | 10 | 11 | 2 |
| overall | 15 | 16 | 15 | 42 | 27 | 17 |
