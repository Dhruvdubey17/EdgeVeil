Detection: model and validators. distilbert/distilbert-base-cased, int4, coreml-ane, compute units cpu-and-ne. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int4-coreml-ane-cpu-and-ne-combined.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.914 [0.849, 0.971] | 0.987 [0.954, 1.000] | 0.949 [0.908, 0.981] | 0.914 [0.849, 0.971] | 0.987 [0.954, 1.000] | 0.985 [0.946, 1.000] |
| EMAIL | 43 | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 0.977 [0.940, 1.000] | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| PHONE | 42 | 0.833 [0.707, 0.943] | 0.952 [0.878, 1.000] | 0.889 [0.795, 0.962] | 0.875 [0.773, 0.959] | 0.976 [0.921, 1.000] | 1.000 [1.000, 1.000] |
| ADDRESS | 47 | 0.462 [0.340, 0.603] | 0.766 [0.640, 0.886] | 0.576 [0.448, 0.708] | 0.718 [0.603, 0.827] | 0.979 [0.930, 1.000] | 0.940 [0.902, 0.977] |
| CREDENTIAL | 40 | 0.733 [0.583, 0.889] | 0.550 [0.400, 0.700] | 0.629 [0.493, 0.754] | 0.833 [0.708, 0.952] | 0.600 [0.455, 0.739] | 0.833 [0.708, 0.924] |
| SSN | 44 | 0.695 [0.544, 0.820] | 0.932 [0.854, 1.000] | 0.796 [0.680, 0.882] | 0.712 [0.561, 0.833] | 0.955 [0.896, 1.000] | 0.981 [0.955, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 43 | 0.658 [0.486, 0.825] | 0.628 [0.468, 0.780] | 0.643 [0.487, 0.776] | 0.854 [0.730, 0.951] | 0.744 [0.605, 0.868] | 0.872 [0.771, 0.957] |
| overall | 374 | 0.765 [0.713, 0.819] | 0.864 [0.830, 0.899] | 0.812 [0.770, 0.854] | 0.846 [0.807, 0.886] | 0.914 [0.885, 0.940] | 0.947 [0.923, 0.967] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 0 | 0 | 4 | 0 | 3 |
| EMAIL | 0 | 0 | 0 | 0 | 0 | 2 |
| PHONE | 0 | 1 | 1 | 0 | 2 | 6 |
| ADDRESS | 1 | 10 | 0 | 18 | 20 | 4 |
| CREDENTIAL | 8 | 2 | 8 | 5 | 3 | 0 |
| SSN | 2 | 1 | 0 | 5 | 1 | 12 |
| CREDIT_CARD | 0 | 0 | 0 | 0 | 0 | 0 |
| GOV_ID | 2 | 5 | 9 | 5 | 8 | 1 |
| overall | 14 | 19 | 18 | 37 | 34 | 28 |
