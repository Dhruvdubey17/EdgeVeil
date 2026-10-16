Detection: model and validators. distilbert/distilbert-base-cased, fp32, torch. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-fp32-torch-combined.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.873 [0.795, 0.943] | 0.920 [0.843, 0.985] | 0.896 [0.825, 0.956] | 0.937 [0.881, 0.987] | 0.987 [0.954, 1.000] | 0.973 [0.935, 0.998] |
| EMAIL | 45 | 0.956 [0.886, 1.000] | 0.956 [0.884, 1.000] | 0.956 [0.905, 0.990] | 0.956 [0.886, 1.000] | 0.956 [0.884, 1.000] | 0.988 [0.963, 1.000] |
| PHONE | 42 | 0.851 [0.741, 0.941] | 0.952 [0.881, 1.000] | 0.899 [0.816, 0.959] | 0.872 [0.776, 0.958] | 0.976 [0.921, 1.000] | 0.995 [0.985, 1.000] |
| ADDRESS | 47 | 0.487 [0.366, 0.622] | 0.787 [0.660, 0.900] | 0.602 [0.475, 0.726] | 0.697 [0.578, 0.803] | 0.936 [0.861, 1.000] | 0.957 [0.921, 0.985] |
| CREDENTIAL | 40 | 0.750 [0.586, 0.895] | 0.525 [0.366, 0.667] | 0.618 [0.455, 0.741] | 0.893 [0.778, 1.000] | 0.625 [0.486, 0.771] | 0.802 [0.674, 0.900] |
| SSN | 44 | 0.769 [0.643, 0.878] | 0.909 [0.816, 0.980] | 0.833 [0.745, 0.905] | 0.788 [0.673, 0.889] | 0.932 [0.853, 1.000] | 0.977 [0.948, 1.000] |
| CREDIT_CARD | 41 | 1.000 [1.000, 1.000] | 0.976 [0.918, 1.000] | 0.988 [0.957, 1.000] | 1.000 [1.000, 1.000] | 0.976 [0.918, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 43 | 0.692 [0.525, 0.864] | 0.628 [0.467, 0.778] | 0.658 [0.506, 0.788] | 0.872 [0.733, 0.976] | 0.744 [0.600, 0.864] | 0.857 [0.764, 0.939] |
| overall | 377 | 0.781 [0.735, 0.822] | 0.841 [0.796, 0.880] | 0.810 [0.769, 0.850] | 0.865 [0.826, 0.900] | 0.902 [0.870, 0.933] | 0.943 [0.921, 0.962] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 5 | 0 | 5 | 5 | 0 |
| EMAIL | 0 | 0 | 2 | 0 | 0 | 2 |
| PHONE | 0 | 1 | 1 | 0 | 1 | 6 |
| ADDRESS | 3 | 7 | 0 | 17 | 16 | 6 |
| CREDENTIAL | 10 | 4 | 5 | 3 | 4 | 0 |
| SSN | 2 | 1 | 1 | 5 | 1 | 6 |
| CREDIT_CARD | 0 | 0 | 1 | 0 | 0 | 0 |
| GOV_ID | 2 | 5 | 9 | 5 | 7 | 0 |
| overall | 18 | 23 | 19 | 35 | 34 | 20 |
