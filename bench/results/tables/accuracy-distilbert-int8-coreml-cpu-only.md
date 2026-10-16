Detection: model only. distilbert/distilbert-base-cased, int8, coreml, compute units cpu-only. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int8-coreml-cpu-only.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.912 [0.845, 0.973] | 0.973 [0.931, 1.000] | 0.942 [0.892, 0.980] | 0.925 [0.861, 0.977] | 0.987 [0.954, 1.000] | 0.985 [0.946, 1.000] |
| EMAIL | 43 | 0.902 [0.788, 1.000] | 0.861 [0.750, 0.957] | 0.881 [0.773, 0.971] | 1.000 [1.000, 1.000] | 0.930 [0.844, 1.000] | 0.917 [0.833, 0.981] |
| PHONE | 42 | 0.541 [0.409, 0.672] | 0.786 [0.657, 0.909] | 0.641 [0.511, 0.742] | 0.574 [0.444, 0.698] | 0.809 [0.690, 0.919] | 0.918 [0.838, 0.976] |
| ADDRESS | 47 | 0.474 [0.351, 0.618] | 0.787 [0.667, 0.907] | 0.592 [0.462, 0.724] | 0.731 [0.622, 0.831] | 0.979 [0.930, 1.000] | 0.964 [0.936, 0.987] |
| CREDENTIAL | 40 | 0.714 [0.555, 0.862] | 0.500 [0.350, 0.659] | 0.588 [0.441, 0.727] | 0.929 [0.833, 1.000] | 0.650 [0.500, 0.796] | 0.771 [0.632, 0.896] |
| SSN | 44 | 0.719 [0.596, 0.836] | 0.932 [0.854, 1.000] | 0.812 [0.718, 0.886] | 0.737 [0.625, 0.849] | 0.955 [0.896, 1.000] | 0.981 [0.955, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.462 [0.267, 0.618] | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.912 [0.839, 0.980] |
| GOV_ID | 43 | 0.600 [0.444, 0.750] | 0.698 [0.543, 0.829] | 0.645 [0.511, 0.758] | 0.720 [0.561, 0.857] | 0.791 [0.656, 0.902] | 0.855 [0.749, 0.944] |
| overall | 374 | 0.695 [0.640, 0.746] | 0.757 [0.707, 0.801] | 0.725 [0.674, 0.768] | 0.794 [0.747, 0.835] | 0.824 [0.782, 0.860] | 0.917 [0.888, 0.943] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 1 | 0 | 5 | 1 | 1 |
| EMAIL | 3 | 3 | 0 | 0 | 4 | 0 |
| PHONE | 1 | 1 | 7 | 0 | 2 | 26 |
| ADDRESS | 1 | 9 | 0 | 17 | 20 | 4 |
| CREDENTIAL | 10 | 6 | 4 | 2 | 6 | 0 |
| SSN | 2 | 1 | 0 | 5 | 1 | 10 |
| CREDIT_CARD | 2 | 0 | 26 | 0 | 0 | 0 |
| GOV_ID | 3 | 4 | 6 | 6 | 6 | 8 |
| overall | 23 | 25 | 43 | 35 | 40 | 49 |
