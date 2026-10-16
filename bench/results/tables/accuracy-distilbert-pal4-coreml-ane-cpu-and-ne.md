Detection: model only. distilbert/distilbert-base-cased, pal4, coreml-ane, compute units cpu-and-ne. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-pal4-coreml-ane-cpu-and-ne.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.925 [0.866, 0.976] | 0.987 [0.957, 1.000] | 0.955 [0.915, 0.987] | 0.938 [0.883, 0.987] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| EMAIL | 43 | 0.889 [0.788, 0.976] | 0.744 [0.615, 0.878] | 0.810 [0.697, 0.919] | 1.000 [1.000, 1.000] | 0.837 [0.722, 0.946] | 0.821 [0.709, 0.925] |
| PHONE | 42 | 0.548 [0.417, 0.680] | 0.809 [0.688, 0.917] | 0.654 [0.526, 0.757] | 0.581 [0.456, 0.702] | 0.857 [0.750, 0.955] | 0.926 [0.846, 0.983] |
| ADDRESS | 47 | 0.542 [0.409, 0.672] | 0.830 [0.718, 0.933] | 0.655 [0.526, 0.774] | 0.736 [0.621, 0.838] | 0.979 [0.930, 1.000] | 0.967 [0.942, 0.990] |
| CREDENTIAL | 40 | 0.692 [0.517, 0.875] | 0.450 [0.308, 0.622] | 0.545 [0.392, 0.704] | 0.962 [0.879, 1.000] | 0.625 [0.471, 0.763] | 0.774 [0.639, 0.888] |
| SSN | 44 | 0.719 [0.589, 0.839] | 0.932 [0.854, 1.000] | 0.812 [0.714, 0.887] | 0.719 [0.589, 0.839] | 0.932 [0.854, 1.000] | 0.983 [0.959, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.462 [0.267, 0.618] | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.936 [0.861, 0.994] |
| GOV_ID | 43 | 0.491 [0.333, 0.650] | 0.628 [0.468, 0.780] | 0.551 [0.400, 0.681] | 0.691 [0.542, 0.830] | 0.791 [0.656, 0.902] | 0.900 [0.814, 0.972] |
| overall | 374 | 0.693 [0.634, 0.745] | 0.741 [0.693, 0.785] | 0.716 [0.665, 0.760] | 0.790 [0.740, 0.833] | 0.816 [0.777, 0.850] | 0.913 [0.884, 0.940] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 0 | 1 | 0 | 5 | 1 | 0 |
| EMAIL | 7 | 4 | 0 | 0 | 4 | 0 |
| PHONE | 3 | 2 | 3 | 0 | 2 | 26 |
| ADDRESS | 1 | 7 | 0 | 17 | 14 | 2 |
| CREDENTIAL | 11 | 7 | 4 | 1 | 7 | 0 |
| SSN | 2 | 0 | 1 | 6 | 0 | 10 |
| CREDIT_CARD | 2 | 0 | 26 | 0 | 0 | 0 |
| GOV_ID | 2 | 7 | 7 | 9 | 11 | 8 |
| overall | 28 | 28 | 41 | 38 | 39 | 46 |
