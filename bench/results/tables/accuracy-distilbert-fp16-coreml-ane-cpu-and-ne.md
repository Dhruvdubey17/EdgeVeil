Detection: model only. distilbert/distilbert-base-cased, fp16, coreml-ane, compute units cpu-and-ne. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-fp16-coreml-ane-cpu-and-ne.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.924 [0.865, 0.976] | 0.973 [0.931, 1.000] | 0.948 [0.904, 0.984] | 0.937 [0.881, 0.987] | 0.987 [0.954, 1.000] | 0.985 [0.946, 1.000] |
| EMAIL | 43 | 0.902 [0.788, 1.000] | 0.861 [0.750, 0.957] | 0.881 [0.773, 0.971] | 1.000 [1.000, 1.000] | 0.930 [0.844, 1.000] | 0.917 [0.833, 0.981] |
| PHONE | 42 | 0.532 [0.403, 0.661] | 0.786 [0.657, 0.909] | 0.635 [0.506, 0.736] | 0.565 [0.441, 0.689] | 0.809 [0.690, 0.919] | 0.918 [0.838, 0.976] |
| ADDRESS | 47 | 0.474 [0.351, 0.618] | 0.787 [0.667, 0.907] | 0.592 [0.462, 0.724] | 0.731 [0.622, 0.831] | 0.979 [0.930, 1.000] | 0.964 [0.936, 0.987] |
| CREDENTIAL | 40 | 0.714 [0.555, 0.862] | 0.500 [0.350, 0.659] | 0.588 [0.441, 0.727] | 0.929 [0.833, 1.000] | 0.650 [0.500, 0.796] | 0.764 [0.625, 0.891] |
| SSN | 44 | 0.719 [0.596, 0.836] | 0.932 [0.854, 1.000] | 0.812 [0.718, 0.886] | 0.737 [0.625, 0.849] | 0.955 [0.896, 1.000] | 0.977 [0.948, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.462 [0.267, 0.618] | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.912 [0.839, 0.980] |
| GOV_ID | 43 | 0.625 [0.474, 0.775] | 0.698 [0.543, 0.829] | 0.659 [0.526, 0.769] | 0.729 [0.571, 0.872] | 0.791 [0.656, 0.902] | 0.880 [0.790, 0.960] |
| overall | 374 | 0.699 [0.644, 0.747] | 0.757 [0.707, 0.801] | 0.727 [0.677, 0.769] | 0.795 [0.747, 0.836] | 0.824 [0.782, 0.860] | 0.917 [0.889, 0.944] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 1 | 0 | 5 | 1 | 0 |
| EMAIL | 3 | 3 | 0 | 0 | 4 | 0 |
| PHONE | 1 | 1 | 7 | 0 | 2 | 27 |
| ADDRESS | 1 | 9 | 0 | 17 | 20 | 4 |
| CREDENTIAL | 11 | 6 | 3 | 2 | 6 | 0 |
| SSN | 2 | 1 | 0 | 5 | 1 | 10 |
| CREDIT_CARD | 2 | 0 | 26 | 0 | 0 | 0 |
| GOV_ID | 2 | 4 | 7 | 6 | 5 | 7 |
| overall | 23 | 25 | 43 | 35 | 39 | 48 |
