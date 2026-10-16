Detection: model and validators. distilbert/distilbert-base-cased, fp16, coreml-ane, compute units all. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-fp16-coreml-ane-all-combined.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.937 [0.881, 0.987] | 0.987 [0.954, 1.000] | 0.961 [0.925, 0.987] | 0.937 [0.881, 0.987] | 0.987 [0.954, 1.000] | 0.985 [0.946, 1.000] |
| EMAIL | 43 | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 0.977 [0.940, 1.000] | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| PHONE | 42 | 0.867 [0.756, 0.958] | 0.929 [0.842, 1.000] | 0.897 [0.809, 0.966] | 0.911 [0.818, 0.979] | 0.976 [0.921, 1.000] | 1.000 [1.000, 1.000] |
| ADDRESS | 47 | 0.474 [0.351, 0.618] | 0.787 [0.667, 0.907] | 0.592 [0.462, 0.724] | 0.731 [0.622, 0.831] | 0.979 [0.930, 1.000] | 0.964 [0.936, 0.987] |
| CREDENTIAL | 40 | 0.767 [0.621, 0.897] | 0.575 [0.425, 0.714] | 0.657 [0.529, 0.763] | 0.833 [0.714, 0.952] | 0.625 [0.486, 0.771] | 0.808 [0.679, 0.905] |
| SSN | 44 | 0.774 [0.647, 0.880] | 0.932 [0.854, 1.000] | 0.845 [0.757, 0.915] | 0.792 [0.675, 0.891] | 0.955 [0.896, 1.000] | 0.977 [0.948, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 43 | 0.732 [0.579, 0.889] | 0.698 [0.543, 0.829] | 0.714 [0.571, 0.828] | 0.854 [0.719, 0.971] | 0.791 [0.656, 0.902] | 0.880 [0.790, 0.960] |
| overall | 374 | 0.796 [0.753, 0.838] | 0.874 [0.842, 0.907] | 0.833 [0.798, 0.868] | 0.869 [0.833, 0.902] | 0.922 [0.894, 0.949] | 0.950 [0.929, 0.968] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 0 | 0 | 5 | 0 | 0 |
| EMAIL | 0 | 0 | 0 | 0 | 0 | 2 |
| PHONE | 0 | 2 | 1 | 0 | 2 | 4 |
| ADDRESS | 1 | 9 | 0 | 17 | 20 | 4 |
| CREDENTIAL | 10 | 2 | 5 | 5 | 2 | 0 |
| SSN | 2 | 1 | 0 | 5 | 1 | 6 |
| CREDIT_CARD | 0 | 0 | 0 | 0 | 0 | 0 |
| GOV_ID | 2 | 4 | 7 | 6 | 5 | 0 |
| overall | 16 | 18 | 13 | 38 | 30 | 16 |
