Detection: model and validators. distilbert/distilbert-base-cased, int8, coreml, compute units cpu-only. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int8-coreml-cpu-only-combined.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.925 [0.861, 0.977] | 0.987 [0.954, 1.000] | 0.955 [0.916, 0.986] | 0.925 [0.861, 0.977] | 0.987 [0.954, 1.000] | 0.985 [0.946, 1.000] |
| EMAIL | 43 | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 0.977 [0.940, 1.000] | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| PHONE | 42 | 0.886 [0.784, 0.974] | 0.929 [0.842, 1.000] | 0.907 [0.825, 0.974] | 0.932 [0.844, 1.000] | 0.976 [0.921, 1.000] | 1.000 [1.000, 1.000] |
| ADDRESS | 47 | 0.474 [0.351, 0.618] | 0.787 [0.667, 0.907] | 0.592 [0.462, 0.724] | 0.731 [0.622, 0.831] | 0.979 [0.930, 1.000] | 0.964 [0.936, 0.987] |
| CREDENTIAL | 40 | 0.767 [0.621, 0.897] | 0.575 [0.425, 0.714] | 0.657 [0.529, 0.763] | 0.833 [0.714, 0.952] | 0.625 [0.486, 0.771] | 0.816 [0.688, 0.912] |
| SSN | 44 | 0.774 [0.647, 0.880] | 0.932 [0.854, 1.000] | 0.845 [0.757, 0.915] | 0.792 [0.675, 0.891] | 0.955 [0.896, 1.000] | 0.981 [0.955, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 43 | 0.698 [0.533, 0.865] | 0.698 [0.543, 0.829] | 0.698 [0.555, 0.817] | 0.837 [0.700, 0.950] | 0.791 [0.656, 0.902] | 0.855 [0.749, 0.944] |
| overall | 374 | 0.792 [0.747, 0.836] | 0.874 [0.842, 0.907] | 0.831 [0.794, 0.867] | 0.867 [0.831, 0.901] | 0.922 [0.894, 0.949] | 0.950 [0.928, 0.968] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 0 | 0 | 5 | 0 | 1 |
| EMAIL | 0 | 0 | 0 | 0 | 0 | 2 |
| PHONE | 0 | 2 | 1 | 0 | 2 | 3 |
| ADDRESS | 1 | 9 | 0 | 17 | 20 | 4 |
| CREDENTIAL | 9 | 2 | 6 | 5 | 2 | 0 |
| SSN | 2 | 1 | 0 | 5 | 1 | 6 |
| CREDIT_CARD | 0 | 0 | 0 | 0 | 0 | 0 |
| GOV_ID | 3 | 4 | 6 | 6 | 6 | 1 |
| overall | 16 | 18 | 13 | 38 | 31 | 17 |
