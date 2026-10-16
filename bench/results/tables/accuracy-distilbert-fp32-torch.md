Detection: model only. distilbert/distilbert-base-cased, fp32, torch. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-fp32-torch.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.873 [0.795, 0.943] | 0.920 [0.843, 0.985] | 0.896 [0.825, 0.956] | 0.937 [0.881, 0.987] | 0.987 [0.954, 1.000] | 0.973 [0.935, 0.998] |
| EMAIL | 45 | 0.833 [0.706, 0.952] | 0.778 [0.650, 0.900] | 0.805 [0.689, 0.918] | 1.000 [1.000, 1.000] | 0.889 [0.780, 0.975] | 0.907 [0.831, 0.972] |
| PHONE | 42 | 0.516 [0.375, 0.638] | 0.762 [0.630, 0.889] | 0.615 [0.480, 0.722] | 0.565 [0.441, 0.689] | 0.809 [0.690, 0.919] | 0.914 [0.834, 0.972] |
| ADDRESS | 47 | 0.474 [0.359, 0.608] | 0.787 [0.660, 0.900] | 0.592 [0.466, 0.713] | 0.679 [0.565, 0.789] | 0.936 [0.861, 1.000] | 0.957 [0.921, 0.985] |
| CREDENTIAL | 40 | 0.714 [0.559, 0.870] | 0.500 [0.346, 0.649] | 0.588 [0.431, 0.727] | 0.929 [0.833, 1.000] | 0.650 [0.500, 0.796] | 0.759 [0.621, 0.884] |
| SSN | 44 | 0.719 [0.596, 0.836] | 0.932 [0.854, 1.000] | 0.812 [0.718, 0.886] | 0.737 [0.625, 0.849] | 0.955 [0.896, 1.000] | 0.977 [0.948, 1.000] |
| CREDIT_CARD | 41 | 1.000 [1.000, 1.000] | 0.293 [0.150, 0.439] | 0.453 [0.261, 0.610] | 1.000 [1.000, 1.000] | 0.293 [0.150, 0.439] | 0.912 [0.839, 0.980] |
| GOV_ID | 43 | 0.604 [0.444, 0.755] | 0.674 [0.521, 0.812] | 0.637 [0.488, 0.757] | 0.750 [0.600, 0.884] | 0.791 [0.656, 0.902] | 0.874 [0.786, 0.953] |
| overall | 377 | 0.677 [0.623, 0.727] | 0.729 [0.679, 0.776] | 0.702 [0.652, 0.748] | 0.788 [0.740, 0.829] | 0.812 [0.768, 0.850] | 0.911 [0.882, 0.937] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 5 | 0 | 5 | 5 | 0 |
| EMAIL | 3 | 5 | 2 | 0 | 7 | 0 |
| PHONE | 1 | 2 | 7 | 0 | 3 | 27 |
| ADDRESS | 3 | 7 | 0 | 17 | 16 | 8 |
| CREDENTIAL | 11 | 6 | 3 | 2 | 6 | 0 |
| SSN | 2 | 1 | 0 | 5 | 1 | 10 |
| CREDIT_CARD | 2 | 0 | 27 | 0 | 0 | 0 |
| GOV_ID | 2 | 5 | 7 | 5 | 7 | 7 |
| overall | 25 | 31 | 46 | 34 | 45 | 52 |
