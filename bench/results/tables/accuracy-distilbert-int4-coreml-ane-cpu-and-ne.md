Detection: model only. distilbert/distilbert-base-cased, int4, coreml-ane, compute units cpu-and-ne. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int4-coreml-ane-cpu-and-ne.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.901 [0.833, 0.963] | 0.973 [0.931, 1.000] | 0.936 [0.889, 0.973] | 0.914 [0.849, 0.971] | 0.987 [0.954, 1.000] | 0.985 [0.946, 1.000] |
| EMAIL | 43 | 0.949 [0.875, 1.000] | 0.861 [0.750, 0.957] | 0.902 [0.812, 0.976] | 1.000 [1.000, 1.000] | 0.907 [0.811, 0.979] | 0.905 [0.815, 0.979] |
| PHONE | 42 | 0.443 [0.315, 0.577] | 0.643 [0.500, 0.784] | 0.524 [0.395, 0.639] | 0.508 [0.377, 0.642] | 0.714 [0.571, 0.845] | 0.896 [0.805, 0.965] |
| ADDRESS | 47 | 0.462 [0.340, 0.603] | 0.766 [0.640, 0.886] | 0.576 [0.448, 0.708] | 0.718 [0.603, 0.827] | 0.979 [0.930, 1.000] | 0.940 [0.902, 0.977] |
| CREDENTIAL | 40 | 0.741 [0.577, 0.889] | 0.500 [0.350, 0.659] | 0.597 [0.444, 0.737] | 0.926 [0.829, 1.000] | 0.625 [0.471, 0.763] | 0.799 [0.669, 0.910] |
| SSN | 44 | 0.621 [0.486, 0.747] | 0.932 [0.854, 1.000] | 0.746 [0.635, 0.835] | 0.636 [0.500, 0.765] | 0.955 [0.896, 1.000] | 0.981 [0.955, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.462 [0.267, 0.618] | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.898 [0.817, 0.971] |
| GOV_ID | 43 | 0.575 [0.413, 0.734] | 0.628 [0.468, 0.780] | 0.600 [0.444, 0.730] | 0.745 [0.593, 0.872] | 0.744 [0.605, 0.868] | 0.872 [0.771, 0.957] |
| overall | 374 | 0.664 [0.606, 0.717] | 0.730 [0.676, 0.776] | 0.696 [0.642, 0.744] | 0.764 [0.713, 0.807] | 0.802 [0.761, 0.840] | 0.911 [0.882, 0.938] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 1 | 1 | 0 | 4 | 1 | 3 |
| EMAIL | 4 | 2 | 0 | 0 | 2 | 0 |
| PHONE | 4 | 3 | 8 | 0 | 4 | 30 |
| ADDRESS | 1 | 10 | 0 | 18 | 20 | 4 |
| CREDENTIAL | 9 | 5 | 6 | 2 | 5 | 0 |
| SSN | 2 | 1 | 0 | 5 | 1 | 19 |
| CREDIT_CARD | 2 | 0 | 26 | 0 | 0 | 0 |
| GOV_ID | 2 | 5 | 9 | 5 | 8 | 7 |
| overall | 25 | 27 | 49 | 34 | 41 | 63 |
