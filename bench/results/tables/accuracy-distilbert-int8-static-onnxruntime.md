Detection: model only. distilbert/distilbert-base-cased, int8-static, onnxruntime. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int8-static-onnxruntime.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.855 [0.766, 0.940] | 0.867 [0.783, 0.941] | 0.861 [0.780, 0.936] | 0.961 [0.912, 1.000] | 0.960 [0.911, 1.000] | 0.922 [0.864, 0.971] |
| EMAIL | 43 | 0.656 [0.484, 0.840] | 0.488 [0.350, 0.643] | 0.560 [0.418, 0.720] | 1.000 [1.000, 1.000] | 0.698 [0.561, 0.837] | 0.584 [0.452, 0.729] |
| PHONE | 42 | 0.404 [0.275, 0.535] | 0.500 [0.340, 0.657] | 0.447 [0.309, 0.578] | 0.615 [0.473, 0.750] | 0.738 [0.595, 0.863] | 0.852 [0.749, 0.927] |
| ADDRESS | 47 | 0.458 [0.333, 0.614] | 0.702 [0.578, 0.826] | 0.555 [0.424, 0.699] | 0.819 [0.725, 0.910] | 0.979 [0.930, 1.000] | 0.916 [0.867, 0.962] |
| CREDENTIAL | 40 | 0.783 [0.600, 0.950] | 0.450 [0.302, 0.612] | 0.571 [0.414, 0.717] | 0.957 [0.864, 1.000] | 0.550 [0.390, 0.711] | 0.716 [0.570, 0.845] |
| SSN | 44 | 0.612 [0.484, 0.746] | 0.932 [0.854, 1.000] | 0.739 [0.632, 0.827] | 0.612 [0.484, 0.746] | 0.932 [0.854, 1.000] | 0.968 [0.925, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 0.275 [0.133, 0.417] | 0.431 [0.235, 0.588] | 1.000 [1.000, 1.000] | 0.275 [0.133, 0.417] | 0.710 [0.578, 0.843] |
| GOV_ID | 43 | 0.490 [0.320, 0.646] | 0.558 [0.391, 0.704] | 0.522 [0.365, 0.660] | 0.735 [0.581, 0.869] | 0.744 [0.603, 0.872] | 0.743 [0.619, 0.860] |
| overall | 374 | 0.613 [0.555, 0.672] | 0.626 [0.574, 0.679] | 0.619 [0.568, 0.673] | 0.801 [0.750, 0.844] | 0.762 [0.720, 0.804] | 0.806 [0.768, 0.846] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 3 | 7 | 0 | 2 | 8 | 1 |
| EMAIL | 12 | 9 | 1 | 0 | 11 | 0 |
| PHONE | 4 | 10 | 7 | 0 | 11 | 20 |
| ADDRESS | 1 | 13 | 0 | 8 | 26 | 5 |
| CREDENTIAL | 15 | 4 | 3 | 1 | 4 | 0 |
| SSN | 2 | 0 | 1 | 6 | 0 | 20 |
| CREDIT_CARD | 7 | 0 | 22 | 0 | 0 | 0 |
| GOV_ID | 7 | 8 | 4 | 2 | 12 | 11 |
| overall | 51 | 51 | 38 | 19 | 72 | 57 |
