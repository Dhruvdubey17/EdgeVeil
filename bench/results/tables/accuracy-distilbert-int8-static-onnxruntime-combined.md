Detection: model and validators. distilbert/distilbert-base-cased, int8-static, onnxruntime. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int8-static-onnxruntime-combined.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.868 [0.784, 0.947] | 0.880 [0.797, 0.947] | 0.874 [0.797, 0.941] | 0.961 [0.912, 1.000] | 0.960 [0.911, 1.000] | 0.922 [0.864, 0.971] |
| EMAIL | 43 | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 0.977 [0.940, 1.000] | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| PHONE | 42 | 0.818 [0.681, 0.932] | 0.857 [0.745, 0.954] | 0.837 [0.723, 0.927] | 0.909 [0.811, 0.980] | 0.929 [0.844, 1.000] | 1.000 [1.000, 1.000] |
| ADDRESS | 47 | 0.458 [0.333, 0.614] | 0.702 [0.578, 0.826] | 0.555 [0.424, 0.699] | 0.819 [0.725, 0.910] | 0.979 [0.930, 1.000] | 0.916 [0.867, 0.962] |
| CREDENTIAL | 40 | 0.741 [0.577, 0.909] | 0.500 [0.354, 0.658] | 0.597 [0.449, 0.730] | 0.852 [0.722, 0.966] | 0.550 [0.400, 0.697] | 0.753 [0.610, 0.867] |
| SSN | 44 | 0.695 [0.545, 0.820] | 0.932 [0.854, 1.000] | 0.796 [0.688, 0.874] | 0.695 [0.545, 0.820] | 0.932 [0.854, 1.000] | 0.968 [0.925, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 43 | 0.600 [0.412, 0.786] | 0.558 [0.391, 0.704] | 0.578 [0.414, 0.722] | 0.900 [0.792, 0.977] | 0.744 [0.603, 0.872] | 0.750 [0.626, 0.868] |
| overall | 374 | 0.752 [0.694, 0.808] | 0.810 [0.770, 0.847] | 0.780 [0.734, 0.824] | 0.881 [0.842, 0.919] | 0.896 [0.865, 0.924] | 0.911 [0.884, 0.935] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 3 | 6 | 0 | 2 | 7 | 1 |
| EMAIL | 0 | 0 | 0 | 0 | 0 | 2 |
| PHONE | 0 | 3 | 3 | 0 | 4 | 4 |
| ADDRESS | 1 | 13 | 0 | 8 | 26 | 5 |
| CREDENTIAL | 14 | 2 | 4 | 4 | 3 | 0 |
| SSN | 2 | 0 | 1 | 6 | 0 | 12 |
| CREDIT_CARD | 0 | 0 | 0 | 0 | 0 | 0 |
| GOV_ID | 7 | 8 | 4 | 2 | 12 | 2 |
| overall | 27 | 32 | 12 | 22 | 52 | 26 |
