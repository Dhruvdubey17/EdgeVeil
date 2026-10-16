Detection: model only. distilbert/distilbert-base-cased, int8-dynamic, onnxruntime. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int8-dynamic-onnxruntime.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.911 [0.844, 0.973] | 0.960 [0.908, 1.000] | 0.935 [0.885, 0.974] | 0.924 [0.859, 0.976] | 0.973 [0.930, 1.000] | 0.970 [0.920, 1.000] |
| EMAIL | 43 | 0.868 [0.761, 0.971] | 0.767 [0.630, 0.897] | 0.815 [0.694, 0.923] | 1.000 [1.000, 1.000] | 0.884 [0.775, 0.974] | 0.837 [0.725, 0.936] |
| PHONE | 42 | 0.517 [0.386, 0.653] | 0.714 [0.575, 0.837] | 0.600 [0.472, 0.707] | 0.569 [0.433, 0.698] | 0.762 [0.630, 0.878] | 0.874 [0.772, 0.950] |
| ADDRESS | 47 | 0.474 [0.346, 0.610] | 0.787 [0.667, 0.907] | 0.592 [0.460, 0.723] | 0.718 [0.609, 0.816] | 0.979 [0.930, 1.000] | 0.954 [0.910, 0.985] |
| CREDENTIAL | 40 | 0.808 [0.667, 0.947] | 0.525 [0.375, 0.676] | 0.636 [0.490, 0.765] | 0.923 [0.826, 1.000] | 0.600 [0.447, 0.744] | 0.790 [0.656, 0.903] |
| SSN | 44 | 0.750 [0.632, 0.855] | 0.955 [0.896, 1.000] | 0.840 [0.760, 0.905] | 0.750 [0.632, 0.855] | 0.955 [0.896, 1.000] | 0.983 [0.959, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.462 [0.267, 0.618] | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.889 [0.794, 0.967] |
| GOV_ID | 43 | 0.566 [0.413, 0.717] | 0.698 [0.543, 0.829] | 0.625 [0.483, 0.748] | 0.679 [0.527, 0.820] | 0.814 [0.690, 0.923] | 0.884 [0.795, 0.962] |
| overall | 374 | 0.693 [0.634, 0.745] | 0.741 [0.690, 0.786] | 0.716 [0.662, 0.762] | 0.785 [0.733, 0.832] | 0.807 [0.763, 0.845] | 0.901 [0.870, 0.929] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 2 | 1 | 0 | 5 | 1 | 1 |
| EMAIL | 5 | 5 | 0 | 0 | 5 | 0 |
| PHONE | 5 | 2 | 5 | 0 | 3 | 25 |
| ADDRESS | 1 | 9 | 0 | 17 | 19 | 5 |
| CREDENTIAL | 9 | 3 | 7 | 2 | 3 | 0 |
| SSN | 2 | 0 | 0 | 5 | 0 | 9 |
| CREDIT_CARD | 3 | 0 | 25 | 0 | 0 | 0 |
| GOV_ID | 2 | 5 | 6 | 10 | 6 | 7 |
| overall | 29 | 25 | 43 | 39 | 37 | 47 |
