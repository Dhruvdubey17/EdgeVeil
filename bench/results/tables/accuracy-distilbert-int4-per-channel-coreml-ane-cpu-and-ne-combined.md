Detection: model and validators. distilbert/distilbert-base-cased, int4-per-channel, coreml-ane, compute units cpu-and-ne. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int4-per-channel-coreml-ane-cpu-and-ne-combined.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.910 [0.841, 0.972] | 0.947 [0.889, 0.987] | 0.928 [0.883, 0.966] | 0.910 [0.841, 0.972] | 0.947 [0.889, 0.987] | 0.950 [0.891, 0.993] |
| EMAIL | 43 | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 0.977 [0.940, 1.000] | 0.956 [0.886, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| PHONE | 42 | 0.841 [0.702, 0.947] | 0.881 [0.773, 0.974] | 0.861 [0.740, 0.947] | 0.932 [0.830, 1.000] | 0.976 [0.921, 1.000] | 1.000 [1.000, 1.000] |
| ADDRESS | 47 | 0.444 [0.327, 0.567] | 0.766 [0.650, 0.880] | 0.562 [0.442, 0.678] | 0.691 [0.582, 0.786] | 0.957 [0.894, 1.000] | 0.944 [0.896, 0.978] |
| CREDENTIAL | 40 | 0.786 [0.640, 0.929] | 0.550 [0.400, 0.700] | 0.647 [0.510, 0.759] | 0.857 [0.731, 0.967] | 0.600 [0.459, 0.750] | 0.803 [0.675, 0.906] |
| SSN | 44 | 0.764 [0.634, 0.875] | 0.955 [0.896, 1.000] | 0.849 [0.755, 0.914] | 0.764 [0.634, 0.875] | 0.955 [0.896, 1.000] | 0.983 [0.959, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| GOV_ID | 43 | 0.730 [0.548, 0.900] | 0.628 [0.464, 0.775] | 0.675 [0.518, 0.806] | 0.919 [0.816, 1.000] | 0.744 [0.604, 0.864] | 0.833 [0.726, 0.930] |
| overall | 374 | 0.779 [0.725, 0.830] | 0.850 [0.814, 0.886] | 0.813 [0.770, 0.852] | 0.860 [0.822, 0.892] | 0.904 [0.874, 0.931] | 0.936 [0.913, 0.958] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 4 | 0 | 0 | 5 | 0 | 2 |
| EMAIL | 0 | 0 | 0 | 0 | 0 | 2 |
| PHONE | 0 | 4 | 1 | 0 | 4 | 3 |
| ADDRESS | 2 | 9 | 0 | 17 | 20 | 8 |
| CREDENTIAL | 8 | 2 | 8 | 4 | 2 | 0 |
| SSN | 2 | 0 | 0 | 4 | 0 | 9 |
| CREDIT_CARD | 0 | 0 | 0 | 0 | 0 | 0 |
| GOV_ID | 4 | 5 | 7 | 2 | 7 | 1 |
| overall | 20 | 20 | 16 | 32 | 33 | 25 |
