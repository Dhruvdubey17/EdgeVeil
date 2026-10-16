Detection: model only. distilbert/distilbert-base-cased, int4-per-channel, coreml-ane, compute units cpu-and-ne. Test set: 235 snippets. Intervals: 1000 bootstrap resamples by snippet. Source: `bench/results/accuracy/distilbert-int4-per-channel-coreml-ane-cpu-and-ne.json`.

Strict precision, recall, and F1 are the headline. Overlap scores count a span as matched if it overlaps a span of the same type. Char recall is the share of gold span characters covered by any emitted span.

| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Overlap precision | Overlap recall | Char recall |
|---|---|---|---|---|---|---|---|
| PERSON | 75 | 0.897 [0.823, 0.962] | 0.933 [0.870, 0.986] | 0.915 [0.861, 0.961] | 0.910 [0.841, 0.972] | 0.947 [0.889, 0.987] | 0.950 [0.891, 0.993] |
| EMAIL | 43 | 0.900 [0.786, 1.000] | 0.628 [0.490, 0.783] | 0.740 [0.615, 0.857] | 1.000 [1.000, 1.000] | 0.698 [0.558, 0.833] | 0.680 [0.539, 0.814] |
| PHONE | 42 | 0.576 [0.440, 0.712] | 0.809 [0.688, 0.923] | 0.673 [0.553, 0.781] | 0.644 [0.509, 0.769] | 0.881 [0.775, 0.975] | 0.941 [0.867, 0.995] |
| ADDRESS | 47 | 0.444 [0.327, 0.567] | 0.766 [0.650, 0.880] | 0.562 [0.442, 0.678] | 0.691 [0.582, 0.786] | 0.957 [0.894, 1.000] | 0.944 [0.896, 0.978] |
| CREDENTIAL | 40 | 0.654 [0.467, 0.840] | 0.425 [0.277, 0.585] | 0.515 [0.356, 0.667] | 0.962 [0.880, 1.000] | 0.600 [0.455, 0.756] | 0.746 [0.599, 0.871] |
| SSN | 44 | 0.764 [0.634, 0.875] | 0.955 [0.896, 1.000] | 0.849 [0.755, 0.914] | 0.764 [0.634, 0.875] | 0.955 [0.896, 1.000] | 0.983 [0.959, 1.000] |
| CREDIT_CARD | 40 | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.462 [0.267, 0.618] | 1.000 [1.000, 1.000] | 0.300 [0.154, 0.447] | 0.835 [0.711, 0.940] |
| GOV_ID | 43 | 0.628 [0.462, 0.789] | 0.628 [0.464, 0.775] | 0.628 [0.476, 0.753] | 0.791 [0.658, 0.907] | 0.744 [0.604, 0.864] | 0.833 [0.726, 0.930] |
| overall | 374 | 0.690 [0.628, 0.745] | 0.709 [0.657, 0.758] | 0.699 [0.644, 0.749] | 0.802 [0.753, 0.839] | 0.783 [0.741, 0.822] | 0.862 [0.826, 0.898] |

Strict errors. A miss is a gold span without an exact match; a false positive (FP) is an emitted span without one. Boundary: overlaps a span of the same type. Wrong type: overlaps only spans of other types.

| Type | Missed | Boundary miss | Wrong-type miss | Spurious FP | Boundary FP | Wrong-type FP |
|---|---|---|---|---|---|---|
| PERSON | 4 | 1 | 0 | 5 | 1 | 2 |
| EMAIL | 13 | 3 | 0 | 0 | 3 | 0 |
| PHONE | 2 | 3 | 3 | 0 | 4 | 21 |
| ADDRESS | 2 | 9 | 0 | 17 | 20 | 8 |
| CREDENTIAL | 10 | 7 | 6 | 1 | 8 | 0 |
| SSN | 2 | 0 | 0 | 4 | 0 | 9 |
| CREDIT_CARD | 6 | 0 | 22 | 0 | 0 | 0 |
| GOV_ID | 4 | 5 | 7 | 2 | 7 | 7 |
| overall | 43 | 28 | 38 | 29 | 43 | 47 |
