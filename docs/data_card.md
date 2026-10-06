# Data card

## Training and validation data

| Field | Value |
|---|---|
| Source | [ai4privacy/pii-masking-openpii-1.5m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) |
| Revision | `a785eb528e28be2693c3718a27e066970de5dadb` |
| Downloaded | 2026-10-06 (23:51 UTC) |
| License | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |
| Attribution | Ai4Privacy / Ai Suisse SA |
| Content | Synthetic text with synthetic personal data, per the dataset card |

The OpenPII 1.5M dataset by Ai4Privacy / Ai Suisse SA is used under CC-BY-4.0. EdgeVeil changes it as follows:

- English filter. Only rows whose `language` field is `en` are kept, in both the train and validation splits.
- Cleaning. Rows where any `privacy_mask` entry's offsets do not reproduce its stored value are dropped, as are rows with overlapping entries and rows carrying labels outside the 19 on the dataset card. Whitespace at the edges of an entity is trimmed. Exact duplicate texts are removed across splits, keeping the validation copy.
- Label remapping. The 19 OpenPII labels are mapped to EdgeVeil's entity types by the rules in `docs/labeling.md`: given names and surnames become PERSON, street, building number, city, and ZIP code groups become ADDRESS, ID card, passport, driver's license, and tax numbers become GOV_ID, and dates, ages, gender, sex, and titles are dropped.
- Added synthetic credentials (planned; the generator is not built yet). CREDENTIAL does not exist in OpenPII. A seeded generator inserts fake passwords, API keys, and tokens into a small share of training rows, along with unlabeled hard negatives such as commit SHAs and UUIDs. All CREDENTIAL training data is synthetic.

Row and span counts come from `edgeveil data prepare`, which writes them to `data/openpii/clean/report.json`. Nothing under `data/` is committed.

## Test data

A hand-labeled set of about 200 snippets written for this project, labeled in Label Studio under `docs/labeling.md`. It contains only self-written synthetic values, including hand-written credentials. It is never used for training or threshold tuning.
