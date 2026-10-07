# Model card

No model has been trained yet. This card is filled in as training and evaluation runs produce results.

## Intended use

Detecting personal data in English text on the device, so EdgeVeil can replace it with placeholders before a prompt goes to a cloud LLM.

## Entity types

PERSON, EMAIL, PHONE, ADDRESS, and CREDENTIAL, plus the optional SSN, CREDIT_CARD, and GOV_ID.

## Training data

The model is trained on [ai4privacy/pii-masking-openpii-1.5m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) by Ai4Privacy / Ai Suisse SA, used under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). EdgeVeil keeps only the English rows, remaps the labels to the entity types above, and adds synthetic credentials. `docs/data_card.md` lists every change.

All CREDENTIAL training data is synthetic. The passwords, API keys, and tokens come from a seeded generator, not from real leaked secrets.

## Results

Not measured yet.

## Limitations

- English only.
- Usernames, IP addresses, URLs, and account numbers are out of scope. The model is not trained to detect them, and they may appear unlabeled in the training data.
- Dates, ages, gender, and titles are not masked.
- A city on its own is not masked.
- CREDENTIAL detection is learned from generated examples, so secrets in formats the generator does not produce may be missed. The validators are meant to cover part of that gap.
