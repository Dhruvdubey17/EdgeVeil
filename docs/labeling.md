# Labeling guide

These rules define the entity spans for the hand-labeled test set. The training labels come from OpenPII through `edgeveil.dataset.openpii.map_entities`, which applies the same rules, so a model trained on one is scored fairly against the other.

Label in Label Studio with `docs/label_studio.xml`. Select character spans.

## Rules for every type

- A span covers the value and nothing else: no surrounding spaces, quotes, brackets, or sentence punctuation.
- Spans never overlap or nest.
- Label every occurrence, including repeats of the same value.
- Do not label dates, ages, gender, sex, or titles. They are outside the entity set.

## PERSON

- A person's given name, surname, or both. A given name or surname on its own is PERSON.
- Name parts separated only by whitespace form one span: `Ada Quill` is one span.
- A comma always splits: `Quill, Ada` is two PERSON spans, `Quill` and `Ada`.
- Titles and honorifics stay outside the span: in `Dr. Ada Quill`, label `Ada Quill`.
- A possessive suffix stays outside: in `Ada's`, label `Ada`.
- Usernames and handles are not PERSON.

## ADDRESS

- Address parts are a street name, a building number (house, apartment, or unit number), a city, and a ZIP or postal code.
- Consecutive address parts form one span when the text between them contains only spaces, commas, periods, `#`, or newlines. That text may also hold one uppercase two-letter region code that has a separator on each side, as in `, CO `.
- A span is ADDRESS only if it contains a street, a building number, or a ZIP code. A city on its own, or a city followed by a region code, is not labeled.
- Any other text between parts ends the span. In `12 Imaginary Lane, Nowhere City, Colorado 00000`, the spelled-out state ends the span, so the labels are `12 Imaginary Lane, Nowhere City` and `00000`.
- A region code after the last part stays outside: in `12 Imaginary Lane, Nowhere City, ZZ`, label `12 Imaginary Lane, Nowhere City`.
- Countries are not labeled.

## EMAIL

- The whole address, `local@domain`. Leave out `mailto:` and angle brackets.

## PHONE

- The number as written, including a leading `+` and country code, parentheses, spaces, dots, and dashes. Leave out labels such as `tel:` or `Phone:`.

## CREDENTIAL

- Passwords, API keys, access tokens, and other secret values.
- Label only the secret: leave out the field name (`password=`, `api_key:`), the `Bearer` scheme word, and surrounding quotes.
- Commit SHAs, UUIDs, checksums, and base64 data that is not a secret are not CREDENTIAL. Leave them unlabeled.
- Usernames are not labeled.

## SSN

- A US Social Security number or another national social insurance number, with its dashes or spaces.

## CREDIT_CARD

- A full payment card number, with its spaces or dashes. A partial number such as the last four digits is not labeled.

## GOV_ID

- An ID card, passport, driver's license, or tax identification number. Label the number only, not words such as `Passport No.`

## Test set size

Target at least 40 instances of each of the 8 types, 320 in all. PERSON, EMAIL, PHONE, and ADDRESS reach 40 without effort. The four rarer types (CREDENTIAL, SSN, CREDIT_CARD, GOV_ID) set the size: about 160 snippets that each carry one of them, most of which also hold a common type, plus about 40 snippets with no entities or with hard negatives only. That comes to about 200 snippets.

Use only synthetic values: emails on `example.com` or `.test`, phone numbers from 555-0100 to 555-0199, SSNs with area numbers 000 or 900 to 999, card numbers from processor test ranges, and addresses that are plainly fictional.
