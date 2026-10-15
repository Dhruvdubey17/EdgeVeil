# Labeling guide

These rules define the entity spans for the hand-labeled test set. The training labels come from OpenPII through `edgeveil.dataset.openpii.map_entities`, which applies the same rules, so a model trained on one is scored fairly against the other.

Label in Label Studio with `docs/label_studio.xml`. Select character spans.

## Rules for every type

- A span covers the value and nothing else: no surrounding spaces, quotes, brackets, or sentence punctuation.
- Spans never overlap or nest.
- Label every occurrence, including repeats of the same value.
- Do not label dates, ages, gender, sex, or titles. They are outside the entity set.
- Field labels and label words never go inside a span: `ZIP`, `postal code`, `postcode`, `Last Name:`, `Surname:`, and the like.

## PERSON

- A person's given name, surname, or both. A given name or surname on its own is PERSON.
- Name parts form one span when the text between them is whitespace, optionally with connectors that each have whitespace on both sides. The connectors are a middle initial (one capital letter, with or without a period), the lowercase particles `van`, `von`, `de`, `da`, `del`, `la`, `le`, `bin`, `binti`, and `al`, and `s/o` or `d/o`. So `Ada Quill`, `Ada B. Quill`, `Ada van Quill`, and `Ada s/o Quill` are each one span.
- Any other word between parts splits the name: in `Ada der Quill` or `Ada Van Quill`, label `Ada` and `Quill` separately.
- A comma always splits, because name parts never merge across a comma: `Smith, John` is two PERSON spans, `Smith` and `John`.
- A form label between parts splits them: in `First Name: Ada` followed by `Last Name: Quill` on the next line, label `Ada` and `Quill` separately.
- Titles and honorifics stay outside the span: in `Dr. Ada Quill`, label `Ada Quill`.
- A possessive suffix stays outside: in `Ada's`, label `Ada`.
- Usernames and handles are not PERSON.

## ADDRESS

- Address parts are a street name, a building number (house, apartment, or unit number), a city, and a ZIP or postal code.
- Consecutive address parts form one span when the text between them is at most 16 characters and consists of separators and allowed words.
  - Separators: spaces, newlines, commas, periods, `#`, hyphens, en dashes, and em dashes.
  - Allowed words: an uppercase two-letter region code such as `CO`, and the unit words `Apt`, `Apartment`, `Suite`, `Ste`, `Unit`, `Flat`, `Floor`, `Fl`, `Room`, `Rm`, `Bldg`, `Block`, `Blk`, `Level`, and `No`, in any case, with or without a trailing period.
  - Each allowed word has a separator before it. The last one may touch the next part, as in `Apt4`.
  - Examples of one span: `12 Imaginary Lane, Nowhere City, ZZ 00000`, `Imaginary Lane Apt. 4`, `Nowhere City – 00000`.
- A span is ADDRESS only if it contains a street, a building number, or a ZIP code. A city on its own, or a city followed by a region code, is not labeled.
- Any other text between parts ends the span. In `12 Imaginary Lane, Nowhere City, Colorado 00000`, the spelled-out state ends the span, so the labels are `12 Imaginary Lane, Nowhere City` and `00000`.
- A city joined to the rest of an address only through a label phrase (`ZIP`, `postal code`, `postcode`) or through ` in ` counts as a lone city and is not labeled. Two examples:
  - `12 Imaginary Lane in Nowhere City`: label `12 Imaginary Lane` only.
  - `Nowhere City (ZIP 00000)`: label `00000` only.
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
- The expiry date and the security code (CVV, CVC) are not labeled. The expiry is a date, and security codes are outside the entity set.

## GOV_ID

- An ID card, passport, driver's license, or tax identification number. Label the number only, not words such as `Passport No.`

## Hard negatives

Include snippets with text that looks sensitive but is not one of the types. Leave all of it unlabeled:

- Lone cities, such as `I moved to Nowhere City.`
- Dates.
- Order and invoice numbers.
- Version strings, such as `v2.14.1`.
- Prices.
- Commit SHAs.
- UUIDs.

## Test set size

Target at least 40 instances of each of the 8 types, 320 in all. PERSON, EMAIL, PHONE, and ADDRESS reach 40 without effort. The four rarer types (CREDENTIAL, SSN, CREDIT_CARD, GOV_ID) set the size: about 160 snippets that each carry one of them, most of which also hold a common type, plus about 40 snippets with no entities or with hard negatives only. The target is 200 snippets.

Use only synthetic values: emails on `example.com` or `.test`, phone numbers from 555-0100 to 555-0199, SSNs with area numbers 000 or 900 to 999, card numbers from processor test ranges, and addresses that are plainly fictional.
