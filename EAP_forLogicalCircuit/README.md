# EAP_forLogicalCircuit

`EAP_forLogicalCircuit` computes edge-level attribution and builds logical circuits from pair CSV datasets.

## Supported Data

Default CSV lookup first checks `EAP_forLogicalCircuit/data/{dataset}.csv`, then falls back to `EAP_forComponent` or `EAP-IG` defaults when available. Supported dataset names include:

- `bool`
- `gender`
- `ioi_mistral`
- `sst2`
- `1_digit_arithmetic`
- `2_digit_arithmetic`
- `3_digit_arithmetic`
- `4_digit_arithmetic`
- `5_digit_arithmetic`

Arithmetic CSVs use the finetuning multiple-choice prompt with deterministically shuffled options as `clean`, replace `the correct option` with `the first option` as `corrupted`, leave `corrupted_hard` empty, store the shuffled correct option-letter token id in `correct_idx`, and use the Mistral token id for `A` as `incorrect_idx`.