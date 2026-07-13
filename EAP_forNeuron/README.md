# EAP_forNeuron

`EAP_forNeuron` refines an existing CSAT gradient mask with parameter-level EAP-style attribution. It reads a mask in the current finetuning format:

```python
{parameter_name: torch.BoolTensor(shape == parameter.shape)}
```

Only entries that are already `True` are scored. The output mask keeps the same keys and shapes, and is always a subset of the input mask.

## Method

For a supported linear layer:

```text
Y = X @ W.T
W shape = [out_features, in_features]
```

For candidate weight `W[o, i]`, the score is:

```text
score[W[o, i]] = sum_d ((X_corrupted[d, i] - X_clean[d, i]) * W[o, i] * d loss / dY_clean[d, o])
```

The implementation hooks `nn.Linear` modules, records corrupted inputs, records clean inputs plus output gradients, and accumulates compact score shards only for input-mask candidates.

## Supported Data

The first version supports copied pair CSVs for:

- `bool`
- `gender`
- `ioi_mistral`
- `1_digit_arithmetic`
- `2_digit_arithmetic`
- `3_digit_arithmetic`
- `4_digit_arithmetic`
- `5_digit_arithmetic`

By default the loader reads `EAP_forNeuron/data/{dataset}.csv`.

Arithmetic CSVs use the multiple-choice prompt from finetuning with deterministically shuffled options as `clean`, replace `the correct option` with `the first option` as `corrupted`, leave `corrupted_hard` empty, store the shuffled correct option-letter token id in `correct_idx`, and use the Mistral token id for `A` as `incorrect_idx`.

## Example

```bash
cd /ssd_users/chenhang/CSAT
/home/chenhang/.conda/envs/LLMSFT_BW/bin/python -m EAP_forNeuron.cli \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --tokenizer_name_or_path mistralai/Mistral-7B-v0.1 \
  --mask_path files/masks/gradient/IOI/with_0.2.pt \
  --dataset_name ioi_mistral \
  --data_path EAP_forNeuron/data/ioi_mistral.csv \
  --output_dir files/masks/eap_for_neuron/IOI \
  --output_ratio 0.1 \
  --ratio_base all \
  --batch_size 1 \
  --max_samples 128
```

`--ratio_base all` means the denominator is the total number of supported scoreable neurons. `--ratio_base candidate` means the denominator is the number of `True` entries in the input mask.

## Outputs

```text
output_dir/
  with_{output_ratio}.pt
  summary.json
  metadata.json
  skipped_parameters.json
  selected_scores.pt
  scores.pt              # only when --save_scores is set
```

## Notes

- Default target modules are `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- `lm_head.weight` is disabled by default because it is very large.
- Dense score saving is disabled by default to avoid 7B-scale memory and disk usage.
- The default metric is the project-local `training_losses.task_loss`; `logit_diff` is also available.
