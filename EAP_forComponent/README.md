# EAP_forComponent

`EAP_forComponent` computes EAP-style attribution scores for coarse Mistral/LLaMA components. It targets:

- attention `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP `gate_proj`, `up_proj`, `down_proj`

The tool writes both a PEFT-compatible `rank_pattern.json` and a CSAT-compatible `component_mask.pt` control mask.

## Attribution Modes

Attention supports two granularities:

- `projection_matrix`: one score for a full projection matrix.
- `head`: one score for each attention head slice inside `q/k/v/o`; MLP components remain matrix-level.

Token aggregation supports two modes:

- `all_active`: all non-padding token positions.
- `label_position`: only the final label token position.

Localization supports two modes:

- `current`: computes `(clean_activation - corrupted_activation) dot grad(metric, corrupted_activation)`.
- `future`: loads a merged finetuned checkpoint as `theta'`, builds `Delta_theta = theta' - theta`, and adds the trapezoid directional correction with `--future_step_k`.

Standard PEFT LoRA cannot assign different ranks to different heads inside the same `nn.Linear`, so head-level scores are aggregated back to matrix-level ranks with `--head_to_matrix_aggregation`.

Supported pair CSV datasets are `bool`, `gender`, `ioi_mistral`, `sst2`, and `1_digit_arithmetic` through `5_digit_arithmetic`. Arithmetic CSVs use the finetuning multiple-choice prompt with deterministically shuffled options as `clean`, replace `the correct option` with `the first option` as `corrupted`, leave `corrupted_hard` empty, store the shuffled correct option-letter token id in `correct_idx`, and use the Mistral token id for `A` as `incorrect_idx`.

## Example

```bash
cd /ssd_users/chenhang/CSAT
/home/chenhang/.conda/envs/LLMSFT_BW/bin/python EAP_forComponent/run_eap_for_component.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --tokenizer_name_or_path mistralai/Mistral-7B-v0.1 \
  --cache_dir /home/chenhang/CSAT/.cache \
  --dataset_name ioi_mistral \
  --data_path EAP_forComponent/data/ioi_mistral.csv \
  --output_dir files/component_scores/ioi_mistral \
  --attention_granularity projection_matrix \
  --score_token_mode all_active \
  --score_normalization sum \
  --min_rank 0 \
  --max_rank 32 \
  --batch_size 1 \
  --max_samples 128
```

Head-level run:

```bash
--attention_granularity head --head_to_matrix_aggregation mean
```

Future localization run:

```bash
--localization_mode future \
--future_model_name_or_path /path/to/merged-finetuned-model \
--future_step_k 1.0 \
--future_hvp_strategy finite_difference
```

`finite_difference` is the practical default for full Mistral runs: it computes all component directional scores with four score passes. Exact `hvp` is kept for small/debug runs, but it requires one second-order parameter-gradient call per component and is much slower at 7B scale.

## Outputs

```text
output_dir/
  component_scores.json
  component_scores.pt
  rank_pattern.json
  lora_allocation.json
  component_mask.pt
  summary.json
```

`component_mask.pt` has the existing mask format:

```python
{parameter_name: torch.BoolTensor(shape == parameter.shape)}
```

## LoRA Finetuning

The finetuning entrypoint can consume this output directory directly when `--finetuning.use_lora 1` is set.

Projection-matrix LoRA uses PEFT `rank_pattern.json`:

```bash
/home/chenhang/.conda/envs/LLMSFT_BW/bin/python src/exec/finetuning_model.py \
  --finetuning.use_lora 1 \
  --finetuning.lora_mode projection_matrix \
  --finetuning.lora_info_dir files/component_scores/ioi_mistral \
  --finetuning.lora_target_modules auto
```

Head-wise LoRA uses `component_scores.json` and wraps each target `nn.Linear` with a component-wise LoRA module. For `q/k/v`, each head updates its output row slice; for `o_proj`, each head updates the full output through its input column slice; MLP modules remain full-matrix components.

```bash
/home/chenhang/.conda/envs/LLMSFT_BW/bin/python src/exec/finetuning_model.py \
  --finetuning.use_lora 1 \
  --finetuning.lora_mode head \
  --finetuning.lora_info_dir files/component_scores/ioi_mistral \
  --finetuning.lora_head_min_rank 0 \
  --finetuning.lora_head_max_rank 32 \
  --finetuning.lora_target_modules auto
```

`standard` keeps the previous fixed PEFT LoRA behavior and defaults to `q_proj,v_proj` with `--finetuning.lora_default_rank 8`.
