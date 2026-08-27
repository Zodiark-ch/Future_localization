# Future Localization

Official implementation for the paper:

> **Beyond Static Interpretability: Anticipating Post-SFT Mechanisms from Current Parameters for Better Tuning**

Future Localization estimates which model mechanisms will become important after supervised fine-tuning (SFT), while operating from the current model parameters and a lightweight probing checkpoint. The repository supports component-level, neuron-level, and multi-task logical-circuit localization, followed by masked fine-tuning with optional component-wise LoRA.

## Citation

Citation metadata will be added after publication. The placeholder below is intentionally incomplete.

```bibtex
@misc{chen2026staticinterpretabilityanticipatingpostsft,
      title={Beyond Static Interpretability: Anticipating Post-SFT Mechanisms from Pre-SFT Parameters for Better Tuning}, 
      author={Hang Chen and Jiaying Zhu and Wenya Wang},
      year={2026},
      eprint={2608.24482},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.24482}, 
}
```

## Installation

Clone the repository and enter its root directory:

```bash
git clone https://github.com/Zodiark-ch/Future_localization.git
cd Future_localization
```

Two isolated Conda environments are provided. Do not mix their PyTorch or CUDA packages.

### Blackwell GPUs

Use [environment-blackwell.yml](environment-blackwell.yml) for NVIDIA Blackwell GPUs such as RTX PRO 6000 (`sm_120`):

```bash
conda env create -f environment-blackwell.yml
conda activate future-localization-blackwell
```

The verified Blackwell setup uses Python 3.11 and a CUDA 13.0 PyTorch wheel. The runtime check in [src/exec/finetuning_model.py](src/exec/finetuning_model.py) rejects CUDA builds older than 12.8 on `sm_120`, because older builds can fail with low-level device-side assertions even when tensor values are valid.

If the exact `cu130` wheel is unavailable on another machine, install the newest PyTorch CUDA wheel supported by the local NVIDIA driver, but keep the Hugging Face and PEFT versions pinned.

### Pre-Blackwell GPUs

Use [environment-legacy.yml](environment-legacy.yml) on GPUs supported by CUDA 11.8:

```bash
conda env create -f environment-legacy.yml
conda activate future-localization-legacy
```

Do not install pip `torch`, `triton`, or `nvidia-*-cu12` packages into the legacy environment. They can overwrite the Conda PyTorch 2.1.1 stack. The pinned `numpy==1.26.4` and `mkl==2023.1.0` avoid known ABI and linker failures.

Both environments include the main dependencies:

- PyTorch and CUDA runtime
- `transformers==4.37.2`
- `peft==0.10.0`
- `accelerate==0.26.1`
- Hugging Face `datasets` and `huggingface_hub`
- `fastargs`
- `lm-eval`
- NumPy, SciPy, pandas, scikit-learn
- sentencepiece and sentence-transformers
- Graphviz, matplotlib, and seaborn

See [enviroment.md](enviroment.md) for compatibility notes and a GPU verification command.

## Repository structure

```text
Future_localization/
├── EAP_forComponent/          # Component/head-level localization
├── EAP_forNeuron/             # Parameter/neuron-level localization
├── EAP_forLogicalCircuit/     # Multi-task logical-circuit localization
├── src/
│   ├── exec/                  # Training entrypoints
│   ├── model/                 # Training orchestration and LoRA modules
│   ├── finetuning/            # Fine-tuning objectives and mask application
│   ├── dataset/               # Target/pervasiveness datasets
│   ├── metrics/               # Task and language-model evaluation
│   └── modeling_patches.py    # Model-version compatibility patches
├── configs/finetuning/        # Example experiment configurations
├── data/datasets/             # Local arithmetic and cached task data
├── scripts/                   # Dataset construction and analysis utilities
├── environment-blackwell.yml
└── environment-legacy.yml
```

### Localization modules

| Module | Purpose | Main output |
| --- | --- | --- |
| `EAP_forComponent` | Scores projection matrices or individual attention heads and allocates component-specific ranks. | `component_scores.json`, `rank_pattern.json`, `component_mask.pt` |
| `EAP_forNeuron` | Refines an existing parameter mask by scoring only candidate weight entries. | `with_<ratio>.pt`, score metadata |
| `EAP_forLogicalCircuit` | Builds task circuits, fuses forward/reversed paths, assigns logical gates, and supports cross-task conflict analysis. | logical edges, component scores, masks, LoRA artifacts |

### Main Python files

| File | Function |
| --- | --- |
| [src/exec/finetuning_model.py](src/exec/finetuning_model.py) | Main probing, masked fine-tuning, LoRA, and multi-task training entrypoint. |
| [src/model/finetuning.py](src/model/finetuning.py) | Loads the model and datasets, resolves masks/LoRA artifacts, runs single-task or two-stage multi-task training, saves checkpoints, and evaluates tasks. |
| [src/model/lora_utils.py](src/model/lora_utils.py) | Implements standard PEFT LoRA, projection-matrix rank patterns, and custom head-wise/component-wise LoRA. |
| [src/finetuning/FT.py](src/finetuning/FT.py) | Implements `TargetFT`, pervasiveness CE/KL preservation, and L1 variants. |
| [src/finetuning/base.py](src/finetuning/base.py) | Shared loss iteration, reference-model KL, and gradient-mask application. |
| [src/dataset/__init__.py](src/dataset/__init__.py) | Routes target and pervasiveness dataset names to task-specific wrappers. |
| [src/dataset/Base.py](src/dataset/Base.py) | Balances multi-task datasets and builds target/pervasiveness batches. |
| [EAP_forComponent/future_localization.py](EAP_forComponent/future_localization.py) | Computes post-SFT directional corrections with exact HVP or finite differences. |
| [EAP_forComponent/rank_allocator.py](EAP_forComponent/rank_allocator.py) | Converts localization scores into bounded LoRA ranks. |
| [EAP_forNeuron/scorer.py](EAP_forNeuron/scorer.py) | Accumulates parameter-level attribution scores for candidate mask entries. |
| [EAP_forNeuron/selection.py](EAP_forNeuron/selection.py) | Selects the final neuron mask under the requested ratio. |
| [EAP_forLogicalCircuit/runner.py](EAP_forLogicalCircuit/runner.py) | Runs task-level circuit localization and writes logical/mask/LoRA artifacts. |
| [EAP_forLogicalCircuit/conflict_analysis.py](EAP_forLogicalCircuit/conflict_analysis.py) | Combines task artifacts into task-specific, conflict, and all-task component sets. |
| [EAP_forLogicalCircuit/graph_export.py](EAP_forLogicalCircuit/graph_export.py) | Exports sparse circuit JSON/DOT/JPG visualizations. |

## Core workflow

A probing model is a short, low-learning-rate SFT checkpoint. It provides an estimate of a nearby post-SFT parameter state. Future localization uses the base model as the current state and the probing checkpoint as the future state. For full-size Mistral runs, finite differences are the practical default; exact HVP remains available for small-scale checks.

All commands below are run from the repository root. Generated checkpoints and localization artifacts are written under `files/`, which is excluded from Git.

## Example 1: component-level localization on Bool

### 1. Train a probing model

```bash
mkdir -p .cache files/logs

python src/exec/finetuning_model.py \
  --overall.model_name mistralai/Mistral-7B-v0.1 \
  --overall.cache_dir .cache \
  --finetuning.probing 1 \
  --finetuning.finetuning_method TargetFT \
  --finetuning.mask_path none \
  --dataset.target_dataset_name bool \
  --dataset.pervasiveness_dataset_name none \
  --logger.json.root files/logs \
  --logger.name probing-bool-component
```

Probing mode forces one epoch, 1% target data, batch size 1, learning rate `1e-7`, and disables evaluation. The merged Hugging Face checkpoint is saved to:

```text
files/logs/probing-bool-component/probingmodel.pt/
```

### 2. Run future component localization

```bash
python EAP_forComponent/run_eap_for_component.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --tokenizer_name_or_path mistralai/Mistral-7B-v0.1 \
  --cache_dir .cache \
  --dataset_name bool \
  --data_path EAP_forComponent/data/bool.csv \
  --output_dir files/localization/component/bool \
  --attention_granularity head \
  --localization_mode future \
  --future_model_name_or_path files/logs/probing-bool-component/probingmodel.pt \
  --future_hvp_strategy finite_difference \
  --future_finite_difference_epsilon 1e-3 \
  --future_step_k 10 \
  --future_step_k_samples 1 \
  --score_token_mode all_active \
  --score_normalization sum \
  --rank_score_source normalized_abs \
  --head_to_matrix_aggregation mean \
  --min_rank 1 \
  --max_rank 32 \
  --mask_fill_strategy magnitude \
  --mask_min_keep_ratio 0.1 \
  --mask_max_keep_ratio 0.9 \
  --max_samples 512 \
  --batch_size 32
```

The output directory contains the component mask and the head-level scores used to configure component-wise LoRA.

### 3. Fine-tune with the localized mask and head-wise LoRA

```bash
python src/exec/finetuning_model.py \
  --overall.model_name mistralai/Mistral-7B-v0.1 \
  --overall.cache_dir .cache \
  --finetuning.finetuning_method TargetFT \
  --finetuning.target_mask_path files/localization/component/bool/component_mask.pt \
  --finetuning.use_lora 1 \
  --finetuning.lora_mode head \
  --finetuning.target_lora_info_dir files/localization/component/bool \
  --finetuning.lora_target_modules auto \
  --finetuning.lora_head_min_rank 1 \
  --finetuning.lora_head_max_rank 32 \
  --finetuning.lora_head_rank_multiple 1 \
  --finetuning.lora_head_rank_score_source normalized_abs \
  --finetuning.lora_alpha_strategy twice_rank \
  --finetuning.num_epochs 3 \
  --finetuning.lr 1e-5 \
  --dataset.target_dataset_name bool \
  --dataset.pervasiveness_dataset_name none \
  --dataset.batch_size 16 \
  --logger.json.root files/logs \
  --logger.name finetune-bool-component
```

In head-wise LoRA mode, `component_scores.json` defines which attention-head slices and MLP components receive adapters and how ranks are allocated. The mask artifact is loaded through the same training interface for reproducibility with masked runs.

## Example 2: neuron-level localization on Bool

Neuron localization refines an existing candidate mask. It does not require LoRA during the final fine-tuning stage.

### 1. Generate a gradient candidate mask and probing model

Choose a fresh mask path when recomputing the mask; an existing file is loaded rather than regenerated.

```bash
python src/exec/finetuning_model.py \
  --overall.model_name mistralai/Mistral-7B-v0.1 \
  --overall.cache_dir .cache \
  --finetuning.probing 1 \
  --finetuning.finetuning_method TargetFT \
  --finetuning.mask_path files/localization/neuron/bool/candidate/with_0.2.pt \
  --finetuning.mask_score_type gradient \
  --finetuning.mask_ratio 0.2 \
  --finetuning.mask_calibration_samples 128 \
  --dataset.target_dataset_name bool \
  --dataset.pervasiveness_dataset_name none \
  --logger.json.root files/logs \
  --logger.name probing-bool-neuron
```

This command creates both:

```text
files/localization/neuron/bool/candidate/with_0.2.pt
files/logs/probing-bool-neuron/probingmodel.pt/
```

### 2. Refine the candidate mask at neuron/parameter level

```bash
python EAP_forNeuron/run_eap_for_neuron.py \
  --model_name_or_path files/logs/probing-bool-neuron/probingmodel.pt \
  --tokenizer_name_or_path files/logs/probing-bool-neuron/probingmodel.pt \
  --cache_dir .cache \
  --mask_path files/localization/neuron/bool/candidate/with_0.2.pt \
  --dataset_name bool \
  --data_path EAP_forNeuron/data/bool.csv \
  --output_dir files/localization/neuron/bool/refined \
  --output_ratio 0.1 \
  --ratio_base all \
  --score_token_mode label_position \
  --score_abs true \
  --max_samples 128 \
  --batch_size 1
```

The refined mask is saved as:

```text
files/localization/neuron/bool/refined/with_0.1.pt
```

### 3. Run ordinary masked fine-tuning without LoRA

```bash
python src/exec/finetuning_model.py \
  --overall.model_name mistralai/Mistral-7B-v0.1 \
  --overall.cache_dir .cache \
  --finetuning.finetuning_method TargetFT \
  --finetuning.target_mask_path files/localization/neuron/bool/refined/with_0.1.pt \
  --finetuning.use_lora 0 \
  --finetuning.num_epochs 3 \
  --finetuning.lr 1e-5 \
  --dataset.target_dataset_name bool \
  --dataset.pervasiveness_dataset_name none \
  --dataset.batch_size 16 \
  --logger.json.root files/logs \
  --logger.name finetune-bool-neuron
```

## Example 3: multi-task logical-circuit localization

This example treats Bool as the target task and IOI, SST-2, and 5-digit arithmetic as pervasiveness tasks.

### 1. Train one probing model per task

```bash
probe_task () {
  dataset_name="$1"
  run_name="$2"
  python src/exec/finetuning_model.py \
    --overall.model_name mistralai/Mistral-7B-v0.1 \
    --overall.cache_dir .cache \
    --finetuning.probing 1 \
    --finetuning.finetuning_method TargetFT \
    --finetuning.mask_path none \
    --dataset.target_dataset_name "$dataset_name" \
    --dataset.pervasiveness_dataset_name none \
    --logger.json.root files/logs \
    --logger.name "$run_name"
}

probe_task bool probing-logical-bool
probe_task IOI probing-logical-ioi
probe_task sst2 probing-logical-sst2
probe_task 5_digit_arithmetic probing-logical-5digit
```

### 2. Localize one logical circuit per task

```bash
localize_task () {
  pair_dataset="$1"
  probing_run="$2"
  output_name="$3"
  python EAP_forLogicalCircuit/run_eap_for_logical_circuit.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --tokenizer_name_or_path mistralai/Mistral-7B-v0.1 \
    --cache_dir .cache \
    --dataset_name "$pair_dataset" \
    --output_dir "files/localization/logical/$output_name" \
    --localization_mode future \
    --future_model_name_or_path "files/logs/$probing_run/probingmodel.pt" \
    --future_hvp_strategy finite_difference \
    --future_finite_difference_epsilon 1e-3 \
    --future_step_k 1.0 \
    --future_step_k_samples 1 \
    --circuit_construction node_induced \
    --component_granularity head \
    --node_topn 500 \
    --rank_score_source normalized_abs \
    --min_rank 1 \
    --max_rank 32 \
    --mask_fill_strategy magnitude \
    --mask_min_keep_ratio 0.1 \
    --mask_max_keep_ratio 0.9 \
    --graph false \
    --max_samples 128 \
    --batch_size 1
}

localize_task bool probing-logical-bool bool
localize_task ioi_mistral probing-logical-ioi IOI
localize_task sst2 probing-logical-sst2 sst2
localize_task 5_digit_arithmetic probing-logical-5digit 5_digit_arithmetic
```

### 3. Run conflict analysis

```bash
python EAP_forLogicalCircuit/run_conflict_analysis.py \
  --task_artifact_dirs "bool=files/localization/logical/bool,IOI=files/localization/logical/IOI,sst2=files/localization/logical/sst2,5_digit_arithmetic=files/localization/logical/5_digit_arithmetic" \
  --output_dir files/localization/logical/conflict_analysis \
  --min_rank 1 \
  --max_rank 32 \
  --head_to_matrix_aggregation mean \
  --rank_score_source normalized_abs \
  --mask_fill_strategy magnitude \
  --mask_min_keep_ratio 0.1 \
  --mask_max_keep_ratio 0.9 \
  --write_dense_masks
```

Important outputs are:

```text
files/localization/logical/conflict_analysis/
├── task_all_components/<task>/
├── conflict_components/
├── all_task_components/
└── conflict_summary.json
```

### 4. Run two-stage multi-task head-wise LoRA fine-tuning

```bash
python src/exec/finetuning_model.py \
  --overall.model_name mistralai/Mistral-7B-v0.1 \
  --overall.cache_dir .cache \
  --finetuning.finetuning_method TargetFT+PervasivenessFT \
  --finetuning.use_lora 1 \
  --finetuning.lora_mode head \
  --finetuning.lora_target_modules auto \
  --finetuning.lora_head_min_rank 1 \
  --finetuning.lora_head_max_rank 32 \
  --finetuning.lora_head_rank_score_source normalized_abs \
  --finetuning.lora_alpha_strategy twice_rank \
  --finetuning.target_mask_path files/localization/logical/bool/component_mask.pt \
  --finetuning.pervasiveness_mask_paths "files/localization/logical/IOI/component_mask.pt,files/localization/logical/sst2/component_mask.pt,files/localization/logical/5_digit_arithmetic/component_mask.pt" \
  --finetuning.conflict_mask_path files/localization/logical/conflict_analysis/conflict_components/component_mask.pt \
  --finetuning.all_component_mask_path files/localization/logical/conflict_analysis/all_task_components/component_mask.pt \
  --finetuning.target_lora_info_dir files/localization/logical/bool \
  --finetuning.pervasiveness_lora_info_dirs "files/localization/logical/IOI,files/localization/logical/sst2,files/localization/logical/5_digit_arithmetic" \
  --finetuning.conflict_lora_info_dir files/localization/logical/conflict_analysis/conflict_components \
  --finetuning.all_component_lora_info_dir files/localization/logical/conflict_analysis/all_task_components \
  --finetuning.multi_task_schedule two_stage_alternating \
  --finetuning.stage1_num_epochs 1 \
  --finetuning.stage2_num_epochs 1 \
  --dataset.target_dataset_name bool \
  --dataset.pervasiveness_dataset_name "IOI,sst2,5_digit_arithmetic" \
  --dataset.batch_size 1 \
  --logger.json.root files/logs \
  --logger.name finetune-logical-multitask
```

Stage 1 alternates target and pervasiveness tasks with their task-specific masks and active LoRA modules. Stage 2 jointly optimizes all tasks with the `all_task_components` mask and LoRA configuration. Conflict artifacts are loaded and recorded for analysis, while the shared all-task artifacts control the current stage-2 update set.

## Supported localization datasets

Due to code safety policies, this public repository includes only examples that can run with Mistral-7B on a single GPU. The following datasets are provided as stable single-GPU examples:

- Bool
- Gender pronoun prediction
- IOI
- SST-2 where supported
- 2- through 5-digit arithmetic

## Outputs and checkpoints

- Fine-tuning runs are stored under `files/logs/<run-name>/`.
- Probing checkpoints are stored in `probingmodel.pt/` within the run directory.
- Standard checkpoints are stored under `checkpoints/`.
- Final evaluation summaries are written to `eval/finetuning_eval_summary.json`.
- Localization masks and rank artifacts are written to the selected `--output_dir`.

These generated artifacts can be large and are excluded from version control.

## License

See [LICENSE](LICENSE).
