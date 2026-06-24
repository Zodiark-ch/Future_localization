# CSAT 修改需求文档

## 第一步：当前项目现有功能盘点

本文档只完成第一步：梳理当前 CSAT 项目已经具备的功能、主流程、关键类/函数、输入输出变量、数据来源以及从模型初始化到最终评估的执行链路。第二步“需要修改的功能”后续再补充。

本次梳理范围主要是 `src/` 下的训练、数据、unlearning、mask、评估、日志与优化器代码；`Edge-Pruning/` 只作为非端到端生成 mask 的外部工具记录其产物关系，不展开分析内部实现。

---

## 1. 项目整体功能

当前项目实现的是 LLM unlearning 任务：

1. 加载一个 HuggingFace causal language model，例如 `mistralai/Mistral-7B-v0.1` 或 `HuggingFaceH4/zephyr-7b-beta`。
2. 构造两类训练数据：
   - `forget_dataset`：需要遗忘或拒答的知识样本。
   - `retain_dataset`：需要在微调中保留能力的样本。
3. 加载或生成 mask，用来限定哪些神经元参数或线性层权重位置允许更新，哪些位置被冻结。
4. 根据配置选择 unlearning loss，例如 FT、GA、GA+FT、KL、CL、CL+KL、RL、NPO、NPO+FT 等。
5. 在 `forget` 与 `retain` 数据组合上训练模型。
6. 每个 epoch 保存 checkpoint，并在 retain/test/downstream 数据集上评估保留能力。
7. 训练结束后保存最终 checkpoint，并调用若干评估函数衡量遗忘效果和保留效果，包括本项目自定义评估与 `lm-evaluation-harness`。

---

## 2. 顶层执行入口

### 2.1 `src/exec/unlearn_model_conlict.py`

这是当前 unlearning 主入口。执行方式：

```bash
python src/exec/unlearn_model_conlict.py
```

也可以通过 fastargs 机制传入配置文件或命令行参数。代码注释说明优先级为：命令行参数 > config 文件 > 环境变量 > 默认值。

#### 顶层环境变量

入口文件在 import 后立即设置：

| 变量 | 值 | 作用 |
| --- | --- | --- |
| `CUDA_LAUNCH_BLOCKING` | `1` | CUDA 同步报错，方便定位 GPU 错误。 |
| `TORCH_USE_CUDA_DSA` | `1` | 启用 CUDA device-side assert 辅助调试。 |
| `CUDA_VISIBLE_DEVICES` | `0,1` | 默认只使用 0、1 两张 GPU。若 `overall.use_cpu=True`，会清空该变量。 |
| `sys.path` | 加入 `src` | 允许 `import model...`、`import dataset...` 等相对项目模块。 |

#### 配置分组与输入变量

`fastargs.Section` 定义了以下配置。

##### `overall`

| 参数 | 类型 | 默认值 | 作用 | 来源 |
| --- | --- | --- | --- | --- |
| `model_name` | `str` | `mistralai/Mistral-7B-v0.1` | HuggingFace 模型名或本地 checkpoint 路径。 | 默认值、config、命令行。 |
| `logger` | `json` 或 `none` | `json` | 选择日志后端，动态导入 `loggers.{logger}_`。 | 默认值、config、命令行。 |
| `cache_dir` | folder | `/home/chenhang/CSAT/.cache` | HuggingFace 模型与数据缓存目录。 | 默认值、config、命令行。 |
| `seed` | `int` | `0` | 随机种子。 | 默认值、config、命令行。 |
| `use_cpu` | bool-as-int | `0` | CPU 调试模式。为 1 时加载 float32 CPU 模型，训练和评估极慢。 | 默认值、config、命令行。 |

##### `unlearn`

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `safe_unlearn_method` | enum | `FT` | safe mask 对应的 unlearner 方法。当前会创建 safe unlearner，但当前手写训练循环实际固定使用 conflict unlearner。 |
| `conflict_unlearn_method` | enum | `FT` | conflict mask 对应的实际训练 unlearner 方法。 |
| `safe_mask_path` | `str` | `/home/chenhang/CSAT/wanda/zephyr/with_0.0.pt` | safe mask 文件路径。存在则加载，不存在则按路径推断 score_type 并生成。 |
| `conflict_mask_path` | `str` | `/home/chenhang/CSAT/wanda/zephyr/with_0.0.pt` | conflict mask 文件路径。当前训练使用该 mask。 |
| `alternate_frequency` | `int` | `1` | 原设计用于 safe/conflict unlearner 交替频率。当前交替逻辑被注释，训练只用 conflict unlearner。 |
| `num_epochs` | `int` | `6` | 训练 epoch 数。 |
| `lr` | `float` | `1e-5` | 优化器学习率。 |
| `weight_decay` | `float` | `0.1` | 权重衰减。 |
| `gradient_accumulation_steps` | `int` | `4` | 梯度累积步数。 |
| `max_grad_norm` | `float` | `1.0` | 梯度裁剪上限。 |
| `task_name` | enum | `downstream` | 控制最终评估分支。当前实现中主要处理 `downstream`。 |
| `sophia` | bool-as-int | `False` | 是否使用 SophiaG 优化器。否则使用 AdamW。 |
| `p` | `float` | `0.01` | mask 生成相关参数，传给 `GenerateMask`。 |
| `q` | `float` | `0.01` | mask 生成相关参数，传给 `GenerateMask`。 |
| `resume_path` | folder or `None` | `None` | 若提供，跳过训练，直接加载该路径做评估。 |
| `max_steps` | `int` | `-1` | 最大训练步数。-1 时由数据集长度、epoch、batch、累积步数推导。 |
| `use_lora` | bool-as-int | `False` | 是否用 PEFT LoRA 包装模型。 |
| `mu` | `float` | `1e-6` | mask 生成里高阶近似或 SNIP 相关参数。 |

##### 条件子配置

| 子配置 | 触发条件 | 参数 | 作用 |
| --- | --- | --- | --- |
| `unlearn.sophia_params` | `unlearn.sophia=True` | `betas_low`, `betas_high`, `rho` | SophiaG 优化器超参数。 |
| `unlearn.NPO+FT` | safe 或 conflict 方法为 `NPO+FT` | `gamma` | retain loss 权重。 |
| `unlearn.l1_sparse` | 方法为 `l1_sparse` | `alpha` | L1 正则权重。注意当前 `get_unlearn_method` 中名字是 `l1sparse`，与配置名不完全一致。 |
| `unlearn.CL+KL` | 方法为 `CL+KL` | `gamma` | CL loss 权重。 |
| `unlearn.GA+KL` | 方法为 `GA+KL` | `gamma` | GA loss 权重。 |
| `unlearn.CL+FT` | 方法为 `CL+FT` | `gamma` | retain FT loss 权重。 |
| `unlearn.GA+FT` | 方法为 `GA+FT` | `gamma` | retain FT loss 权重。 |
| `unlearn.KL` | 方法为 `KL` | `gamma` | forget KL 项权重。 |

##### `dataset`

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `forget_dataset_name` | `str` | `WMDPCyber` | forget 数据集名。支持 `SafePku`、`WMDPCyber`、`WMDPBio`、`WMDPALL`、`HP`、`Tofu_*`、`wikitext`、`IOI` 等。 |
| `retain_dataset_name` | `str` | `IOI,gender` | retain 数据集名。逗号分隔时会被转成列表。支持多个 retain 数据集随机采样。 |
| `dataset_seed` | `int` | `1000` | 数据采样和合成数据随机种子。 |
| `forget_ratio` | `float` | `400` | 若 >1，表示 forget 样本数量；若 0 到 1，表示 forget 数据比例。 |
| `self_retain` | bool-as-int | `False` | 是否从 forget 数据剩余部分构造 retain 数据。 |
| `batch_size` | `int` | `1` | 训练 batch size。 |

##### `logger`

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `logger.name` | `str` | 当前时间戳 | 本次运行名。 |
| `logger.json.root` | folder | `/home/chenhang/CSAT/files/logs` | JSON 日志根目录。实际输出为 `root/name/`。 |

---

## 3. 顶层 `Main` 类功能

### 3.1 `Main.__init__()`

输入：无显式输入，由命令行/config/default 收集配置。

执行顺序：

1. `make_config()`：构建并校验配置。
2. 若 `overall.use_cpu=True`，清空 `CUDA_VISIBLE_DEVICES`。
3. `setup_seed()`：固定随机性。
4. `init_model()`：收集各配置并实例化 unlearning 调度器。
5. `init_logger()`：创建日志器。
6. `run()`：启动训练或评估。

输出：无直接返回。副作用是创建 `self.config`、`self.model`、`self.logger` 并启动任务。

### 3.2 `Main.make_config(quiet=False)`

输入：

| 参数 | 作用 |
| --- | --- |
| `quiet` | False 时打印 fastargs 配置表。 |

输出：无返回。创建 `self.config`。

主要行为：

1. 调用 `get_current_config()` 获取 fastargs 全局配置对象。
2. 创建 `argparse.ArgumentParser("LLM unlearning")`。
3. `augment_argparse(parser)` 把 fastargs 参数注册到 argparse。
4. `collect_argparse_args(parser)` 合并配置来源。
5. `validate()` 做类型和必填检查。
6. `summary()` 打印参数表。

### 3.3 `Main.setup_seed(seed)`

输入：

| 参数 | 来源 | 作用 |
| --- | --- | --- |
| `seed` | `overall.seed` | Python、NumPy、Torch、CUDA 随机种子。 |

输出：无返回。副作用是设置随机种子，并关闭 cudnn benchmark，设置 deterministic。

### 3.4 `Main.init_model(model_name)`

输入：

| 参数 | 来源 | 作用 |
| --- | --- | --- |
| `model_name` | `overall.model_name` | 模型名或 checkpoint 路径。 |

内部派生变量：

| 变量 | 产生方式 | 作用 |
| --- | --- | --- |
| `kwargs` | 合并 `overall`、`unlearn`、`dataset`、条件子配置 | 传给 `model.unlearn_conflict.get(**kwargs)`。 |
| `dataset_names` | 由 `forget_dataset_name` 与 `retain_dataset_name` 构造 | 形如 `{"forget": "WMDPCyber", "retain": ["IOI", "gender"]}`。 |

输出：无返回。创建 `self.model`，实际类型是 `src/model/unlearn_conflict.py` 中的 `Unlearn` 实例。

### 3.5 `Main.init_logger(logger)`

输入：

| 参数 | 来源 | 作用 |
| --- | --- | --- |
| `logger` | `overall.logger` | 动态导入 `loggers.json_` 或 `loggers.none_`。 |

输出：无返回。创建 `self.logger`。

### 3.6 `Main.run()`

输入：无。

输出：无直接返回。调用 `self.model.run(self.logger)`。

---

## 4. 核心模型调度器

### 4.1 `src/model/unlearn_conflict.py::Unlearn`

`Unlearn` 是当前 unlearning 训练与评估的核心调度类。

#### `Unlearn.__init__(model_name, cache_dir, **kwargs)`

输入：

| 参数 | 作用 |
| --- | --- |
| `model_name` | HuggingFace 模型名或本地 checkpoint 路径。 |
| `cache_dir` | 模型/tokenizer 缓存路径。 |
| `safe_unlearn_method` | safe unlearner 方法名。 |
| `conflict_unlearn_method` | conflict unlearner 方法名，当前实际训练使用它。 |
| `safe_mask_path` | safe mask 路径。 |
| `conflict_mask_path` | conflict mask 路径。 |
| `alternate_frequency` | 交替频率。当前交替训练逻辑被注释。 |
| `batch_size` | 训练 batch size。 |
| `dataset_names` | forget/retain 数据集名结构。 |
| `dataset_seed` | 数据采样种子。 |
| `forget_ratio` | forget 抽样数量或比例。 |
| `self_retain` | 是否从 forget 剩余样本构造 retain。 |
| `num_epochs` | epoch 数。 |
| `lr` | 学习率。 |
| `gradient_accumulation_steps` | 梯度累积步数。 |
| `weight_decay` | 权重衰减。 |
| `max_grad_norm` | 梯度裁剪上限。 |
| `alpha` | L1 sparse 方法参数。 |
| `gamma` | 多种混合 loss 的权重。 |
| `task_name` | 最终评估任务类型。 |
| `sophia` | 是否用 SophiaG。 |
| `betas_low`, `betas_high`, `rho` | SophiaG 超参数。 |
| `p`, `q`, `mu` | mask 生成参数。 |
| `resume_path` | checkpoint 评估路径。 |
| `max_steps` | 最大训练步数。 |
| `use_lora` | 是否启用 LoRA。 |
| `use_cpu` 或 `CSAT_FORCE_CPU` | 是否 CPU 模式。 |

输出：无直接返回。初始化大量实例字段，但真正的模型、数据、mask、optimizer 在 `run()` 中延迟初始化。

#### `Unlearn._training_args(logger_root, output_dir, **overrides)`

输入：

| 参数 | 作用 |
| --- | --- |
| `logger_root` | 日志根目录，用于 `logging_dir`。 |
| `output_dir` | Trainer checkpoint 输出目录。 |
| `**overrides` | 覆盖默认 TrainingArguments，例如 `save_steps`。 |

输出：`transformers.TrainingArguments`。

默认参数包括：

- `per_device_train_batch_size = self.batch_size`
- `gradient_accumulation_steps = self.gradient_accumulation_steps`
- `warmup_steps = max(1, self.max_steps // 10)`
- `max_steps = self.max_steps`
- `learning_rate = self.lr`
- `bf16 = not self.use_cpu`
- `logging_steps = max(1, self.max_steps // 20)`
- `optim = "adamw_torch"`
- `weight_decay = self.weight_decay`
- `remove_unused_columns = False`
- `report_to = []`

#### `Unlearn.init_model()`

输入：使用实例字段 `model_name`、`cache_dir`、`use_cpu`、`use_lora`、`if_llama`。

输出：无返回。设置：

| 字段 | 内容 |
| --- | --- |
| `self.model` | `AutoModelForCausalLM` 或 LoRA 包装后的模型。 |
| `self.tokenizer` | `AutoTokenizer`。 |

行为：

1. CPU 模式下用 `torch.float32` 和 `device_map="cpu"`。
2. GPU 模式下用 `torch.bfloat16` 和 `device_map="auto"`。
3. 若 `use_lora=True`，构造 `LoraConfig`：`r=8`、`lora_alpha=32`、`target_modules=["q_proj", "v_proj"]`、`lora_dropout=0.05`、`task_type="CAUSAL_LM"`，并调用 `get_peft_model`。
4. 设置 `model.seqlen = model.config.max_position_embeddings`。
5. 加载 tokenizer，若无 pad token：
   - llama 系列添加 `[pad]`。
   - 其他模型将 pad token 设置为 eos token。
6. `resize_token_embeddings(len(tokenizer))` 保证 embedding 大小与 tokenizer 对齐。

#### `Unlearn.init_dataset()`

输入：使用 `dataset_names`、`tokenizer`、`dataset_seed`、`forget_ratio`、`self_retain`、`if_llama`。

调用：`dataset.get_dataset(...)`。

输出：无返回。设置：

| 字段 | 内容 |
| --- | --- |
| `self.unlearn_dataset` | 训练数据集，类型为 `UnlearnDataset`。 |
| `self.test_datasets` | retain/test 数据集字典，或兼容旧逻辑的单个数据集。 |
| `self.downstream_datasets` | 不在 retain 中的 downstream test 数据集字典。 |
| `self.unlearn_collator` | `unlearncollector`。 |
| `self.test_collator` | `default_data_collator`。 |
| `self.max_steps` | 若原值为 -1，则根据数据长度推导。 |
| `self.steps_per_epoch` | 每个 epoch 的 optimizer step 数。 |

`max_steps` 推导公式：

```text
int(num_epochs * len(unlearn_dataset)) // (batch_size * gradient_accumulation_steps * num_devices)
```

`steps_per_epoch` 推导公式：

```text
len(unlearn_dataset) // (batch_size * gradient_accumulation_steps * num_devices)
```

#### `Unlearn.init_mask(logger)`

输入：

| 参数 | 作用 |
| --- | --- |
| `logger` | 提供 `logger.get_root()`，mask 生成时需要输出目录。 |

输出：无返回。设置 `self.safe_mask`、`self.conflict_mask`、`self.mask`。

行为：

1. 如果 `safe_mask_path` 存在，`torch.load(..., map_location="cpu")` 加载。
2. 如果 `conflict_mask_path` 存在，同样加载。
3. 加载后调用 `_move_mask_to_device(mask, ..., mask_name)`，按模型线性层所在设备移动 mask。
4. 如果路径不存在，调用 `_generate_mask(mask_path, logger, mask_name)` 自动生成。
5. 默认 `self.mask = self.safe_mask`，但当前训练 unlearner 使用的是传给 conflict unlearner 的 `self.conflict_mask`。

#### `Unlearn._generate_mask(mask_path, logger, mask_name)`

输入：

| 参数 | 作用 |
| --- | --- |
| `mask_path` | mask 输出路径。路径倒数第二级被解析为 `score_type`，文件名里的 `with_*.pt` 被解析为 ratio。 |
| `logger` | 提供根目录。 |
| `mask_name` | 打印用名字，`safe` 或 `conflict`。 |

输出：无直接返回。副作用是保存 mask 到 `mask_path`。

行为：

1. `score_type = mask_path.split("/")[-2]`，例如路径包含 `/wanda/.../with_0.0.pt` 则 `score_type="wanda"`。
2. 从文件名中解析 ratio，例如 `with_0.9.pt` 得到 `ratio=0.9`。
3. 创建 `GenerateMask`。
4. 若 `score_type == "wanda"`，使用 forget_ratio=128 重新构造用于校准的 dataset。
5. 调用 `GenerateMask.get_mask()` 生成不同 ratio 的 mask。
6. 若 `score_type == "snip_forget_reinit"`，删除目标路径并返回 None。
7. 其他情况保存 mask。

#### `Unlearn.init_unlearner(logger)` 与 `_init_conflict_unlearners(logger)`

输入：`logger`。

输出：无返回。设置：

| 字段 | 内容 |
| --- | --- |
| `self.safe_unlearner` | 由 `get_unlearn_method(safe_unlearn_method, ...)` 创建。 |
| `self.conflict_unlearner` | 由 `get_unlearn_method(conflict_unlearn_method, ...)` 创建。 |

传入 unlearner 的关键变量：

- `model = self.model`
- `tokenizer = self.tokenizer`
- `train_dataset = self.unlearn_dataset`
- `data_collator = self.unlearn_collator`
- `eval_collector = self.test_collator`
- `args = TrainingArguments`
- `alpha = self.alpha`
- `gamma = self.gamma`
- `mask = self.safe_mask` 或 `self.conflict_mask`
- `if_wanda = True`
- 若已有 `self.optimizer`，传入 `optimizers=(self.optimizer, None)`。

#### `Unlearn.init_optimizer()`

输入：使用实例字段 `sophia`、`lr`、`betas`、`rho`、`weight_decay`。

输出：无返回。设置 `self.optimizer`。

行为：

- 若 `sophia=True`，调用 `create_sophia_optimizer(...)`。
- 否则创建 `torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)`。

#### `Unlearn.run(logger)`

输入：`logger`。

输出：无直接返回。完整执行训练或评估。

流程：

1. 若 `resume_path is None`：
   - `init_model()`
   - `init_optimizer()`
   - `init_dataset()`
   - `init_mask(logger)`
   - `init_unlearner(logger)`
   - 保存初始 checkpoint，tag 为 `init`
   - 对 retain/test 数据做一次 in-memory 准确率评估
   - `_run_conflict_training(logger)`
   - 保存最终 checkpoint，tag 为 `final`
   - 删除 `logger_root/unlearn_checkpoint`
   - `eval(logger)` 做最终评估
2. 若 `resume_path` 不为空：
   - `init_model()`
   - `init_dataset()`
   - `eval(logger)`，不训练。

#### `Unlearn._run_conflict_training(logger)`

输入：`logger`。

输出：无返回。调用 `_custom_conflict_training_loop(...)`。

行为：打印 unlearn 方法和交替频率，计算 `steps_per_epoch`。

#### `Unlearn._custom_conflict_training_loop(steps_per_epoch, logger)`

输入：

| 参数 | 作用 |
| --- | --- |
| `steps_per_epoch` | 每 epoch optimizer step 数。 |
| `logger` | 保存 checkpoint 与 retain eval 输出目录。 |

输出：无返回。副作用是更新模型参数、保存 checkpoint、打印 retain 评估。

当前训练行为：

1. `self.model.train()`。
2. 若 optimizer 为空，初始化 optimizer。
3. 当前实际训练 unlearner 固定为 `self.conflict_unlearner`。safe/conflict 交替代码被注释。
4. 打印 mask 冻结比例报告。
5. 创建 `DataLoader(self.unlearn_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.unlearn_collator)`。
6. 每个 batch：
   - 将 batch tensor 移到 `self.model.device`。
   - GPU 模式用 bf16 autocast，CPU 模式不用 autocast。
   - 调用 `current_unlearner.compute_loss(current_unlearner.model, batch)`。
   - 若 loss 是 NaN/Inf，跳过 batch。
   - `loss.backward()`。
   - `clip_grad_norm_` 做梯度裁剪。
   - 达到梯度累积步时：
     - 检查梯度是否 NaN/Inf。
     - 若有 mask，调用 `current_unlearner.mask_gradient(...)`。
     - `optimizer.step()`。
     - `optimizer.zero_grad()`。
     - `current_step += 1`。
7. 每个 epoch 结束：
   - 打印平均 loss。
   - 保存 checkpoint，tag 为 `epoch-{n}`。
   - 调用 `_print_retain_accuracy(...)` 做 in-memory retain 评估。

#### `Unlearn.save(logger, tag=None)`

输入：

| 参数 | 作用 |
| --- | --- |
| `logger` | 提供根目录。 |
| `tag` | checkpoint 名后缀，例如 `init`、`epoch-1`、`final`。 |

输出：无返回。保存模型和 tokenizer 到：

```text
{logger_root}/checkpoints/checkpoint-{timestamp}-{tag}
```

同时更新 `self.latest_checkpoint_path`。

#### `Unlearn.eval(logger)`

输入：`logger`。

输出：无返回。副作用是生成评估 JSON 文件。

行为：

1. 释放内存中的 `self.model` 并清 CUDA cache。
2. 若 `resume_path` 有值，评估该路径；否则评估 `latest_checkpoint_path` 或日志目录下最新 checkpoint。
3. 当前代码主要处理 `task_name == "downstream"`：
   - 若 forget 数据集名包含 `WMDP`：用 `eval_few_shots` 分别评估 `wmdp` 与 `mmlu`。
   - 若 forget 数据集为 `SafePku`：调用 `eval_toxic`。
   - 对每个 retain test 数据集调用 `eval_acc`，输出到各自目录。
   - 对不在 retain 中的 downstream 数据集也调用 `eval_acc`。
   - 最后调用默认 `eval_few_shots`，输出到 `few_shots.json`。
4. `eval_ppl` 当前被注释。

#### `Unlearn.eval_accuracy(...)`、`eval_accuracy_in_memory(...)`

功能：评估 retain/test 数据集准确率。

区别：

- `eval_accuracy` 调用 `metrics.simple_accuracy.eval_acc`，从磁盘重新加载模型。
- `eval_accuracy_in_memory` 直接用当前内存中的 `self.model`，用于每个 epoch 后快速打印 retain accuracy。

准确率定义：对每条样本取 `labels != -100` 且 `attention_mask == 1` 的最后一个有效 token，比较模型预测 token 是否等于 label。

#### `_print_parameter_freeze_report(unlearner)`

输入：`unlearner`，需要有 `mask` 与 `if_wanda` 字段。

输出：无返回。打印：

1. `requires_grad=False` 参数标量比例。
2. 若有 mask：
   - Wanda mask：统计每个线性层 weight 中 mask==0 的比例。
   - 参数名 mask：统计 named_parameters 中 mask==0 的比例。

---

## 5. 旧版或辅助模型封装

### 5.1 `src/model/base.py::BaseModel`

该类不像当前主入口那样处理 conflict unlearning，更像旧版稀疏训练/恢复/评估封装。

输入：

| 参数 | 作用 |
| --- | --- |
| `model_name`, `cache_dir` | 模型加载。 |
| `sparse_training` | 是否用 sparse trainer。 |
| `recovery` | 是否执行恢复训练。 |
| `lr`, `num_warmup_steps`, `epochs` | 训练参数。 |
| `dataset_name`, `batch_size` | 数据与 batch 参数。 |

主要方法：

| 方法 | 输入 | 输出/副作用 |
| --- | --- | --- |
| `init_model()` | 无 | 加载 causal LM 和 tokenizer，设置 device。 |
| `init_trainer()` | 无 | 根据 `sparse_training` 选择 `sparsetrainer` 或 `trainer`。 |
| `init_loaders()` | 无 | 用 BookCorpus 构造 dataloader。 |
| `prune(pruner, logger)` | pruner、logger | 调用 pruner 并记录 sparsity。 |
| `eval(logger)` | logger | 保存后调用 few-shot、ppl、toxic、MIA 评估。 |
| `save(logger)` | logger | 保存 model/tokenizer。 |
| `recover(logger)` | logger | 若 `recovery=True` 则训练恢复并保存。 |

---

## 6. 数据结构与数据管线

### 6.1 训练样本标准字段

#### Forget 样本

forget 数据集样本通常包含：

| 字段 | 类型 | 作用 | 如何得到 |
| --- | --- | --- | --- |
| `input_ids` | `torch.Tensor[seq_len]` | 模型输入 token。 | 由 tokenizer 对 prompt+answer 或纯文本编码得到。 |
| `attention_mask` | `torch.Tensor[seq_len]` | 非 padding token 标记。 | tokenizer 产生或手工 padding 产生。 |
| `label` | `torch.Tensor[seq_len]` | 正常目标标签，通常问题部分为 -100，只在答案或文本位置计算 loss。 | 预处理函数构造。 |
| `refused_label` | `torch.Tensor[seq_len]` | 拒绝回答标签，用于 CL/RL 等遗忘方法。 | 从 polite refusal CSV 中随机选拒绝回答后拼接。 |
| `question_length` | `torch.Tensor` 或 int | 问题 token 长度，用于定位 answer 起点。 | tokenizer 统计 prompt 部分长度，WMDP 中有时是随机替换位置。 |

#### Retain 样本

retain/test/downstream 样本通常包含：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `input_ids` | `torch.Tensor[seq_len]` | 语言模型输入。 |
| `attention_mask` | `torch.Tensor[seq_len]` | attention mask。 |
| `label` | `torch.Tensor[seq_len]` | 训练或评估标签。 |

### 6.2 `src/dataset/Base.py::BaseDataset`

基础类，定义通用 prompt 模板：

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `question_start_token` | `### Question: ` | 问题前缀。 |
| `question_end_token` | `\n` | 问题结束符。 |
| `answer_start_token` | `### Answer: ` | 答案前缀。 |

抽象方法：

| 方法 | 输入 | 输出 |
| --- | --- | --- |
| `get_dataset()` | 无 | 原始 train/test 数据。 |
| `__preprocess__(tokenizer, ...)` | tokenizer 等 | 预处理并设置 tensor 格式。 |
| `build_dataset(tokenizer, ...)` | tokenizer 等 | 返回 `{"train": train_dataset, "test": test_dataset}`。 |

### 6.3 `src/dataset/Base.py::UnlearnDataset`

功能：把 forget 数据和一个或多个 retain 数据组合为训练样本。

#### `__init__(datasets, forget_ratio, dataset_seed, self_retain=False)`

输入：

| 参数 | 作用 |
| --- | --- |
| `datasets` | 字典，至少可能包含 `forget`，以及 `retain`、`retain1`、`retain2` 等键。 |
| `forget_ratio` | 若 >1 表示抽样数量，若 0 到 1 表示比例。 |
| `dataset_seed` | 随机采样种子。 |
| `self_retain` | 是否把未被抽中的 forget 样本作为 retain。 |

输出：无返回。设置内部 `forget_dataset` 和 `retain_datasets`。

#### `build_unlearn_dataset()`

输入：使用实例字段。

输出：无返回。副作用：

1. 从 forget 数据集随机抽取 `forget_ratio` 对应样本。
2. 若 `self_retain=True`，将剩余 forget 样本放入第一个 retain 数据集。
3. 调用底层数据集 `.select(indices)`。

#### `__len__()`

输出：

- 若有 forget 数据集，返回 forget 长度。
- 否则返回第一个非空 retain 数据集长度。

#### `__getitem__(idx)`

输出：一个组合样本字典。

- 若有 forget：返回 `{"forget": forget_data, "retain": retain_data}` 或 `{"forget": forget_data, "retain1": retain_data}`。
- 多 retain 时，随机选择一个可用 retain 数据集并随机抽一个样本。
- 若没有 forget，则只返回 retain 样本。

### 6.4 `src/dataset/Base.py::unlearncollector(samples)`

输入：`samples`，即 `UnlearnDataset.__getitem__` 产生的样本列表。

输出：batch 字典。

| 键 | 值 | 说明 |
| --- | --- | --- |
| `forget` | 五元组 `(input_ids, attention_mask, label, refused_label, question_length)` 或 None | 用于 forget loss。 |
| `retain` 或 `retain{i}` | 三元组 `(input_ids, attention_mask, label)` 或 None | 用于 retain loss。 |

---

## 7. 数据集功能清单

### 7.1 `src/dataset/__init__.py::get_dataset(...)`

这是主流程统一数据入口。

输入：

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `dataset_names` | dict | `{"forget": name, "retain": name_or_list}`。 |
| `tokenizer` | HF tokenizer | 所有数据集的 tokenizer。 |
| `dataset_seed` | int | 抽样、合成数据随机种子。 |
| `forget_ratio` | float | forget 抽样数量或比例。 |
| `self_retain` | bool | 是否从 forget 剩余样本生成 retain。 |
| `if_llama` | bool | 控制 ToFU 等数据的 Llama 风格模板。 |

输出：

```python
unlearn_dataset, test_datasets, unlearn_collator, test_collator, downstream_datasets
```

| 输出 | 类型 | 作用 |
| --- | --- | --- |
| `unlearn_dataset` | `UnlearnDataset` | 训练时每步产出 forget+retain 样本。 |
| `test_datasets` | dict | retain 数据对应 test/validation 数据集。单 retain 时键为 `test`，多 retain 时为 `test1`、`test2`。 |
| `unlearn_collator` | function | `unlearncollector`。 |
| `test_collator` | function | `default_data_collator`。 |
| `downstream_datasets` | dict | 不在 retain 中的机制/下游数据集 test 集，键为 `downstream_{name}`。 |

### 7.2 Forget 数据集

#### `SafePkuDataset`

文件：`src/dataset/SafePku.py`

功能：加载 PKU-SafeRLHF 中 unsafe response，作为安全对齐遗忘数据。

输入：

| 参数 | 作用 |
| --- | --- |
| `dataset_name` | 数据集标识。 |
| `with_retain` | 继承字段，当前未显著使用。 |
| `if_llama` | 是否 Llama 模板。 |
| `tokenizer` | `build_dataset(tokenizer)` 中传入。 |

数据来源：

- HuggingFace `PKU-Alignment/PKU-SafeRLHF`，固定 revision。
- 拒绝回答 CSV：`files/data/polite_refusal_responses/polite_refusal_responses.csv`。

输出字段：

- `input_ids`
- `attention_mask`
- `label`
- `refused_label`
- `question_length`

关键行为：

1. 对每个 prompt 取不安全的 `response_0` 和/或 `response_1`。
2. 构造 `### Question: {prompt}\n### Answer: {unsafe_response}`。
3. label 中问题部分置为 -100。
4. 随机选 polite refusal response，构造 `refused_label`。

#### `ToFU`

文件：`src/dataset/Tofu.py`

功能：加载 ToFU 虚构作者/事实数据，用于目标知识遗忘与 retain 测试。

输入：

| 参数 | 作用 |
| --- | --- |
| `dataset_name` | 通常为 `TOFU`。 |
| `subset` | 例如 `forget01`、`forget10`、`retain99`、`real_authors`、`world_facts`、`full`。 |
| `if_llama` | 是否使用 `[INST] ... [\INST]` 风格模板。 |
| `tokenizer` | 构造 token 字段。 |

数据来源：HuggingFace `locuslab/TOFU`。

输出：

- `build_dataset(tokenizer)` 返回 `train` 与原始 `test`。
- 训练集字段为 `input_ids`、`attention_mask`、`label`、`refused_label`、`question_length`。

关键行为：

1. 根据 `subset` 加载 train。
2. 根据 `subset` 选择对应 perturbed test，例如 `forget01_perturbed`。
3. `label` 只对答案部分计算 loss。
4. `refused_label` 从 `polite_refusal_responses_tofu.csv` 随机抽拒绝回答。
5. `build_pretrain_dataset(tokenizer, subset="full")` 可生成预训练式 train dataset 与 collector。

#### `WMDPCyber`、`WMDPBio`、`WMDPALL`

文件：`src/dataset/wmdp.py`

功能：加载 WMDP 危险知识语料与选择题评测集。

数据来源：

| 类 | 训练数据 | 测试数据 |
| --- | --- | --- |
| `WMDPCyber` | `cais/wmdp-corpora` 的 cyber forget/retain corpus | `cais/wmdp`, `wmdp-cyber` |
| `WMDPBio` | `cais/wmdp-bio-forget-corpus` 或 bio retain corpus | `cais/wmdp`, `wmdp-bio` |
| `WMDPALL` | cyber + bio 拼接 | bio + cyber test 拼接 |

输入：

| 参数 | 作用 |
| --- | --- |
| `dataset_name` | 数据集标识。 |
| `subset` | `forget` 或 `retain`，决定语料来源。 |
| `tokenizer` | 编码文本和测试 prompt。 |

训练输出字段：

- `input_ids`
- `attention_mask`
- `label`
- `refused_label`
- `question_length`

测试输出字段：

- `input_ids`
- `attention_mask`
- `answer`

关键行为：

1. forget/retain train 语料为纯文本，`label` 通常等于 `input_ids`。
2. `refused_label` 用随机拒绝回答拼接，WMDP Cyber/Bio 从第 10 到 20 个 token 附近开始替换。
3. test 集构造多选题 prompt，保留正确答案编号 `answer`。

#### `HP`

文件：`src/dataset/HorryPotter.py`

功能：加载 Harry Potter QA/原文数据，作为版权或特定文本知识遗忘数据。

数据来源：

- `files/data/hp/hp_qa.jsonl`
- `files/data/hp/hp.jsonl`
- `files/data/polite_refusal_responses/polite_refusal_responses_copyright.csv`

主要方法：

| 方法 | 输入 | 输出/作用 |
| --- | --- | --- |
| `build_dataset(tokenizer)` | tokenizer | 返回 train/test，其中 test 当前为 None。训练字段含 refused_label。 |
| `build_pretrain_dataset(tokenizer)` | tokenizer | 拼接 QA 与原文，返回 `DatasetDict(train, test=None)`。 |
| `build_test_dataset(tokenizer, path)` | tokenizer、jsonl 路径 | 返回带 prompt/response/text/tokenized prompt 的测试集。 |
| `build_test_dataset_without_tokenized(path)` | jsonl 路径 | 返回未 tokenizer 的测试集。 |

### 7.3 通用语言建模 retain 数据集

#### `C4`

文件：`src/dataset/C4.py`

数据来源：`allenai/c4`。

输入：`dataset_name`、`tokenizer`。

输出字段：`input_ids`、`attention_mask`、`label`。

行为：对 C4 train/validation 文本进行 `max_length=512` 编码，`label=input_ids`。

#### `wikitext`

文件：`src/dataset/wikitext2.py`

数据来源：`wikitext-2-raw-v1`。

输入：`dataset_name`、`tokenizer`。

输出字段：`input_ids`、`attention_mask`、`label`。

行为：对 train/test 文本进行 `max_length=512` 编码，`label=input_ids`。

### 7.4 标准 NLP retain/downstream 数据集

#### `SST2`

文件：`src/dataset/sst2.py`

数据来源：`stanfordnlp/sst2`。

输入：`dataset_name`、`tokenizer`。

输出字段：`input_ids`、`attention_mask`、`label`。

行为：

1. 把情感分类样本转成生成式 QA：
   ```text
   Is the sentiment of following sentence positive or negative?{sentence}
   Answer: It is {positive|negative}
   ```
2. 手工构造 causal LM 的 input/label 对齐格式。
3. padding 到 `max_len=200`。

#### `Winogrande`

文件：`src/dataset/winogrande.py`

数据来源：`allenai/winogrande`, `winogrande_debiased`。

输入：`dataset_name`、`tokenizer`。

输出字段：`input_ids`、`attention_mask`、`label`。

行为：

1. 将句子和候选项转成生成式判断：
   ```text
   {sentence}
   Should the '_' be {answer}?
   Answer: Yes
   ```
2. padding 到 `max_len=200`。

### 7.5 机制解释/合成 retain 与 downstream 数据集

#### `IOIDataset`

文件：`src/dataset/ioi_dataset.py`

功能：生成 Indirect Object Identification 合成句子。

主要输入：

| 参数 | 作用 |
| --- | --- |
| `prompt_type` | `ABBA`、`BABA`、`mixed`、`ABC` 等模板类型。 |
| `N` | 样本数量。 |
| `tokenizer` | 默认 GPT-2 tokenizer。 |
| `nb_templates` | 使用模板数量。 |
| `seed` | 随机种子，必填。 |
| `prepend_bos` | 是否加 BOS。 |

主要输出字段：

| 字段 | 作用 |
| --- | --- |
| `ioi_prompts` | prompt 元数据列表，包含文本、IO、S 等角色。 |
| `sentences` | 生成的文本句子。 |
| `toks` | GPT-2 tokenizer 后的 token tensor。 |
| `word_idx` | 语义位置索引字典。 |
| `io_tokenIDs`, `s_tokenIDs` | IO/S 名字对应 token id。 |

相关 helper 函数：

- `gen_prompt_uniform(...)`：按模板生成 prompt。
- `gen_flipped_prompts(...)`：生成角色翻转的 prompt。
- `get_name_idxs(...)`、`get_word_idxs(...)`、`get_end_idxs(...)`、`get_idx_dict(...)`：计算关键 token 位置。
- `flip_prefixes(...)`、`flip_names(...)`：构造扰动 prompt。

#### `IOIDatasetWrapper`

文件：`src/dataset/__init__.py`

功能：把 `IOIDataset` 的 GPT-2 token/文本转成当前目标模型 tokenizer 可训练格式。

输入：

| 参数 | 作用 |
| --- | --- |
| `ioi_dataset` | 原始 IOI 合成数据。 |
| `target_tokenizer` | 当前 LLM tokenizer。 |

输出：实现 `__len__`、`__getitem__`、`select`，样本字段为 `input_ids`、`attention_mask`、`label`。

处理方式：对原始文本重新 tokenize，固定 padding 到 `max_len=50`。

#### `InductionDatasetWrapper`

输入：

| 参数 | 作用 |
| --- | --- |
| `induction_dataset` | `get_validation_data()` 下载/加载的 token tensor。 |
| `target_tokenizer` | 当前模型 tokenizer。 |

数据来源：HuggingFace Hub `ArthurConmy/redwood_attn_2l` 的 `validation_data.pt`。

行为：

1. 用 `ArthurConmy/redwood_tokenizer` 解码原 token。
2. 用目标 tokenizer 重新编码。
3. padding 到 `max_len=600`。

#### `DocstingDatasetWrapper` 与 `Prompt`

文件：`src/dataset/docstring.py` 与 `src/dataset/__init__.py`

功能：构造 docstring induction 合成任务。

关键类：`Prompt`

| 字段 | 作用 |
| --- | --- |
| `clean_prompt` | 正常 prompt。 |
| `corrupt_prompt` | 扰动 prompt，可为字符串或字典。 |
| `correct_answers` | 正确答案列表。 |
| `wrong_answers` | 错误答案列表。 |

关键函数：

| 函数 | 输入 | 输出 |
| --- | --- | --- |
| `docstring_prompt_templ(style, ...)` | docstring 风格、函数名、参数名、描述词等 | Python 函数 docstring prompt 字符串。 |
| `docstring_induction_prompt_generator(style, ...)` | 参数数量、描述长度、seed 等 | `Prompt` 对象。 |

Wrapper 行为：把 `clean_prompt + correct_answers[0]` 转成 `input_ids`、`attention_mask`、`label`，padding 到 `max_len=50`。

#### `GenderDatasetWrapper`

数据来源：本地 `/ssd_users/chenhang/CSAT/files/data/gp`。

输入：`gender_data`、`target_tokenizer`。

行为：读取样本里的 `prefix` 和 `pronoun`，组成文本并编码为 causal LM 样本，padding 到 `max_len=50`。

#### `BoolDatasetWrapper`

数据来源：本地 `/ssd_users/chenhang/CSAT/files/data/boolean_expressions`。

输入：`bool_data`、`target_tokenizer`。

行为：构造 instruction prompt：

```text
[INST] <<SYS>>
Evaluate the following boolean expression as either 'True' or 'False'.
<</SYS>>

{expression} [/INST] {target}
```

输出字段：`input_ids`、`attention_mask`、`label`，padding 到 `max_len=100`。

#### `gender.py` 与 `bool.py::load_datasets(...)`

输入：

| 参数 | 作用 |
| --- | --- |
| `dataset_path` | 本地 dataset 路径或 HuggingFace dataset 名。 |
| `max_train_samples` | train 最大样本数。 |
| `max_eval_samples` | validation/test 最大样本数。 |
| `train_split` | train split 名，默认 `train`。 |

输出：`DatasetDict`。若原数据无 validation，则从 train 中切出 validation。

### 7.6 数据加载辅助函数

文件：`src/dataset/dataset.py`

| 函数/类 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| `set_seed(seed)` | seed | 无 | 设置 NumPy/Torch 随机种子。 |
| `TokenizerWrapper(input_ids)` | input_ids | wrapper 对象 | 给某些评估包装 tokenized tensor。 |
| `get_wikitext2(nsamples, seed, seqlen, tokenizer)` | 样本数、seed、序列长度、tokenizer | `(trainloader, testenc)` | SparseGPT 风格 wikitext2 采样。 |
| `get_c4(nsamples, seed, seqlen, tokenizer)` | 同上 | `(trainloader, valenc)` | C4 片段采样。 |
| `get_loaders(name, ...)` | 数据集名等 | loader | 根据 name 路由到 wikitext2 或 C4。 |
| `create_pku_dataloader_from_dataset(tokenizer, dataset, fraction, batch_size, dataset_seed)` | tokenizer、PKU dataset 等 | DataLoader | 取 unsafe response 构造 harmful QA dataloader。 |
| `get_pku_test_dataset(dataset_seed, fraction)` | seed、比例 | dataset | 加载 PKU-SafeRLHF test。 |
| `get_real_toxic_dataset(dataset_seed, fraction)` | seed、比例 | dataset | 加载 RealToxicityPrompts。 |
| `bookcorpus_loaders(tokenizer, batch_size)` | tokenizer、batch | DataLoader | BookCorpus 语言建模 loader。 |
| `get_WikiMIA_dataset(LENGTH)` | 长度档 | dataset | 加载 WikiMIA 数据。 |
| `build_unlearn_dataset(dataset, dataset_seed, forget_ratio)` | WikiMIA 等带 label 数据 | `(forget_dataset, remain_dataset, test_dataset)` | 按 label 和比例拆分 MIA 数据。 |

---

## 8. Unlearning 方法与 Loss 功能

### 8.1 `src/unlearn/__init__.py::get_unlearn_method(name, *args, **kwargs)`

功能：根据字符串返回具体 Trainer 子类。

输入：

| 参数 | 作用 |
| --- | --- |
| `name` | 方法名。 |
| `*args`, `**kwargs` | 传给具体类，包含 model、tokenizer、dataset、TrainingArguments、mask、alpha、gamma 等。 |

输出：具体 unlearner 实例。

当前支持：

| name | 类 |
| --- | --- |
| `FT` | `FT` |
| `l1sparse` | `FT_l1` |
| `GA` | `GA` |
| `GA+FT` | `GA_FT` |
| `GA+KL` | `GA_KL` |
| `RL` | `RL` |
| `KL` | `KL(if_kl=True)` |
| `CL` | `CL` |
| `CL+FT` | `CL_FT(if_kl=True)` |
| `CL+KL` | `CL_KL(if_kl=True)` |
| `NPO` | `NPO(if_kl=True)` |
| `NPO+FT` | `NPO_FT(if_kl=True)` |

### 8.2 `src/unlearn/base.py::BaseTrainer`

继承自 `transformers.Trainer`，加入 eval collator、mask、KL 参考模型以及梯度 mask 应用。

#### `__init__(eval_collector, alpha=None, gamma=None, if_kl=False, mask=None, if_wanda=False, *args, **kwargs)`

输入：

| 参数 | 作用 |
| --- | --- |
| `eval_collector` | eval dataloader 的 collator。 |
| `alpha` | L1 loss 权重。 |
| `gamma` | 混合 loss 权重。 |
| `if_kl` | True 时 deepcopy 当前模型为 `infer_model`，作为参考模型。 |
| `mask` | 参数名 mask 或 Wanda int-key mask。 |
| `if_wanda` | True 时按线性层编号套 mask，否则按 named_parameters。 |
| `*args`, `**kwargs` | HuggingFace Trainer 标准参数。 |

输出：无返回。设置实例字段。

#### `mask_gradient(model, if_wanda=False)`

输入：

| 参数 | 作用 |
| --- | --- |
| `model` | 当前训练模型。 |
| `if_wanda` | 是否使用 Wanda mask 格式。 |

输出：无返回。副作用是原地修改梯度。

行为：

- 非 Wanda：遍历 `model.named_parameters()`，执行 `tensor.grad *= self.mask[key]`。
- Wanda：遍历 `model.model.layers` 或 `model.model.decoder.layers` 的所有线性层，按计数器 `cnt` 取 `self.mask[cnt]`，执行 `subset[name].weight.grad *= self.mask[cnt]`。

#### `get_loss(output, labels)` 与 `compute_metrics(pred)`

- `get_loss`：将 logits 和 labels 右移，对 causal LM 计算 cross entropy，忽略 -100。
- `compute_metrics`：计算 token 级 accuracy 与 loss。

### 8.3 具体 loss 类

所有 `compute_loss(model, inputs, return_outputs=False)` 的 `inputs` 都来自 `unlearncollector`。

#### `FT`

文件：`src/unlearn/FT.py`

输入：retain 数据三元组。

输出：retain 平均 CE loss。

行为：遍历所有 `retain*` 键，对每个 retain 数据调用模型，平均 `outputs.loss`。没有 retain 时返回 0 loss。

#### `FT_l1`

输入：单个 `inputs["retain"]`、`alpha`。

输出：`retain CE loss + alpha * sum(L1(trainable parameters))`。

#### `GA`

文件：`src/unlearn/GA.py`

输入：`inputs["forget"]` 中的正常 `label`。

输出：`-forget CE loss`。

作用：通过梯度上升增加 forget 数据 loss，使模型遗忘。

#### `GA_FT`

输入：forget 数据、一个或多个 retain 数据、`gamma`。

输出：`-forget_loss + gamma * mean(retain_loss)`。

作用：同时遗忘 forget、保持 retain。

#### `GA_KL`

输入：forget 数据、retain 数据、`gamma`、`infer_model`。

输出：`forget_loss + (1 - gamma) * retain_KL`，其中 `forget_loss=-outputs.loss`。

retain KL 由当前模型和参考模型在 retain logits 上计算。

#### `NPO`

输入：forget 数据、`infer_model`。

输出：负偏好优化 loss：

```text
- log_sigmoid(0.1 * (current_forget_loss - ref_forget_loss)).mean() * 2 / 0.1
```

作用：让当前模型相对参考模型更不偏好 forget 答案。

#### `NPO_FT`

输入：forget 数据、retain 数据、`infer_model`、`gamma`。

输出：`NPO forget loss + gamma * mean(retain_loss)`。

#### `KL`

文件：`src/unlearn/KL.py`

输入：forget 数据、retain 数据、`infer_model`、`gamma`。

输出：

- 有 retain：`-gamma * forget_KL + retain_KL`
- 无 retain：`-gamma * forget_KL`

`kl_loss(prob_p, prob_q)` 实际实现为：

```python
-(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()
```

#### `KL_GA`

输入：forget 数据、retain 数据、`infer_model`、`gamma`。

输出：`gamma * (-forget CE loss) + retain_KL`。

当前 `get_unlearn_method` 没有直接暴露 `KL_GA` 名称。

#### `KL_CL`

输入：forget 数据的 `refused_label`、retain 数据、`infer_model`、`gamma`。

输出：`gamma * CL-style forget loss + retain_KL`。

当前 `get_unlearn_method` 没有直接暴露 `KL_CL` 名称，`CL+KL` 使用的是 `CL_KL`。

#### `CL`

文件：`src/unlearn/CL.py`

输入：forget 五元组，尤其是：

- `forget_data[0]`: input_ids
- `forget_data[1]`: attention_mask
- `forget_data[3]`: refused_label
- `forget_data[4]`: question_length

输出：对拒绝回答目标计算 CE loss。

行为：从 `question_length` 后开始，把 input 中答案部分替换成 `refused_label` 对应 token，并对拒绝回答标签训练。

#### `CL_FT`

输入：forget 数据、retain 数据、`gamma`。

输出：`CL forget loss + gamma * mean(retain_loss)`。

#### `CL_KL`

输入：forget 数据、retain 数据、`infer_model`、`gamma`。

输出：`CL forget loss + gamma * retain_KL`。

#### `RL`

文件：`src/unlearn/RL.py`

输入：forget 数据与第一个可用 retain 数据。

输出：用 retain labels 替换 forget answer 部分后的 CE loss。

行为：

1. 取第一个 retain 数据集的 labels。
2. 对齐到 forget labels 长度。
3. 从 `question_length` 后替换 forget input 的答案部分。
4. 训练模型在 forget prompt 后生成 retain label 风格内容。

#### `CUT`

文件：`src/unlearn/CUT.py`

功能：基于 steering vector 的层级干预雏形。

输入：`layer_id`、keyword list、model/tokenizer。

当前行为：

- 从 `files/data/key_word_list/keywords.json` 读取关键词。
- 对 “novice” 与 “expert” prompt 做 forward hook，提取某层 activation 差值作为 steering direction。

当前未在 `get_unlearn_method` 注册，主流程默认不会使用。

---

## 9. Mask 生成与应用功能

### 9.1 Mask 格式

当前有两种 mask 格式。

#### 参数名 mask

格式：

```python
{
  parameter_name: torch.BoolTensor(shape == parameter.shape),
  ...
}
```

应用：`BaseTrainer.mask_gradient(if_wanda=False)` 中按 `model.named_parameters()` 直接 `grad *= mask[name]`。

#### Wanda/线性层编号 mask

格式：

```python
{
  0: torch.BoolTensor(shape == first_linear.weight.shape),
  1: torch.BoolTensor(shape == second_linear.weight.shape),
  ...
}
```

应用：按 transformer 层顺序遍历所有 `nn.Linear`，用计数器 `cnt` 匹配 mask，并只作用在 `linear.weight.grad`。

Edge-Pruning 目录可以非端到端生成这类 mask；主训练只消费 `.pt` mask 文件。

### 9.2 `src/unlearn/generate_mask.py::GenerateMask`

继承：`transformers.Trainer`。

#### `__init__(score_type, ratios, mask_dir, p, q, mu, *args, **kwargs)`

输入：

| 参数 | 作用 |
| --- | --- |
| `score_type` | mask 打分方法，例如 `gradient`、`weight`、`random`、`wanda`、`snip_advanced`。 |
| `ratios` | 外部传入 ratio。当前代码内部固定覆盖为 `[0.0, 0.1, 0.2, 0.5, 0.8, 0.9]`。 |
| `mask_dir` | mask 输出目录。 |
| `p`, `q`, `mu` | SNIP/高阶近似相关参数。 |
| `*args`, `**kwargs` | Trainer 标准参数，包括 model、dataset、collator、TrainingArguments。 |

输出：无返回，设置实例字段。

#### `get_mask()`

输入：无显式输入，使用实例字段。

输出：通常无直接返回，而是保存多个 `with_{ratio}.pt` 文件；`wanda` 分支也直接保存文件。

支持的 `score_type`：

| score_type | 行为 |
| --- | --- |
| `gradient` | 计算 forget 梯度绝对值，取负作为 score。 |
| `gradient_vis` | 同 gradient，并保存 `scores.pt` 后退出。 |
| `weight` | 用参数绝对值取负作为 score。 |
| `weight_vis` | 同 weight，并保存 `scores.pt` 后退出。 |
| `random` | 随机 score。 |
| `snip_advanced` | 结合 forget gradient、retain gradient 和 `mu` 估计重要性。 |
| `snip_advanced_CL` | 使用 CL 风格 forget loss 的 advanced SNIP。 |
| `snip_advanced_gn` | 结合 retain gradient 近似 Hessian。 |
| `snip_advanced_visualization` | 生成并保存 score 用于可视化。 |
| `snip_advanced_new` | 另一个 SNIP 高阶近似版本。 |
| `FFN` | 只保留名称含 `fc` 或 `final_layer_norm` 的参数。 |
| `wanda` | 计算 Wanda 权重激活指标并按线性层编号保存 mask。 |

#### `score2mask(scores, ratio, return_rank=False)`

输入：

| 参数 | 作用 |
| --- | --- |
| `scores` | 展平后的全模型 score。 |
| `ratio` | 阈值比例。 |
| `return_rank` | True 时返回每个元素排名。 |

输出：

- `return_rank=True`：返回 `ranks`。
- 否则返回 `hard_dict`，即参数名 mask。

#### `gradient(dataset="forget")`

输入：`dataset` 指示用 `forget` 或 `retain` loss。

输出：设置 `self.scores`。

行为：遍历 train dataloader，调用 `compute_loss_adapted`，累积梯度，最后把 `-abs(gradient)` 拼成全局 score。

#### `weight()`

输出：设置 `self.scores = concat(-abs(parameter.data))`。

#### `random()`

输出：设置随机 score。

#### `snip_advanced(...)`、`snip_advanced_gn(...)`、`snip_advanced_new(...)`

功能：分别计算 forget/retain 梯度，组合参数值、retain 梯度、近似 Hessian 或 `mu`，得到更复杂的重要性 score。

输出：设置 `self.scores`。

#### `wanda()`

输入：使用 train dataloader 的 forget batch 做校准。

输出：保存 `{cnt: bool_mask}` 到 `with_{ratio}.pt`。

行为：

1. 关闭 `model.config.use_cache`。
2. `prepare_calibration_input` 抓取第一层输入 activation。
3. 遍历 transformer 层和线性层。
4. 用 forward hook 统计输入 L2 norm：`WrappedGPT.scaler_row`。
5. 对每个线性层计算：
   ```python
   W_metric = abs(weight) * sqrt(scaler_row)
   ```
6. 每个 ratio 下按列排序生成 bool mask。

#### `compute_loss_adapted(model, inputs, key, CL=False, FT=False, return_outputs=False)`

输入：

| 参数 | 作用 |
| --- | --- |
| `model` | 当前模型。 |
| `inputs` | `unlearncollector` 产出的 batch。 |
| `key` | `forget` 或 `retain`。 |
| `CL` | True 且 key=forget 时使用 refused_label 构造 loss。 |
| `FT` | True 时额外加 retain CE loss。 |
| `return_outputs` | 是否返回 `(loss, outputs)`。 |

输出：loss 或 `(loss, outputs)`。

---

## 10. Pruner 工具函数

文件：`src/pruner/utils.py`

| 函数/类 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| `find_layers(module, layers=[nn.Linear], name="")` | PyTorch module、目标层类型 | `{name: layer}` | 递归查找线性层，mask 与 Wanda 都依赖它。 |
| `check_sparsity(model)` | 模型 | sparsity float | 统计每层线性层 weight 中 0 的比例。 |
| `prepare_calibration_input(model, dataloader, device)` | 模型、校准 dataloader、device | `(inps, outs, attention_mask, position_ids)` | 捕获第一层输入，用于稀疏化校准。 |
| `WrappedGPT(layer, layer_id=0, layer_name="none")` | 线性层 | wrapper | 在 `add_batch(inp, out)` 中统计输入 token 的行尺度，用于 Wanda。 |

---

## 11. 优化器功能

### 11.1 `src/optim/__init__.py`

#### `get_decay_parameter_names(model)`

输入：模型。

输出：需要 weight decay 的参数名列表。

行为：使用 Transformers 的 `get_parameter_names`，排除 LayerNorm 类参数和名称含 `bias` 的参数。

#### `create_sophia_optimizer(model, weight_decay, lr, betas, rho)`

输入：

| 参数 | 作用 |
| --- | --- |
| `model` | 需要优化的模型。 |
| `weight_decay` | 衰减系数。 |
| `lr` | 学习率。 |
| `betas` | SophiaG beta 参数。 |
| `rho` | SophiaG clipping/缩放参数。 |

输出：`SophiaG` optimizer。

行为：把参数分两组：需要 weight decay 的参数、不需要 weight decay 的参数。

### 11.2 `src/optim/sophia.py::SophiaG`

功能：实现 SophiaG 优化器。

关键方法：

| 方法 | 输入 | 输出/副作用 |
| --- | --- | --- |
| `__init__(params, lr, betas, rho, weight_decay, maximize=False, capturable=False)` | 参数组和超参数 | 初始化 optimizer。 |
| `update_hessian()` | 无 | 用当前梯度平方更新 hessian 估计。 |
| `step(closure=None, bs=5120)` | 可选 closure、batch size | 根据 exp_avg 和 hessian 更新参数。 |

当前主流程中只有 `sophia=True` 时使用 SophiaG，否则用 AdamW。

---

## 12. 评估功能

### 12.1 `src/model/unlearn_conflict.py` 中的评估调度

当前训练结束后 `Unlearn.eval(logger)` 调用：

1. WMDP/MMLU：当 forget 名包含 `WMDP` 时，用 `eval_few_shots(..., task_list=["wmdp"])` 与 `task_list=["mmlu"]`。
2. SafePku harmful：当 forget 为 `SafePku` 时，用 `eval_toxic`。
3. Retain test accuracy：对 `test_datasets` 调 `eval_acc`。
4. Downstream test accuracy：对 `downstream_datasets` 调 `eval_acc`。
5. 通用 few-shot：调用默认 `eval_few_shots`。

### 12.2 `src/metrics/simple_accuracy.py::eval_acc(...)`

输入：

| 参数 | 作用 |
| --- | --- |
| `model_name` | checkpoint 路径或模型名。 |
| `retain_dataset` | test/retain dataset。 |
| `output_dir` | 写入 `accuracy.json` 的目录。 |
| `batch_size` | eval batch size。 |
| `device` | 默认 `cuda`。 |

输出：accuracy 百分比 float。

行为：

1. 从 `model_name` 加载模型和 tokenizer。
2. 如果传入的是 `UnlearnDataset`，尝试取其 retain_dataset；否则直接使用普通 dataset。
3. 对每个样本只比较最后一个有效 label token 的预测是否正确。
4. 保存：
   ```text
   {output_dir}/accuracy.json
   ```

### 12.3 `src/metrics/few_shots.py::eval_few_shots(...)`

输入：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `model_name` | 必填 | checkpoint 路径或模型名。 |
| `task_list` | boolq、rte、hellaswag、winogrande、arc、openbookqa、piqa、truthfulqa | lm-eval 任务列表。 |
| `output_path` | `.` | lm-eval 输出路径。 |

输出：无 Python 返回。副作用是执行 `lm_eval` 子进程并输出结果文件。

实际命令核心：

```bash
lm_eval --model hf \
  --model_args pretrained={model_name},cache_dir=./.cache,device_map=auto,parallelize=True \
  --tasks {tasks} \
  --batch_size 16 \
  --output_path {output_path}
```

### 12.4 `src/metrics/ppl.py::eval_ppl(...)`

功能：用 `lm_eval` 计算 perplexity。

默认任务：`wikitext`。

输入输出与 `eval_few_shots` 类似。

### 12.5 `src/metrics/wmdp.py::eval_wmdp(...)`

输入：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `model_name` | 必填 | checkpoint 路径。 |
| `output_dir` | `.` | 输出目录。 |
| `batch_size` | `8` | batch size。 |

输出：Cyber accuracy。副作用是保存 `wmdp_generation.json`。

行为：

1. 加载模型和 tokenizer。
2. 构造 WMDP cyber test 与 bio test。
3. 对 A/B/C/D 四个标签 token 的最后位置 logits 做 softmax。
4. 选最大概率标签作为预测。
5. 保存 cyber/bio accuracy、题目、真值、预测。

### 12.6 `src/metrics/toxic.py`

#### `eval_toxic(model_name, batch_size=128, dataset_seed=8888, fraction=1.0, output_dir=".", dataset=None)`

输入：

| 参数 | 作用 |
| --- | --- |
| `model_name` | checkpoint 路径。 |
| `batch_size` | 生成与分类 batch size。 |
| `dataset_seed`, `fraction` | 控制 PKU/RealToxic 数据采样。 |
| `output_dir` | 输出目录。 |
| `dataset` | 可选 unlearn dataset，用于 forget set 毒性评估。 |

输出：无返回。保存：

- `forget.json`：若传入 dataset，保存 forget 生成毒性。
- `harmful.json`：保存 PKU harmful 与 RealToxicityPrompts 毒性结果。

依赖：`unitary/toxic-bert` pipeline。

子函数：

| 函数 | 输入 | 输出 |
| --- | --- | --- |
| `eval_toxic_forget(model, tokenizer, dataset, batch_size)` | 内存模型、tokenizer、forget dataset | toxic rate、mean score、分类结果、生成文本。 |
| `eval_real_toxic(model, tokenizer, ...)` | 模型、tokenizer | RealToxicityPrompts 分类结果与生成文本。 |
| `eval_pku_toxic(model, tokenizer, ...)` | 模型、tokenizer | PKU prompt 分类结果与生成文本。 |

### 12.7 `src/metrics/Tofu.py::eval_tofu(...)`

功能：综合评估 ToFU forget/retain/real_authors/world_facts。

输入：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `model_name` | 必填 | checkpoint 路径。 |
| `forget_subset` | `forget01` | 目标遗忘子集。 |
| `retain_subset` | `retain99` | 保留子集。 |
| `output_dir` | `.` | 输出目录。 |
| `if_llama` | False | Llama prompt 模板。 |
| `if_system` | False | 是否加入系统提示，要求拒绝作者信息。 |

主要指标/输出：

- truth ratio
- truth probability
- RougeL recall
- 语义匹配 accuracy
- MIA AUC
- KS test
- 生成答案列表

依赖：`SentenceTransformer("paraphrase-MiniLM-L6-v2")`、Rouge、SciPy KS、sklearn ROC AUC。

子函数：

| 函数 | 作用 |
| --- | --- |
| `compute_prob(...)` | 计算指定 prompt-answer 的平均 token 概率。 |
| `generate_answer(...)` | 对问题生成回答。 |
| `eval_tofu_forget(...)` | 评估 forget 子集遗忘情况。 |
| `eval_tofu_retain(...)` | 评估 retain 子集保留情况。 |
| `eval_tofu_other(...)` | 评估 real_authors/world_facts。 |
| `MIA(...)` | 对 ToFU forget/retain 构造 min-k prob AUC。 |

### 12.8 `src/metrics/copyright.py::eval_copyright(...)`

功能：评估 Harry Potter 文本/QA 的版权记忆泄漏。

输入：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `model_name` | 必填 | checkpoint 路径。 |
| `batch_size` | `128` | 生成 batch size。 |
| `output_dir` | `.` | 输出目录。 |
| `if_llama` | False | 模板控制。 |

输出：保存 `copyright.json`。

子功能：

- `eval_leakage_rate`：生成续写，与 ground truth response 计算 BLEU 和 RougeL。
- `eval_privacy_score`：基于 PPL、lowercase ratio、zlib、min-k prob 计算隐私/记忆分数，目前主函数中相关代码被注释。

### 12.9 `src/metrics/PII.py::eval_PII(...)`

功能：评估邮件等 PII 信息抽取能力或泄露程度。

输入：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `model_name` | 必填 | checkpoint 路径。 |
| `output_dir` | `.` | 输出目录。 |
| `batch_size` | `8` | 生成 batch size。 |

输出：保存 `PII.json`。

子功能：

| 函数 | 作用 |
| --- | --- |
| `generate_responses(model, tokenizer, prompts)` | 批量生成 prompt 后续文本。 |
| `extract_first_email(text)` | 正则提取第一个 email。 |
| `eval_context_extraction(...)` | 从 context prompt 中直接抽取 email，返回准确率。 |
| `eval_few_shots_extraction(...)` | one-shot/two-shot/domain/non-domain 多模板抽取评估。 |

数据来源：`files/data/PII/*.jsonl` 与 `prompt_template.json`。

### 12.10 `src/metrics/MIA.py::eval_MIA(...)`

功能：Membership Inference Attack 评估。

输入：

| 参数 | 作用 |
| --- | --- |
| `model_name` | 目标模型路径。 |
| `ref_model_name` | 参考模型路径。 |
| `dataset` | 例如 WikiMIA 数据集。 |
| `dataset_seed` | 拆分种子。 |
| `fraction` | forget 比例。 |
| `output_dir` | 输出目录。 |

输出：保存 `MIA.json`。

主要指标：

- PPL
- 目标模型与参考模型 PPL 差值
- lowercase PPL ratio
- zlib ratio
- min-k probability

攻击方式：用 retain/test 作为 shadow train/test 训练 SVC，再预测 forget 是否像训练成员。

---

## 13. 日志与输出文件

### 13.1 `src/loggers/base.py::BaseLogger`

抽象接口：

| 方法 | 输入 | 输出/作用 |
| --- | --- | --- |
| `log(data)` | dict | 记录指标。 |
| `truncate(epoch)` | int | 截断日志。 |
| `save_ckpt(name, data)` | 名称、对象 | 保存 checkpoint。 |
| `load_ckpt(name)` | 名称 | 加载 checkpoint。 |
| `save_img(name, data)` | 名称、图片对象 | 保存图片。 |

### 13.2 `src/loggers/json_/main.py::JSONLogger`

输入：

| 参数 | 作用 |
| --- | --- |
| `root` | 日志根目录。 |
| `name` | run 名。 |
| `config` | 完整配置 dict。 |

输出目录结构：

```text
{root}/{name}/
  config.json
  log.json
  checkpoints/
  images/
```

主要方法：

| 方法 | 作用 |
| --- | --- |
| `log(data)` | 给 data 加上 start/current/relative time，追加到 `log.json`。 |
| `truncate(epoch)` | 保留日志前 epoch 条。 |
| `save_ckpt(name, model, use_lora)` | 保存模型；LoRA 时可 merge 后保存。当前 `Unlearn.save` 直接用 model.save_pretrained，不主要依赖该方法。 |
| `load_ckpt(name, device="cpu")` | 从 `ckpt_root` 加载模型。 |
| `clear_ckpt_root()` | 清空 checkpoint 目录。 |
| `save_img(name, img)` | 保存 PNG。 |
| `get_root()` | 返回绝对日志根目录。 |

### 13.3 `src/loggers/none_.py::NoneLogger`

功能：空日志器。`log(data)` 只 print，其它方法基本 no-op。

### 13.4 `src/loggers/wandb_.py::WANDBLogger`

功能：初始化 wandb 并 `wandb.log(data)`。

输入：`root`、`name`、`config`、`project`。

注意：该类没有完整实现 `BaseLogger` 的所有抽象方法，当前主配置可选项也只包含 `json` 和 `none`。

---

## 14. 额外执行脚本与调试工具

### 14.1 `src/exec/Fine_tune_hp.py`

功能：对 Harry Potter 数据做普通微调。

输入参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--model_name` | required | 初始模型。 |
| `--cache_dir` | `.cache` | 缓存目录。 |
| `--seed` | 0 | 随机种子。 |
| `--epochs` | 10 | 训练 epoch。 |
| `--lr` | `1e-5` | 学习率。 |
| `--num_warmup_steps` | 0 | warmup steps。 |
| `--batch_size` | 4 | batch size。 |
| `--save_dir` | `files/models/hp` | 模型保存目录。 |

流程：加载 `HP.build_pretrain_dataset`，创建 `Trainer`，训练并保存。

### 14.2 `src/exec/check_batch_indices.py`

功能：检查 test/retain batch 中 `input_ids` 与 `label` 是否越过 `config.vocab_size` 或 `len(tokenizer)`，避免 GPU cross entropy device-side assert。

主要输入参数：

- `--model_name`
- `--cache_dir`
- `--forget_dataset_name`
- `--retain_dataset_name`
- `--dataset_seed`
- `--forget_ratio`
- `--self_retain`
- `--batch_size`
- `--max_batches`
- `--datasets`，可为 `test`、`downstream`、`all`
- `--local_files_only`

主要输出：打印越界统计，包括：

- `input_ids_neg`
- `input_ids_ge_config_vocab`
- `input_ids_ge_len_tokenizer`
- `shift_labels_bad_vs_config_vocab`
- `shift_labels_bad_vs_len_tokenizer`

---

## 15. 配置文件功能

目录：`configs/unlearn/`

当前存在：

- `conflict_stable.json`
- `Tofu/PO+WAGLE.json`
- `Tofu/GradDiff+WAGLE.json`
- `Tofu/NPO+WAGLE.json`
- `wmdp/NPO+WAGLE.json`
- `wmdp/GradDiff+WAGLE.json`

配置文件通常包含：

| 分组 | 作用 |
| --- | --- |
| `overall` | 模型、logger、cache、seed。 |
| `unlearn` | safe/conflict 方法、mask 路径、epoch、lr、optimizer、LoRA、任务类型等。 |
| `dataset` | forget/retain 数据集、采样比例、batch size。 |
| `logger` | JSON 日志输出目录。 |

示例 `conflict_stable.json` 使用：

- 模型：`HuggingFaceH4/zephyr-7b-beta`
- forget：`WMDPCyber`
- retain：`sst2`
- safe 方法：`GA`
- conflict 方法：`GA+FT`
- safe/conflict mask：外部路径下 `.pt` 文件
- task：`downstream`

---

## 16. 从初始化到评估的完整执行链路

### 16.1 输入准备

用户通过默认值、配置文件或命令行提供：

1. 模型名 `overall.model_name`。
2. 数据集选择 `dataset.forget_dataset_name`、`dataset.retain_dataset_name`。
3. mask 路径 `unlearn.safe_mask_path`、`unlearn.conflict_mask_path`。
4. unlearning 方法 `unlearn.safe_unlearn_method`、`unlearn.conflict_unlearn_method`。
5. 训练超参数，如 `num_epochs`、`lr`、`batch_size`、`gradient_accumulation_steps`。
6. 评估任务类型 `task_name`。
7. 日志目录 `logger.json.root` 与 run name。

### 16.2 配置与调度器构造

1. `Main.make_config()` 收集配置。
2. `Main.setup_seed()` 固定随机性。
3. `Main.init_model()` 合并配置，构造 `dataset_names`。
4. 动态导入 `model.unlearn_conflict.get(**kwargs)`。
5. `get(**kwargs)` 返回 `Unlearn(**kwargs)`。
6. `Main.init_logger()` 创建 JSON 或 None logger。
7. `Main.run()` 调用 `Unlearn.run(logger)`。

### 16.3 模型初始化

1. `Unlearn.init_model()` 调用 `AutoModelForCausalLM.from_pretrained(...)`。
2. 加载 tokenizer。
3. 补 pad token。
4. 可选 LoRA。
5. 调整 embedding 大小。

### 16.4 数据初始化

1. `Unlearn.init_dataset()` 调用 `get_dataset(...)`。
2. `get_dataset` 加载 forget 数据集。
3. 加载一个或多个 retain 数据集。
4. 创建不在 retain 中的 downstream test 数据集。
5. 构造 `UnlearnDataset`。
6. 返回 collator。
7. 推导 `max_steps` 与 `steps_per_epoch`。

### 16.5 Mask 初始化

1. `Unlearn.init_mask(logger)` 加载 safe/conflict mask。
2. 若 mask 文件不存在，调用 `_generate_mask`。
3. `_generate_mask` 从路径解析 score_type 和 ratio。
4. `GenerateMask` 根据 score_type 计算并保存 mask。
5. `_move_mask_to_device` 将 mask 放到对应层权重设备。

### 16.6 Optimizer 与 unlearner 初始化

1. `Unlearn.init_optimizer()` 创建 AdamW 或 SophiaG。
2. `_init_conflict_unlearners` 创建 safe/conflict 两个 unlearner。
3. 每个 unlearner 持有同一个模型、同一个训练数据、对应 mask、loss 超参数。

### 16.7 训练

1. 保存初始 checkpoint。
2. in-memory retain accuracy 评估。
3. 进入 `_custom_conflict_training_loop`。
4. 每个 batch 计算 conflict unlearner loss。
5. backward。
6. 梯度裁剪。
7. 梯度累积到步数后套 mask。
8. optimizer step。
9. 每个 epoch 保存 checkpoint 并评估 retain accuracy。
10. 训练结束保存 final checkpoint。

### 16.8 最终评估

1. 释放内存模型，选择最新 checkpoint。
2. 若 WMDP forget，跑 WMDP 和 MMLU lm-eval。
3. 若 SafePku forget，跑 toxic 评估。
4. 对 retain test 数据集跑 `eval_acc`。
5. 对 downstream test 数据集跑 `eval_acc`。
6. 跑默认 few-shot lm-eval。
7. 结果写入 logger 目录下 JSON 文件或 lm-eval 输出文件。

---

## 17. 当前功能边界与注意事项

这些不是第二步修改需求，只是当前代码行为记录，后续改动时需要留意。

1. `Edge-Pruning/` 不在主训练循环中执行，主流程只消费它或 `GenerateMask` 生成的 `.pt` mask。
2. `safe_unlearner` 与 `conflict_unlearner` 都会创建，但当前 `_custom_conflict_training_loop` 中 safe/conflict 交替逻辑被注释，实际固定使用 conflict unlearner。
3. `alternate_frequency` 当前只打印，不实际控制交替。
4. `get_unlearn_method` 中 L1 方法名为 `l1sparse`，但 fastargs 配置可选项写的是 `l1_sparse`，两者不一致。
5. `GenerateMask.__init__` 接收 `ratios`，但当前直接覆盖为 `[0.0, 0.1, 0.2, 0.5, 0.8, 0.9]`。
6. 当前 `Unlearn.eval` 只对 `task_name == "downstream"` 有完整分支，虽然 metrics 中存在 tofu、copyright、PII、MIA 等函数。
7. `wandb_.py` 没有完整实现 `BaseLogger` 的所有抽象接口，当前主配置只允许 `json` 和 `none`。
8. `accuracy.py` 与 `simple_accuracy.py` 都实现了 `eval_acc`，当前主流程使用的是 `metrics.simple_accuracy.eval_acc`。
9. 当前训练代码在 batch 移动设备时使用 `self.model.device`，对于 `device_map="auto"` 的多 GPU 模型，需要注意模型对象是否总有该字段。
10. 当前环境需要保持 `torch==2.1.1`、`transformers==4.37.2`、`peft==0.10.0`、`numpy==1.26.4`、`mkl==2023.1.0` 等兼容版本，避免导入错误影响训练入口。

---

## 18. 当前第一步结论

当前项目已经具备一个较完整的 LLM unlearning 实验框架：

1. 支持基于 fastargs/config 的实验配置。
2. 支持 HuggingFace causal LM 加载、LoRA 可选包装、CPU 调试模式。
3. 支持 forget/retain/downstream 多类数据集构造。
4. 支持多个 retain 数据集随机采样组合训练。
5. 支持基于 `.pt` mask 的参数/神经元/线性层权重梯度冻结。
6. 支持内部生成多种 mask，包括 gradient、weight、random、SNIP 类方法和 Wanda。
7. 支持 FT、GA、GA+FT、GA+KL、KL、CL、CL+FT、CL+KL、RL、NPO、NPO+FT 等 unlearning loss。
8. 支持 AdamW 和 SophiaG 优化器。
9. 支持按 epoch 保存 checkpoint 和 retain accuracy 评估。
10. 支持最终 downstream、WMDP、toxic、few-shot、PPL、TOFU、copyright、PII、MIA 等评估函数，其中主流程当前主要自动调用 downstream/WMDP/toxic/few-shot/retain accuracy。

第二步文档可以在本文档后继续增加“目标修改功能清单”，逐项说明要新增、替换或删除的模块，以及每个改动对上述输入输出链路的影响。

---

## 第二步：目标修改需求清单

本步骤的目标是把当前项目从 `LLM unlearning` 框架改造成 `LLM finetuning` 框架。新的任务定义为：在 `target task` 上学习并提升表现，同时尽量保持 `pervasiveness task` 上已有能力不下降。

因此，代码不再以“遗忘某个任务”为中心，而是以“学习目标任务 + 保持泛化能力/已有能力”为中心。当前第一步文档中所有与 `unlearn`、`forget`、`retain`、拒绝标签、梯度上升遗忘、安全/冲突双 mask、遗忘专用评估相关的模块都需要调整。

---

## 19. 总体改造目标

### 19.1 新任务语义

当前任务语义：

```text
forget task: 需要遗忘或拒答的知识。
retain task: 需要保持不下降的能力。
训练目标: 降低 forget task 表现，保持 retain task 表现。
```

目标任务语义：

```text
target task: finetuning 需要学习和提升的任务。
pervasiveness task: LLM 原本已经具备、finetuning 后需要尽量保持的任务。
训练目标: 提升 target task 表现，保持 pervasiveness task 表现。
```

### 19.2 五类必须修改的功能

1. 命名体系修改：`unlearn -> finetuning`，`forget -> target`，`retain -> pervasiveness`。
2. Loss 修改：删除或重写遗忘导向 loss，只保留 finetuning 合理的目标任务学习 loss 与 pervasiveness 保持 loss。
3. 数据集修改：删除空标签、拒绝标签、固定 `I don't know` 类标签构造，target label 必须是正确 next token。
4. 评估修改：删除 unlearning 知识遗忘评估，只保留 target task 和 pervasiveness task 评估。
5. Mask 与训练流程修改：删除 safe/conflict 双 mask 与交替训练，只保留单一 mask 加载和单一训练流程。

---

## 20. 第一类修改：命名体系从 unlearning 改为 finetuning

本节对应第一步中的第 1、2、3、4、6、7、8、9、12、15、16、17 节。

### 20.1 顶层目录、文件、类、函数命名修改

需要把项目主流程中的 unlearning 命名统一替换为 finetuning 命名。建议不要只做机械字符串替换，而是按模块语义重命名。

| 当前名称 | 修改后名称 | 修改位置 | 修改说明 |
| --- | --- | --- | --- |
| `src/exec/unlearn_model_conlict.py` | `src/exec/finetuning_model.py` | 主入口文件 | 删除 `conlict` 拼写遗留问题，并表明入口是 finetuning。 |
| `src/model/unlearn_conflict.py` | `src/model/finetuning.py` | 核心调度器文件 | 删除 conflict 语义，改为单一 finetuning 调度器。 |
| `src/unlearn/` | `src/finetuning/` | loss/trainer/mask 包 | package 语义从遗忘改为微调。 |
| `model.unlearn_conflict.get` | `model.finetuning.get` | 动态导入 | `Main.init_model` 中的 import path 要同步修改。 |
| `Unlearn` | `Finetuning` | 核心调度类 | 表示当前任务是 finetuning 调度器。 |
| `get_unlearn_method` | `get_finetuning_method` | 方法工厂函数 | 按 finetuning method 返回 trainer/loss 类。 |
| `BaseTrainer` | `BaseFinetuningTrainer` | trainer 基类 | 表明该 Trainer 只负责 finetuning loss 与 mask 梯度控制。 |
| `UnlearnDataset` | `FinetuningDataset` | 数据组合类 | 组合 target/pervasiveness 数据。 |
| `unlearncollector` | `finetuning_collator` | collator | 输出 target/pervasiveness batch。 |
| `unlearn_dataset` | `finetuning_dataset` | 调度器字段 | 训练数据对象。 |
| `unlearn_collator` | `finetuning_collator` | 调度器字段 | 训练 collator。 |
| `unlearner` | `finetuning_trainer` 或 `trainer` | 调度器字段 | 当前只保留一个训练器。 |

### 20.2 配置分组命名修改

第一步文档第 2.1 节中 `unlearn` 和 `dataset` 配置需要改为如下结构。

#### 当前配置结构

```json
{
   "unlearn": {
      "safe_unlearn_method": "GA",
      "conflict_unlearn_method": "GA+FT",
      "safe_mask_path": "...",
      "conflict_mask_path": "...",
      "alternate_frequency": 1
   },
   "dataset": {
      "forget_dataset_name": "WMDPCyber",
      "retain_dataset_name": "sst2",
      "forget_ratio": 400,
      "self_retain": false
   }
}
```

#### 目标配置结构

```json
{
   "finetuning": {
      "finetuning_method": "TargetFT+PervasivenessFT",
      "mask_path": "...",
      "num_epochs": 6,
      "lr": 1e-6,
      "weight_decay": 0.1,
      "gradient_accumulation_steps": 8,
      "max_grad_norm": 0.5,
      "sophia": false,
      "resume_path": null,
      "max_steps": -1,
      "use_lora": false,
      "target_weight": 1.0,
      "pervasiveness_weight": 1.0,
      "kl_weight": 1.0,
      "alpha": 0.0,
      "p": 0.01,
      "q": 0.01,
      "mu": 1e-6
   },
   "dataset": {
      "target_dataset_name": "sst2",
      "pervasiveness_dataset_name": "mmlu,hellaswag,winogrande",
      "dataset_seed": 1000,
      "target_ratio": 400,
      "batch_size": 1
   },
   "evaluation": {
      "target_eval": true,
      "pervasiveness_eval": true,
      "pervasiveness_lm_eval_tasks": "mmlu,hellaswag,winogrande,arc_easy,arc_challenge,boolq,piqa",
      "eval_batch_size": 8
   }
}
```

### 20.3 变量名映射总表

以下变量需要在代码、配置、日志输出、checkpoint 目录、评估结果文件中统一修改。

| 当前变量/字段 | 修改后变量/字段 | 说明 |
| --- | --- | --- |
| `unlearn` | `finetuning` | 配置分组、package、日志语义。 |
| `unlearn_method` | `finetuning_method` | 当前选择的微调 loss 方法。 |
| `safe_unlearn_method` | 删除 | 不再区分 safe/conflict。 |
| `conflict_unlearn_method` | `finetuning_method` | 只保留一个训练方法。 |
| `forget` | `target` | batch key、dataset key、loss 变量前缀。 |
| `retain` | `pervasiveness` | batch key、dataset key、loss 变量前缀。 |
| `forget_dataset_name` | `target_dataset_name` | 目标任务数据集名称。 |
| `retain_dataset_name` | `pervasiveness_dataset_name` | 保持能力的数据集名称，支持逗号分隔多个数据集。 |
| `forget_ratio` | `target_ratio` | target 数据抽样数量或比例。 |
| `self_retain` | 删除或改为 `target_holdout_as_pervasiveness` | finetuning 中不建议默认把 target 剩余部分当 pervasiveness；若保留，需要显式表达含义。 |
| `forget_dataset` | `target_dataset` | target 训练集。 |
| `retain_dataset` | `pervasiveness_dataset` | pervasiveness 训练集。 |
| `retain_datasets` | `pervasiveness_datasets` | 多个 pervasiveness 训练集。 |
| `unlearn_dataset` | `finetuning_dataset` | target/pervasiveness 混合训练数据对象。 |
| `unlearn_collator` | `finetuning_collator` | 训练 batch collator。 |
| `safe_mask_path` | 删除 | 不再需要 safe mask。 |
| `conflict_mask_path` | `mask_path` | 单一 mask 文件路径。 |
| `safe_mask` | 删除 | 不再保存。 |
| `conflict_mask` | `mask` | 单一训练 mask。 |
| `alternate_frequency` | 删除 | 不再交替训练。 |
| `safe_unlearner` | 删除 | 单 trainer。 |
| `conflict_unlearner` | `finetuning_trainer` | 当前训练器。 |
| `_init_conflict_unlearners` | `init_finetuning_trainer` | 创建单一 trainer。 |
| `_run_conflict_training` | `_run_finetuning_training` | 启动训练。 |
| `_custom_conflict_training_loop` | `_custom_finetuning_training_loop` | 单一训练循环。 |
| `eval_accuracy_in_memory` | `eval_pervasiveness_accuracy_in_memory` 或 `eval_task_accuracy_in_memory` | 每 epoch 后评估 pervasiveness 或指定 task。 |
| `task_name` | `evaluation_mode` 或删除 | 当前 `downstream` 分支不再适合，应拆成 target/pervasiveness eval 开关。 |

### 20.4 `Main` 类需要修改的内容

对应第一步第 3 节。

| 当前函数 | 修改后函数 | 修改要求 |
| --- | --- | --- |
| `Main.make_config()` | 保持名称 | argparse 描述从 `LLM unlearning` 改为 `LLM finetuning`；fastargs section 从 `unlearn` 改为 `finetuning`。 |
| `Main.init_model(model_name)` | 保持名称 | 合并配置时读取 `finetuning` 分组；构造 `dataset_names={"target": ..., "pervasiveness": ...}`；动态导入 `model.finetuning`。 |
| `Main.run()` | 保持名称 | 调用 `self.model.run(self.logger)`，但 `self.model` 实例类型应是 `Finetuning`。 |

`dataset_names` 的目标结构：

```python
dataset_names = {
      "target": target_dataset_name,
      "pervasiveness": pervasiveness_dataset_name_or_list,
}
```

---

## 21. 第二类修改：Loss 从遗忘目标改为微调目标

本节对应第一步第 8 节。

当前 `src/unlearn` 中大量 loss 是为遗忘设计的，例如梯度上升、拒绝回答训练、负偏好优化、让模型偏离参考模型等。这些目标和 finetuning “学习 target task” 相冲突，需要删除或重写。

### 21.1 Loss 方法保留、删除、重写表

| 当前方法 | 当前作用 | 目标处理 | 修改后建议名称 | 修改说明 |
| --- | --- | --- | --- | --- |
| `FT` | 在 retain 数据上做 CE | 重写 | `TargetFT` | 改为在 target 数据上做标准 next-token CE。 |
| `FT_l1` | retain CE + L1 | 重写 | `TargetFT_L1` | 改为 target CE + 可选 pervasiveness CE/KL + L1 正则。 |
| `GA` | forget CE 梯度上升 | 删除 | 无 | 梯度上升会降低 target 表现，与 finetuning 冲突。 |
| `GA+FT` | forget 梯度上升 + retain CE | 删除/重写为新方法 | `TargetFT+PervasivenessFT` | 不保留 GA，只保留 target CE + pervasiveness CE。 |
| `GA+KL` | forget 梯度上升 + retain KL | 删除/重写为新方法 | `TargetFT+PervasivenessKL` | 不保留 GA，只保留 target CE + pervasiveness KL。 |
| `KL` | 最大化 forget 与 reference 的差异，保持 retain KL | 重写 | `TargetFT+PervasivenessKL` | KL 只用于 pervasiveness 保持，不再用于 target 遗忘。 |
| `KL_GA` | KL + GA 变体 | 删除 | 无 | 仍包含遗忘导向。 |
| `KL_CL` | KL + refusal label | 删除 | 无 | 仍包含拒绝标签。 |
| `CL` | 把 forget prompt 训练成 refused_label | 删除 | 无 | finetuning label 应为正确答案，不应默认拒答。 |
| `CL+FT` | refusal label + retain CE | 删除 | 无 | 不再使用 refusal label。 |
| `CL+KL` | refusal label + retain KL | 删除 | 无 | 不再使用 refusal label。 |
| `RL` | 用 retain label 替换 forget answer | 删除 | 无 | target 应学习正确 target label，而不是 pervasiveness label。 |
| `NPO` | 负偏好优化，让模型不偏好 forget answer | 删除 | 无 | 目标方向与学习 target 冲突。 |
| `NPO+FT` | NPO + retain CE | 删除 | 无 | 仍是遗忘导向。 |
| `CUT` | steering vector 雏形 | 暂不纳入主流程 | 无 | 若后续做 steering finetuning，再单独设计。 |

### 21.2 新的 finetuning loss 方法

建议 `src/finetuning/__init__.py::get_finetuning_method` 只暴露以下方法。

#### `TargetFT`

功能：只在 target task 上做标准监督微调。

输入 batch：

```python
inputs["target"] = (input_ids, attention_mask, labels)
```

变量说明：

| 变量 | 作用 |
| --- | --- |
| `target_input_ids` | target 样本 token。 |
| `target_attention_mask` | target 样本 attention mask。 |
| `target_labels` | 正确 next-token label；prompt 部分可为 -100，answer 部分必须是真实答案 token。 |

输出：

```python
loss = target_loss
```

其中：

```python
target_loss = model(
      input_ids=target_input_ids,
      attention_mask=target_attention_mask,
      labels=target_labels,
).loss
```

#### `TargetFT+PervasivenessFT`

功能：学习 target，同时用 pervasiveness CE 保持已有任务能力。

输入 batch：

```python
inputs["target"] = (target_input_ids, target_attention_mask, target_labels)
inputs["pervasiveness"] = (pervasiveness_input_ids, pervasiveness_attention_mask, pervasiveness_labels)
```

输出：

```python
loss = target_weight * target_loss + pervasiveness_weight * pervasiveness_loss
```

新增配置变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `target_weight` | `1.0` | target CE loss 权重。 |
| `pervasiveness_weight` | `1.0` | pervasiveness CE loss 权重。 |

#### `TargetFT+PervasivenessKL`

功能：学习 target，同时让模型在 pervasiveness 数据上的输出分布接近 finetuning 前的 reference model。

输入：

```python
inputs["target"] = (target_input_ids, target_attention_mask, target_labels)
inputs["pervasiveness"] = (pervasiveness_input_ids, pervasiveness_attention_mask, pervasiveness_labels)
reference_model = deepcopy(initial_model).eval()
```

输出：

```python
loss = target_weight * target_loss + kl_weight * pervasiveness_kl_loss
```

变量说明：

| 变量 | 作用 |
| --- | --- |
| `reference_model` | finetuning 前冻结模型，用于 pervasiveness 保持。 |
| `pervasiveness_logits` | 当前模型在 pervasiveness 样本上的 logits。 |
| `reference_logits` | reference model 在同一 pervasiveness 样本上的 logits。 |
| `pervasiveness_kl_loss` | 当前模型与 reference model 输出分布的 KL。 |
| `kl_weight` | KL 保持项权重。 |

注意：该方法中的 KL 只能用于 pervasiveness 保持，不能再对 target 数据做“远离 reference”的遗忘目标。

#### `TargetFT_L1`

功能：在 target CE 或 target+pervasiveness loss 上加入 L1 稀疏正则。

输出：

```python
loss = base_finetuning_loss + alpha * l1_loss
```

变量说明：

| 变量 | 作用 |
| --- | --- |
| `base_finetuning_loss` | `TargetFT`、`TargetFT+PervasivenessFT` 或 `TargetFT+PervasivenessKL` 的基础 loss。 |
| `alpha` | L1 正则权重。 |
| `l1_loss` | 所有可训练参数绝对值之和。 |

### 21.3 `BaseFinetuningTrainer` 修改要求

对应当前 `src/unlearn/base.py::BaseTrainer`。

需要保留：

1. `mask_gradient(...)`：mask 梯度功能仍然需要。
2. Trainer 基础能力：dataloader、optimizer、scheduler、logging。
3. `reference_model` 支持：只在 `TargetFT+PervasivenessKL` 中启用。

需要修改：

| 当前字段/函数 | 修改后 | 修改要求 |
| --- | --- | --- |
| `infer_model` | `reference_model` | 更准确表达“初始冻结模型”，只用于 pervasiveness KL。 |
| `if_kl` | `use_reference_model` | 语义从遗忘 KL 改为保持 KL。 |
| `compute_metrics` | `compute_task_metrics` | 保留 token loss/accuracy，但命名不要带 unlearning 语义。 |
| `mask_gradient(if_wanda=True)` | 保持 | 仍支持参数名 mask 与 Wanda 数字编号 mask。 |

### 21.4 batch 输入输出协议修改

当前 loss 使用：

```python
inputs["forget"] = (input_ids, attention_mask, label, refused_label, question_length)
inputs["retain"] = (input_ids, attention_mask, label)
```

目标 loss 必须使用：

```python
inputs["target"] = (input_ids, attention_mask, labels)
inputs["pervasiveness"] = (input_ids, attention_mask, labels)
```

字段要求：

| 字段 | 是否保留 | 说明 |
| --- | --- | --- |
| `input_ids` | 保留 | 模型输入。 |
| `attention_mask` | 保留 | attention mask。 |
| `label` 或 `labels` | 保留，建议统一为 `labels` | 正确 next-token 标签。 |
| `refused_label` | 删除 | finetuning 不再默认拒答。 |
| `question_length` | 删除或改为 `prompt_length` | 只在需要 prompt mask 时保留，不能再用于拒绝标签替换。 |

---

## 22. 第三类修改：数据集生成方式改为正确 next-token label

本节对应第一步第 6、7 节。

### 22.1 标准训练样本格式修改

当前第一步第 6.1 节中的 Forget 样本需要完全重写。

#### 当前 forget 样本

```python
{
      "input_ids": ...,
      "attention_mask": ...,
      "label": ...,
      "refused_label": ...,
      "question_length": ...,
}
```

#### 修改后 target 样本

```python
{
      "input_ids": ...,
      "attention_mask": ...,
      "labels": ...,
}
```

变量说明：

| 字段 | 作用 | 生成方式 |
| --- | --- | --- |
| `input_ids` | prompt + 正确答案 token | tokenizer 对完整训练文本编码。 |
| `attention_mask` | 非 padding token 标记 | tokenizer 或 padding 函数生成。 |
| `labels` | 正确 next-token 目标 | 通常复制 `input_ids`，并将 prompt 部分置为 -100；若是纯 LM 语料，则可等于 `input_ids`。 |

如果数据集是 instruction/question-answer 形式，label 规则应为：

```text
prompt 部分: -100，不参与 loss。
answer 部分: 正确答案 token，参与 loss。
```

如果数据集是纯语言建模语料，label 规则应为：

```text
完整文本全部作为 next-token 目标，labels = input_ids。
```

### 22.2 `FinetuningDataset` 修改要求

对应当前 `UnlearnDataset`。

#### 新类签名

```python
class FinetuningDataset(Dataset):
      def __init__(
            self,
            datasets,
            target_ratio,
            dataset_seed,
            target_holdout_as_pervasiveness=False,
      ):
            ...
```

输入变量：

| 参数 | 作用 |
| --- | --- |
| `datasets` | 至少包含 `target`，可包含 `pervasiveness`、`pervasiveness1`、`pervasiveness2` 等。 |
| `target_ratio` | target 样本数量或比例，替代 `forget_ratio`。 |
| `dataset_seed` | target 抽样和 pervasiveness 随机采样种子。 |
| `target_holdout_as_pervasiveness` | 可选功能，只有明确需要时才使用 target 剩余样本作为 pervasiveness。 |

输出样本：

```python
{
      "target": target_sample,
      "pervasiveness": pervasiveness_sample,
}
```

多 pervasiveness 时输出：

```python
{
      "target": target_sample,
      "pervasiveness1": pervasiveness_sample,
}
```

### 22.3 `finetuning_collator(samples)` 修改要求

对应当前 `unlearncollector`。

目标输出：

```python
batch = {
      "target": (target_input_ids, target_attention_mask, target_labels),
      "pervasiveness": (pervasiveness_input_ids, pervasiveness_attention_mask, pervasiveness_labels),
}
```

删除输出：

```python
refused_label
question_length
```

如果后续有多个 pervasiveness 数据集，collator 应保留当前多 key 逻辑：

```python
"pervasiveness1"
"pervasiveness2"
...
```

### 22.4 `get_dataset(...)` 修改要求

对应当前 `src/dataset/__init__.py::get_dataset(...)`。

#### 当前输入

```python
get_dataset(dataset_names, tokenizer, dataset_seed, forget_ratio, self_retain, if_llama=False)
```

#### 修改后输入

```python
get_dataset(
      dataset_names,
      tokenizer,
      dataset_seed,
      target_ratio,
      target_holdout_as_pervasiveness=False,
      if_llama=False,
)
```

#### 当前输出

```python
unlearn_dataset, test_datasets, unlearn_collator, test_collator, downstream_datasets
```

#### 修改后输出

```python
finetuning_dataset, target_test_datasets, pervasiveness_test_datasets, finetuning_collator, test_collator
```

输出变量说明：

| 输出变量 | 作用 |
| --- | --- |
| `finetuning_dataset` | target/pervasiveness 混合训练数据。 |
| `target_test_datasets` | target task 的测试集或验证集。 |
| `pervasiveness_test_datasets` | pervasiveness task 的测试集或验证集。 |
| `finetuning_collator` | 训练 collator。 |
| `test_collator` | 评估 collator。 |

### 22.5 各数据集模块修改清单

#### `SafePkuDataset`

当前问题：当前实现把 unsafe response 作为 forget 内容，同时构造 `refused_label`。这适合 unlearning/refusal，不适合普通 finetuning。

目标修改：

1. 如果 `SafePku` 作为 target safety finetuning 数据集，则 target label 应是安全、正确、期望模型学习的 response。
2. 如果原始样本只有 unsafe response，不能继续把 unsafe response 当作“应学习答案”，除非实验目标明确是学习 unsafe 内容。
3. 删除 polite refusal CSV 的默认加载和 `refused_label` 构造。
4. 输出字段统一为 `input_ids`、`attention_mask`、`labels`。
5. 若目标任务就是“学会拒绝危险问题”，则应把拒绝回答作为 target 正确标签，但必须通过数据集配置显式指定，例如 `target_response_type="refusal"`，不能作为所有 target 数据集的默认行为。

#### `ToFU`

当前问题：ToFU 代码为 forget 子集构造 `refused_label`，评估也强调 forget truth ratio。

目标修改：

1. `target` 子集应使用真实 `answer` 或 `paraphrased_answer` 作为正确 labels。
2. 删除 `polite_refusal_responses_tofu.csv` 默认拒绝标签构造。
3. `forget01`、`forget10` 等名字如果继续使用，会造成语义混乱，建议新增或映射为 `target01`、`target10`，或在配置中写作 `target_subset`。
4. `retain99` 应改名为 `pervasiveness99` 或在代码中仅作为 ToFU 原始 subset 名保留，但外层变量必须叫 `pervasiveness_subset`。
5. 测试输出应支持 target accuracy/ROUGE/semantic similarity，而不是 forget truth ratio。

#### `WMDPCyber`、`WMDPBio`、`WMDPALL`

当前问题：WMDP forget 语料会构造 refusal label 或特殊替换位置。

目标修改：

1. 作为 target 时，训练 label 应是语料正确 next token，即 `labels=input_ids` 或 QA 正确答案 token。
2. 删除 `refused_label` 和随机替换 answer 的逻辑。
3. `subset="forget"`、`subset="retain"` 的外层语义要改为 `subset="target"`、`subset="pervasiveness"`；若底层 HuggingFace 数据配置仍叫 `cyber-forget-corpus`，只能在数据加载内部作为原始数据名使用，不能暴露为训练语义。
4. WMDP 多选题评估如果作为 target task，应评估 target accuracy；如果作为 pervasiveness task，也可以作为保持能力的一项。

#### `HP`

当前问题：HP 会构造版权拒绝回答 `refused_label`，评估偏向泄露/版权记忆。

目标修改：

1. 如果 HP 是 target finetuning 数据集，labels 应是原始 QA 正确 answer 或文本续写 next token。
2. 删除 `polite_refusal_responses_copyright.csv` 默认拒绝标签构造。
3. 版权泄露评估不再作为默认 unlearning 评估；若 HP 是 target，可保留 BLEU/ROUGE 作为 target generation 质量评估。

#### `C4` 与 `wikitext`

当前基本可保留。

修改要求：

1. 外层命名从 retain/pervasiveness 统一。
2. 字段名从 `label` 建议统一为 `labels`。
3. 可作为 pervasiveness 语言建模保持数据。

#### `SST2` 与 `Winogrande`

当前基本适合 finetuning。

修改要求：

1. 若作为 target，answer token 是正确标签，保持现有 QA-style label 逻辑。
2. 若作为 pervasiveness，作为能力保持数据。
3. 字段名统一为 `labels`。

#### IOI、Induction、Docstring、Gender、Bool wrapper

当前基本适合作为 pervasiveness/downstream 保持任务。

修改要求：

1. 外层从 `retain` 或 `downstream` 统一纳入 `pervasiveness`。
2. 若作为 target，也应使用正确答案 token，不得构造拒绝答案。
3. 评估名称中不要再出现 retain/downstream 语义混用，建议统一输出到 `target/` 或 `pervasiveness/` 目录。

### 22.6 数据集配置命名建议

建议将 dataset 配置改为：

| 配置项 | 类型 | 说明 |
| --- | --- | --- |
| `target_dataset_name` | `str` | target 训练数据集。 |
| `pervasiveness_dataset_name` | `str` | pervasiveness 训练数据集，逗号分隔时为多个。 |
| `target_ratio` | `float` | target 样本数量或比例。 |
| `dataset_seed` | `int` | 抽样种子。 |
| `batch_size` | `int` | 训练 batch size。 |
| `target_response_type` | optional str | 对特殊安全数据集指定正确 response 来源，例如 `safe_response` 或 `refusal`。 |
| `target_holdout_as_pervasiveness` | bool | 可选，不建议默认开启。 |

---

## 23. 第四类修改：评估方式只保留 target 与 pervasiveness

本节对应第一步第 12、16.8、17 节。

当前评估包含大量 unlearning 特有指标，例如 toxic forget、ToFU forget truth ratio、copyright leakage、PII/MIA 等。目标框架应删除默认 unlearning 评估，只评估两个问题：

```text
1. target task 学会了吗？
2. pervasiveness task 下降了吗？
```

### 23.1 `Finetuning.eval(logger)` 目标流程

对应当前 `Unlearn.eval(logger)`。

目标流程：

1. 选择 `latest_checkpoint_path` 或 `resume_path`。
2. 加载 target test datasets。
3. 对 target task 调用目标任务评估函数。
4. 加载 pervasiveness test datasets。
5. 对 pervasiveness task 调用 accuracy、PPL 或 lm-evaluation-harness。
6. 写出统一的 `finetuning_eval_summary.json`。

建议输出目录：

```text
{logger_root}/eval/
   target/
      accuracy.json
      generation.json
      task_metrics.json
   pervasiveness/
      dataset_accuracy/
         {dataset_name}/accuracy.json
      lm_eval/
         few_shots.json
         mmlu.json
         wmdp.json
   finetuning_eval_summary.json
```

### 23.2 评估函数保留、删除、重写表

| 当前评估函数 | 当前作用 | 目标处理 | 修改后建议 |
| --- | --- | --- | --- |
| `metrics.simple_accuracy.eval_acc` | retain/downstream accuracy | 保留并重命名 | `eval_task_accuracy`，同时支持 target/pervasiveness。 |
| `metrics.few_shots.eval_few_shots` | 通用 lm-eval | 保留 | 主要用于 pervasiveness 综合评估，也可用于 target benchmark。 |
| `metrics.ppl.eval_ppl` | PPL | 保留为可选 | 可用于 pervasiveness 语言建模保持。 |
| `metrics.wmdp.eval_wmdp` | WMDP MCQ | 保留但语义重写 | 只有当 WMDP 被配置为 target 或 pervasiveness 时才运行。 |
| `metrics.toxic.eval_toxic` | harmful/unlearning 毒性遗忘评估 | 从默认流程删除 | 若 target 是 safety finetuning，可作为 target safety 指标单独配置。 |
| `metrics.Tofu.eval_tofu` | ToFU forget/retain/KS/MIA | 删除默认调用，重写 target 评估 | 保留 answer accuracy、ROUGE、semantic similarity；删除 forget truth ratio、KS forget/retain 对比、MIA。 |
| `metrics.copyright.eval_copyright` | HP 泄露/版权记忆 | 删除默认调用 | 若 HP 是 target，可保留 BLEU/ROUGE generation 质量，不做泄露式遗忘指标。 |
| `metrics.PII.eval_PII` | PII 泄露抽取 | 删除默认调用 | 不属于通用 finetuning 默认评估；除非 target 明确是 PII 抽取任务。 |
| `metrics.MIA.eval_MIA` | membership inference | 删除默认调用 | MIA 是隐私/遗忘安全评估，不属于默认 target/pervasiveness。 |

### 23.3 Target task 评估要求

target task 评估应由 target 数据集类型决定。

| target 数据类型 | 评估方式 | 输出变量 |
| --- | --- | --- |
| 分类/多选生成式任务，如 SST2、Winogrande、WMDP | 最后有效 token accuracy 或 A/B/C/D 概率 accuracy | `target_accuracy`、`correct_predictions`、`total_predictions`。 |
| QA/open-ended 任务，如 ToFU、HP QA | exact match、ROUGE-L、semantic similarity，可选 BLEU | `target_rougeL`、`target_semantic_acc`、`generated_answers`。 |
| 语言建模语料，如 C4、wikitext、WMDP corpus | PPL 或 token loss | `target_ppl`、`target_loss`。 |
| 安全拒答 finetuning | refusal accuracy 或 harmful response rate | `target_refusal_acc`、`target_harmful_rate`，但必须由 `target_response_type="refusal"` 显式启用。 |

### 23.4 Pervasiveness task 评估要求

pervasiveness task 评估用于衡量微调后已有能力是否下降。

推荐包含两类：

1. 项目内部 pervasiveness dataset accuracy：对配置中的 pervasiveness test datasets 调用 `eval_task_accuracy`。
2. `lm-evaluation-harness` 综合评估：通过 `pervasiveness_lm_eval_tasks` 指定，例如：

```text
mmlu,hellaswag,winogrande,arc_easy,arc_challenge,boolq,piqa,truthfulqa
```

目标输出：

```json
{
   "target": {
      "accuracy": 0.0,
      "loss": 0.0
   },
   "pervasiveness": {
      "dataset_accuracy": {
         "sst2": 0.0,
         "winogrande": 0.0
      },
      "lm_eval": {
         "mmlu": 0.0,
         "hellaswag": 0.0
      }
   }
}
```

### 23.5 `task_name` 分支删除或重写

当前 `Unlearn.eval` 主要依赖：

```python
if self.task_name == "downstream":
      ...
```

目标代码不应再以 `downstream` 作为唯一评估模式。建议改为：

```python
if self.target_eval:
      self.eval_target(...)

if self.pervasiveness_eval:
      self.eval_pervasiveness(...)
```

新增配置：

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `evaluation.target_eval` | `true` | 是否评估 target task。 |
| `evaluation.pervasiveness_eval` | `true` | 是否评估 pervasiveness task。 |
| `evaluation.pervasiveness_lm_eval_tasks` | 常用 lm-eval 任务列表 | pervasiveness 综合评估任务。 |
| `evaluation.eval_batch_size` | `8` | 评估 batch size。 |

---

## 24. 第五类修改：删除 safe/conflict 双 mask，只保留单 mask 单训练流程

本节对应第一步第 2、4、9、16、17 节。

当前代码为 unlearning 设计了 `safe_mask` 和 `conflict_mask` 两套 mask，并创建 `safe_unlearner`、`conflict_unlearner`。目标 finetuning 不再需要两套 mask，也不需要交替训练。

### 24.1 配置删除与替换

| 当前配置 | 目标处理 | 修改后配置 |
| --- | --- | --- |
| `safe_mask_path` | 删除 | 无 |
| `conflict_mask_path` | 替换 | `mask_path` |
| `safe_unlearn_method` | 删除 | 无 |
| `conflict_unlearn_method` | 替换 | `finetuning_method` |
| `alternate_frequency` | 删除 | 无 |
| `p`, `q`, `mu` | 保留 | mask 生成仍可能使用。 |
| `task_name` | 删除/替换 | `evaluation.target_eval`、`evaluation.pervasiveness_eval`。 |

### 24.2 `Finetuning.init_mask(logger)` 修改要求

对应当前 `Unlearn.init_mask(logger)`。

#### 当前行为

```python
self.safe_mask = load_or_generate(safe_mask_path)
self.conflict_mask = load_or_generate(conflict_mask_path)
self.mask = self.safe_mask
```

#### 目标行为

```python
self.mask = load_or_generate(mask_path)
```

输入变量：

| 参数 | 作用 |
| --- | --- |
| `mask_path` | 单一 mask 文件路径。存在则加载，不存在则根据 `mask_score_type` 或路径生成。 |
| `logger` | 提供输出目录。 |

输出变量：

| 字段 | 作用 |
| --- | --- |
| `self.mask` | 单一训练 mask，传给 `finetuning_trainer`。 |

### 24.3 `_generate_mask(...)` 修改要求

当前 `_generate_mask` 从 `mask_path` 解析 `score_type` 和 `ratio` 的方式可以保留，但语义要从 forget/retain 改成 target/pervasiveness。

需要修改：

| 当前变量/score | 修改后 | 说明 |
| --- | --- | --- |
| `forget_ratio=128` 的 Wanda 校准数据 | `target_ratio=128` 或单独 `mask_calibration_samples` | Wanda 校准应使用 target 样本，或按配置选择 target/pervasiveness 混合。 |
| `dataset="forget"` | `dataset="target"` | gradient mask 对 target loss 求梯度。 |
| `dataset="retain"` | `dataset="pervasiveness"` | 若需要保护已有能力，可对 pervasiveness loss 求梯度。 |
| `snip_forget_reinit` | 删除或重命名 | 包含 forget 语义，不应保留。 |

建议新增配置：

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `mask_path` | required/optional | 单 mask 路径。 |
| `mask_score_type` | 从路径推断或显式指定 | `gradient`、`weight`、`random`、`wanda` 等。 |
| `mask_ratio` | 从路径推断或显式指定 | 选择可训练/冻结比例。 |
| `mask_calibration_dataset` | `target` | 生成 mask 时使用 target、pervasiveness 或 mixed。 |
| `mask_calibration_samples` | `128` | Wanda/SNIP 校准样本数。 |

### 24.4 单一 trainer 初始化

对应当前 `init_unlearner` 与 `_init_conflict_unlearners`。

#### 当前行为

```python
self.safe_unlearner = get_unlearn_method(..., mask=self.safe_mask)
self.conflict_unlearner = get_unlearn_method(..., mask=self.conflict_mask)
```

#### 目标行为

```python
self.finetuning_trainer = get_finetuning_method(
      self.finetuning_method,
      model=self.model,
      tokenizer=self.tokenizer,
      train_dataset=self.finetuning_dataset,
      data_collator=self.finetuning_collator,
      eval_collector=self.test_collator,
      args=training_args,
      target_weight=self.target_weight,
      pervasiveness_weight=self.pervasiveness_weight,
      kl_weight=self.kl_weight,
      alpha=self.alpha,
      mask=self.mask,
      if_wanda=self.if_wanda,
)
```

输出字段：

```python
self.finetuning_trainer
```

### 24.5 单一训练循环

对应当前 `_custom_conflict_training_loop`。

#### 当前行为

1. 创建 safe/conflict 两个 trainer。
2. 打印 alternate frequency。
3. 实际固定使用 conflict trainer。
4. 每 epoch 保存 checkpoint 并评估 retain accuracy。

#### 目标行为

1. 只创建一个 `finetuning_trainer`。
2. 训练时每个 batch 调用：
    ```python
    loss = self.finetuning_trainer.compute_loss(self.model, batch)
    ```
3. backward 后调用：
    ```python
    self.finetuning_trainer.mask_gradient(self.model, if_wanda=self.if_wanda)
    ```
4. 每 epoch 保存 checkpoint。
5. 每 epoch 可选评估：
    - target in-memory accuracy 或 loss。
    - pervasiveness in-memory accuracy 或 loss。

新函数建议：

| 当前函数 | 修改后函数 | 作用 |
| --- | --- | --- |
| `_run_conflict_training` | `_run_finetuning_training` | 调用单一训练循环。 |
| `_custom_conflict_training_loop` | `_custom_finetuning_training_loop` | 手写训练循环。 |
| `_print_parameter_freeze_report` | 保持或改为 `_print_mask_freeze_report` | 统计单 mask 冻结比例。 |

---

## 25. 对第一步各节的逐项修改索引

本节把第一步文档中需要修改的章节逐一对应到本步骤需求。

| 第一步章节 | 当前内容 | 第二步修改要求 |
| --- | --- | --- |
| 1. 项目整体功能 | LLM unlearning、forget/retain、遗忘评估 | 改为 LLM finetuning、target/pervasiveness、target 提升与 pervasiveness 保持评估。 |
| 2. 顶层执行入口 | `unlearn_model_conlict.py`、`unlearn` 配置、safe/conflict mask | 入口改为 `finetuning_model.py`；配置分组改为 `finetuning`；只保留 `mask_path`。 |
| 3. `Main` 类功能 | 构造 `model.unlearn_conflict.Unlearn` 和 `dataset_names={forget, retain}` | 构造 `model.finetuning.Finetuning` 和 `dataset_names={target, pervasiveness}`。 |
| 4. 核心模型调度器 | `Unlearn`、safe/conflict unlearner、双 mask、conflict training loop | 改为 `Finetuning`、单 `finetuning_trainer`、单 mask、单 training loop。 |
| 5. 旧版辅助模型封装 | `BaseModel` 中仍有 unlearn/recovery/toxic/MIA 等旧语义 | 若保留，需改名和剥离 unlearning 评估；若不用主流程，可标记 deprecated。 |
| 6. 数据结构与数据管线 | Forget 样本含 `refused_label`、`question_length` | Target 样本只保留 `input_ids`、`attention_mask`、`labels`；删除拒绝标签。 |
| 7. 数据集功能清单 | Forget/retain 数据集分类，部分数据构造拒绝标签 | 改为 target/pervasiveness 数据集分类；所有 target label 改为正确 next-token。 |
| 8. Unlearning 方法与 Loss | GA、CL、RL、NPO、forget KL 等遗忘方法 | 删除遗忘方法；新增 `TargetFT`、`TargetFT+PervasivenessFT`、`TargetFT+PervasivenessKL`、`TargetFT_L1`。 |
| 9. Mask 生成与应用 | safe/conflict mask、forget/retain gradient | 单一 mask；mask 生成使用 target/pervasiveness 语义。 |
| 10. Pruner 工具函数 | 层查找、Wanda 工具 | 可保留，只需修改调用侧命名。 |
| 11. 优化器功能 | AdamW/SophiaG | 可保留，配置分组从 unlearn 改为 finetuning。 |
| 12. 评估功能 | WMDP forget、toxic、ToFU forget、copyright、PII、MIA | 默认只保留 target 与 pervasiveness 评估；unlearning 评估从默认流程删除。 |
| 13. 日志与输出文件 | `unlearn_checkpoint`、unlearning config | 日志目录、checkpoint 名称、config.json 内容改为 finetuning 语义。 |
| 14. 额外脚本 | `Fine_tune_hp.py`、batch 检查脚本 | HP finetune 脚本可作为新目标样例；batch 检查脚本变量改为 target/pervasiveness。 |
| 15. 配置文件功能 | `configs/unlearn/` | 改为 `configs/finetuning/`，并重写配置字段。 |
| 16. 完整执行链路 | 初始化 unlearn、构造 forget/retain、训练 conflict、遗忘评估 | 改为初始化 finetuning、构造 target/pervasiveness、单训练、target/pervasiveness 评估。 |
| 17. 当前功能边界 | 记录 unlearning 潜在问题 | 更新为 finetuning 重构注意事项。 |
| 18. 当前第一步结论 | 总结现有 unlearning 框架 | 可保留作为现状；第二步作为目标状态，不覆盖第一步。 |

---

## 26. 目标代码执行链路

改造完成后的理想流程应为：

### 26.1 输入准备

用户提供：

1. `overall.model_name`：初始 LLM。
2. `dataset.target_dataset_name`：需要学习的任务。
3. `dataset.pervasiveness_dataset_name`：需要保持的任务，可以多个。
4. `finetuning.finetuning_method`：微调 loss，例如 `TargetFT+PervasivenessFT`。
5. `finetuning.mask_path`：单一 mask，可为空或不存在时生成。
6. `finetuning.num_epochs`、`lr`、`batch_size` 等训练参数。
7. `evaluation.target_eval`、`evaluation.pervasiveness_eval`、`evaluation.pervasiveness_lm_eval_tasks`。

### 26.2 初始化

1. `Main.make_config()` 加载配置。
2. `Main.init_model()` 构造 `Finetuning` 调度器。
3. `Finetuning.init_model()` 加载模型、tokenizer、可选 LoRA。
4. `Finetuning.init_dataset()` 构造 target/pervasiveness 数据。
5. `Finetuning.init_mask()` 加载或生成单一 mask。
6. `Finetuning.init_optimizer()` 创建 AdamW 或 SophiaG。
7. `Finetuning.init_finetuning_trainer()` 创建单一 trainer。

### 26.3 训练

1. 每个 batch 包含 target 和可选 pervasiveness 样本。
2. `compute_loss` 计算 target CE 和 pervasiveness 保持项。
3. backward。
4. 根据 `mask` 屏蔽不更新参数的梯度。
5. optimizer step。
6. 每 epoch 保存 checkpoint。
7. 每 epoch 可选评估 target/pervasiveness 中间结果。

### 26.4 最终评估

1. 评估 target task 是否提升。
2. 评估 pervasiveness task 是否保持。
3. 输出统一 summary。
4. 不再默认输出 forget truth ratio、toxic forget、MIA、copyright leakage 等 unlearning 评估。

---

## 27. 第二步结论

第二步的核心不是简单改名，而是把项目的训练目标从“降低某类知识表现”改成“提升目标任务表现”。因此所有模块需要围绕以下新约定统一：

1. `target` 是学习对象，loss 必须鼓励模型生成正确答案。
2. `pervasiveness` 是保持对象，loss 和评估都用于防止已有能力下降。
3. `refused_label`、`question_length` 驱动的拒绝回答训练不再是默认能力。
4. GA、CL、RL、NPO 等遗忘 loss 不再适合作为主流程。
5. safe/conflict 双 mask 和双 trainer 不再存在，只保留一个 `mask` 和一个 `finetuning_trainer`。
6. 默认评估只回答 target 是否变好、pervasiveness 是否下降这两个问题。

完成这些修改后，项目应从当前 `LLM unlearning with masked partial finetuning` 重构为 `LLM finetuning with masked parameter update and pervasiveness preservation`。
