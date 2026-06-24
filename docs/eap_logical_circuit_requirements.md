# EAP_forLogicalCircuit 与多任务 Finetuning 功能需求分析

本文档定义一个新的功能增量：在现有 `EAP_forComponent` 与 `finetuning_model.py` 的基础上，新增 `EAP_forLogicalCircuit/`，用于对单任务生成 edge-level circuit、融合得到 logical circuit、对多任务 logical circuit 做冲突分析，并把分析结果接入多任务 finetuning。

文档目标不是直接给出实现代码，而是给出可执行的需求边界、输入输出契约、关键算法语义、训练策略、模块拆分与验证要求，作为后续开发的统一依据。

本文档的主要参考基线有三类：

- 现有 `EAP_forComponent`：已经支持 component/node 级 current/future attribution、mask 输出、LoRA rank 输出、K 采样 future 聚合。
- 现有 `src/model/finetuning.py` 与 `src/exec/finetuning_model.py`：已经支持 target/pervasiveness 单训练循环、单 mask、standard/projection/head 三种 LoRA 模式。
- `docs/eap_ig_implementation.md` 中对 EAP-IG edge/node/circuit 机制的总结：特别是 `induction.py`、`induction_or.py`、`get_logical_edge.py`、`conflict.py` 的语义。

## 1. 需求背景

当前项目已经具备如下能力：

1. `EAP_forComponent` 能对单一任务数据集计算 component-level attribution，并输出：
   - `component_scores.json/pt`
   - `component_mask.pt`
   - `rank_pattern.json`
   - `lora_allocation.json`
   - `summary.json`
2. `finetuning_model.py` 能加载上述单任务 component 结论，执行：
   - 普通全参微调
   - 基于 bool mask 的受限微调
   - 基于 projection-matrix rank pattern 的 PEFT LoRA 微调
   - 基于 component/head score 的自定义 head-wise LoRA 微调
3. 当前多任务能力只停留在 target + 若干 pervasiveness 数据集的 loss 聚合，不支持“每个任务使用不同 mask / 不同 LoRA 信息”的分阶段训练。

新的需求不是替换 `EAP_forComponent`，而是在其旁边新增一个更接近 EAP-IG 原始 edge circuit 逻辑的新系统：`EAP_forLogicalCircuit`。

## 2. 核心目标

新增功能必须同时覆盖两个层面：

1. 可解释性层面：
   - 为单个任务计算 edge-level attribution，而不是 component/node-level attribution。
   - 支持 current 与 future 两种定位方式，其中 future 要延续 `EAP_forComponent` 的 future 算法思想。
   - 从 edge attribution 中构建 `circuit`、`circuit_or`，再融合为 `logical_circuit`。
   - 对多个任务得到的 `logical_circuit` 做冲突分析，输出任务级、冲突级、全集级 component 信息。
2. 训练层面：
   - 在不破坏现有单任务功能的前提下，为 `finetuning_model.py` 增加“多任务 mask / 多任务 LoRA / 两阶段训练”的能力。
   - 第一阶段允许不同任务用不同 mask / 不同 LoRA 配置交替训练。
   - 第二阶段使用所有任务涉及 component 的统一 mask / 统一 LoRA 信息，对所有任务联合训练。

## 3. 非目标

以下内容不属于本次需求的第一版范围：

1. 不要求在 `EAP_forLogicalCircuit` 中自动遍历多个任务并自动完成全量 conflict 分析流水线。多任务处理允许通过多次运行单任务 EAP，再运行单独 conflict 分析步骤完成。
2. 不要求在第一版实现图像可视化或交互式 graph 浏览器。
3. 不要求把 `EAP_forLogicalCircuit` 直接替换 `EAP_forComponent` 的全部输出契约。两者应并存。
4. `conflict_mask_path` 第一版只要求输出与保存，训练中暂不加载使用。
5. 不要求第一版把 edge-level logical circuit 直接作为训练对象。训练接入仍然落在 component-level 结果上。

## 4. 术语定义

为避免混淆，本文档统一使用以下术语：

- `component`：与 `EAP_forComponent` 相同，指 projection matrix 或 attention head slice 或 MLP matrix 等 coarse module unit。
- `node`：EAP-IG 图中的计算节点。对当前 Mistral 路径，通常是 input、attention head、MLP、logit node。
- `edge`：source node 到 destination node input 的连接。
- `circuit`：正向 EAP 语义下选出的 edge 子图。对应 clean 替换 corrupted 的方向。
- `circuit_or`：反向 EAP 语义下选出的 edge 子图。对应 corrupted 替换 clean 的方向。
- `logical_circuit`：由 `circuit` 与 `circuit_or` 融合得到的逻辑电路结果。
- `task component set`：从单任务 `logical_circuit` 中投影得到的 component 集合。
- `conflict components`：多任务 conflict 分析中无法一致赋值、需要视为全任务冲突的 component。
- `all-task components`：所有任务 logical circuit 涉及的 component 的并集，但只保留赋值为 1 或无法赋值的 component；赋值为 0 的 component 不进入输出。

## 5. 总体架构

建议新增独立目录：

```text
EAP_forLogicalCircuit/
    __init__.py
    cli.py
    run_eap_for_logical_circuit.py
    schemas.py
    model_loader.py              # 可复用 EAP_forComponent 逻辑或薄封装
    data.py                      # 可复用 pair dataset 加载
    graph_registry.py            # 定义 edge/node/component 映射
    edge_hooks.py                # 捕获 source->destination edge activation 所需的 hook
    current_localization.py      # edge-level current attribution
    future_localization.py       # edge-level future attribution
    circuit_builder.py           # 根据 score 构建 circuit / circuit_or
    logical_fusion.py            # 融合 circuit 与 circuit_or -> logical_circuit
    conflict_analysis.py         # 多任务 logical circuit 冲突分析
    component_projection.py      # edge/node -> component 聚合
    rank_allocator.py            # component-level LoRA rank 分配，可复用/改造现有实现
    mask_builder.py              # component-level bool mask 生成，可复用/改造现有实现
    outputs.py                   # 保存 task/conflict/all-task outputs
    runner.py
    tests/
```

### 5.1 架构原则

1. 复用现有 `EAP_forComponent` 的模型加载、pair dataset 解析、future K 采样思路、mask/rank 输出风格。
2. 新增逻辑主要放在 edge attribution、circuit 构造、logical 融合、conflict 分析、多任务训练流程。
3. 训练接入层不要直接消费 edge graph，而要消费从 logical circuit 投影出的 component-level artifacts。

## 6. 单任务 EAP_forLogicalCircuit 需求

### 6.1 输入参数

`EAP_forLogicalCircuit` 必须与 `EAP_forComponent` 保持相似 CLI，但语义切换到 edge-level：

| 参数 | 说明 |
| --- | --- |
| `--model_name_or_path` | current/base 模型路径。 |
| `--tokenizer_name_or_path` | tokenizer 路径。 |
| `--future_model_name_or_path` | future/finetuned/probing 模型路径，仅 `future` 模式需要。 |
| `--future_model_cache_dir` | future 模型 cache。 |
| `--future_base_model_name_or_path` | 如果需要与 current model 分开指定 base model，应提供兼容能力；默认可与 `model_name_or_path` 相同。 |
| `--dataset_name` | 单任务数据集名。 |
| `--data_path` | pair CSV 路径。 |
| `--corruption_column` | clean/corrupted 中 corrupted 列名选择。 |
| `--input_format` | `auto/prompt/raw`。 |
| `--metric` | `task_loss/logit_diff`。 |
| `--localization_mode` | `current/future`。 |
| `--future_step_k` | 当 `future_step_k_samples <= 1` 时的单 K。 |
| `--future_step_k_min/max/samples/seed` | 与 `EAP_forComponent` 一致的 future K 采样参数。 |
| `--batch_size` | pair dataloader batch size。 |
| `--max_samples` | attribution 样本上限。 |
| `--max_length` | tokenizer 最大长度。 |
| `--output_dir` | 输出目录。 |
| `--edge_topn` 或 `--edge_threshold` | 选择 circuit 的 edge 策略。 |
| `--edge_score_abs` | 是否按绝对值选 edge。 |
| `--component_granularity` | `projection_matrix/head`，控制 downstream component projection 粒度。 |
| `--score_normalization` | 默认 `sum`，不得默认按 neuron 数量除。 |
| `--rank_score_source` | component-level rank 分配时用哪个 score view。 |

### 6.2 数据输入格式

单任务 EAP_forLogicalCircuit 第一版继续使用与 `EAP_forComponent` 相同的 pair dataset 约定：

```text
clean,corrupted,correct_idx,incorrect_idx
```

对于 `bool`、`gender`、`ioi_mistral` 等数据集，沿用现有 `EAP_forComponent/data.py` 的 clean/corrupted 对构造方式。

## 7. Edge-level attribution 需求

### 7.1 与 EAP_forComponent 的核心差异

`EAP_forComponent` 当前计算的是 node/component attribution：

```text
score(component) = (clean_activation - corrupted_activation) dot grad(metric, receiver_activation)
```

`EAP_forLogicalCircuit` 必须改为 edge attribution：

```text
score(edge source -> destination) = edge_signal_difference(source -> destination) dot grad(metric, destination_input)
```

其中关键差异是：

- receiver gradient 仍然来自 destination/input 侧，这一点与 `EAP_forComponent` future 算法保持一致。
- 参与乘积的不再是 destination 的“完整 activation difference”，而是 source 节点传递到 destination 输入位置的那一条 edge 上的 activation contribution。

### 7.2 Current edge attribution

第一版 current 算法必须与 EAP-IG 原始 edge-level做法保持一致：

1. 在 corrupted 输入上记录 edge source 对 destination input 的贡献。
2. 在 clean 输入上记录相同 edge contribution。
3. 构造：

```text
edge_delta = clean_edge_contribution - corrupted_edge_contribution
```

4. 对 destination input 的梯度做 backward hook：

```text
edge_score = sum(edge_delta * grad_destination_input)
```

### 7.3 Future edge attribution

future 算法要与 `EAP_forComponent` 一样，支持 current theta 与 future theta' 之间的方向修正，但 score 的对象从 node 改成 edge。

要求：

1. current 部分仍然计算 edge-level `current_score`。
2. future correction 仍然基于 destination/input 的梯度信息做 directional score。
3. 方向修正的 trapezoid 形式与 `EAP_forComponent/future_localization.py` 一致：

```text
raw_score(edge) = current_score(edge) + 0.5 * (direction_theta(edge) + direction_theta_hat(edge))
```

这里必须注意一个实现细节：按照当前 `EAP_forComponent/future_localization.py` 的做法，
`direction_theta` 和 `direction_theta_hat` 已经包含了 `future_step_k * Delta_theta` 的缩放，
因此最终公式外层不能再额外乘一次 `K`。如果 future edge 版本沿用同样的实现风格，
应保持同样的约定：`K` 只在方向项的计算过程中进入一次。

4. 当启用 `future_step_k_samples > 1` 时：
   - 在 `[k_min, k_max]` 中均匀采样 10 个四位小数 K 值。
   - 每个 K 独立得到一套 edge score。
   - 最终 edge attribution score 为 10 次 raw score 的均值。
   - 最终 edge attribution rank 为 10 次 `abs(raw_score)` 排名值的均值。
   - 不得用 10 次均值 score 再重新排序计算 rank。

### 7.4 K 采样兼容性

与 `EAP_forComponent` 一致：

- 正式 runner 支持随机 K 范围采样。
- 未来如新增 `sweep_future_k` 对应脚本，应显式固定 `future_step_k_samples=1`，只保留 exact-K 兼容逻辑。

## 8. Graph 与 edge registry 需求

### 8.1 Graph registry 职责

必须新增一层显式 registry，把当前 HF/patch 模型映射成 edge graph 元素。它至少要解决三件事：

1. source node 定义：input、attention head、MLP block、logit input 等。
2. destination input 定义：attention q/k/v input、MLP input、residual/logit input。
3. edge 到 component 的投影规则：某条 edge 归属于哪个 downstream component，或涉及哪些 component。

### 8.2 最低支持图结构

第一版建议支持与 EAP-IG 文档一致的最小图结构：

- source nodes：
  - `input`
  - attention heads
  - MLP outputs
- destination inputs：
  - attention head `q/k/v` inputs
  - MLP inputs
  - final logits / resid post input

### 8.3 输出 score 视图

为后续 conflict 和 finetuning 接入，必须同时保存：

1. edge-level score：原始 graph 结果。
2. component-level投影结果：供 mask / LoRA / conflict summary 使用。

## 9. circuit、circuit_or 与 logical_circuit

### 9.1 `circuit`

`circuit` 是标准方向的 edge 子图，语义参考 EAP-IG 中 `induction.py`：

- clean 是正确语义输入。
- corrupted 是扰动输入。
- score 代表“把 clean edge 替换成 corrupted edge 对目标 metric 的影响”。

### 9.2 `circuit_or`

`circuit_or` 语义参考 EAP-IG 中 `induction_or.py`：

- clean/corrupted 角色反转。
- correct/incorrect label 角色反转。
- 其意义是寻找反向支持路径，即错误方向或 OR 门补充路径。

### 9.3 选择策略

第一版不要求完全复刻 EAP-IG 所有 graph 操作，但必须支持下列一种或多种方式：

1. `topn`：按 edge score 绝对值选 top-n。
2. `threshold`：按阈值选 edge。

输出必须在 summary 中清楚记录：

- edge selection mode
- edge selection threshold / topn
- 是否 absolute

### 9.4 logical 融合

必须新增 `logical_fusion.py`，实现参考 `get_logical_edge.py` 的融合逻辑。第一版需求上定义如下：

1. 输入：
   - `circuit`
   - `circuit_or`
   - 两者各自的 edge attribution score
2. 输出：
   - `logical_circuit`
   - logical edge assignment / gate type / source metadata
3. 融合结果需要能回答每条 edge 或每个 component 是否：
   - 属于正向 path
   - 属于 OR path
   - 属于两者共同 path
   - 在 logical 规则下被赋值为 1 / 0 / unresolved

注意：第一版即使内部仍使用 edge graph 表示，最终也必须提供 component-level logical projection结果，供 conflict 分析与训练消费。

## 10. 单任务输出需求

对任意一个任务，`EAP_forLogicalCircuit` 最终必须输出三类结果：

1. edge-level原始结果
2. logical circuit 结果
3. component-level训练对接结果

建议输出目录结构：

```text
output_dir/
    circuit/
        edge_scores.json
        edge_scores.pt
        circuit_edges.json
        circuit_summary.json
    circuit_or/
        edge_scores.json
        edge_scores.pt
        circuit_or_edges.json
        circuit_or_summary.json
    logical_circuit/
        logical_edges.json
        logical_components.json
        logical_component_scores.json
        logical_component_scores.pt
        component_mask.pt
        rank_pattern.json
        lora_allocation.json
        summary.json
```

### 10.1 component-level分数定义

用户已明确要求：

对每个 task 的全部 component 信息，其 attribution score 定义为：

```text
component_score_task = (component_score_from_circuit + component_score_from_circuit_or) / 2
```

这里的 `component_score_from_circuit` 和 `component_score_from_circuit_or` 是把 edge-level score 聚合到 component-level 后再平均，而不是先对 edge 取 union 后直接求和。

### 10.2 component-level rank 定义

与 `EAP_forComponent` 保持一致：

- score 默认使用 raw/sum attribution 语义。
- rank 默认使用 `abs(raw_score)` 排名。
- 若 future K 采样启用，则最终 rank 为 per-K rank value 的均值，不得用平均 score 反推 rank。

### 10.3 component-level mask 与 LoRA 输出

每个 task 的 logical component 输出必须继续兼容当前训练接口：

- `component_mask.pt`：`{parameter_name: BoolTensor}`
- `rank_pattern.json`
- `component_scores.json`
- `summary.json`

也就是说，虽然新系统内部是 edge graph，但对 finetuning 的接口层仍然必须保持 component-level artifact 兼容性。

## 11. 多任务 conflict 分析需求

### 11.1 输入

conflict 分析模块不要求自动重新跑 attribution。输入是多个任务的 `logical_circuit` 结果，至少包含：

- 任务名
- logical component 列表
- 每个 component 的 logical assignment 状态
- 每个 component 的 attribution score / rank / lora / mask 信息

### 11.2 conflict 分析目标

目标不是简单求交集或并集，而是参考 EAP-IG `conflict.py`，在“尽量最小化冲突 component 数量”的前提下，识别三类 component：

1. 单任务全部 component：
   - 对该任务而言，赋值为 1 的 component
   - 加上无法赋值的 conflict component
   - 这两者合起来构成该任务的“全部 component”输出
2. 全任务冲突 component：
   - 所有任务 logical circuit 中无法一致赋值的 component
   - 且这些 unresolved component 进入全任务 conflict 集合
3. 所有任务涉及 component：
   - 所有任务 logical circuit 中赋值为 1 的 component
   - 加上所有 unresolved/conflict component
   - 赋值为 0 的 component 不进入结果

### 11.3 OR_Gate 语义

需求中明确提到：

- 有些 component 之所以只在部分 circuit 中出现，是因为 OR gate 不需要微调。

因此 conflict 分析必须在输出元数据中保留“该 component 为什么没有进入某任务最终训练集”的解释字段。最低要求：

| 字段 | 说明 |
| --- | --- |
| `logical_assignment` | `1` / `0` / `unresolved` |
| `assignment_reason` | `positive_path` / `or_gate_excluded` / `conflict_unresolved` / `absent` |
| `source_tasks` | 包含该 component 的任务列表 |

### 11.4 conflict 分析输出

conflict 分析模块必须输出三套 component-level artifacts：

1. 每个 task 的全部 component 信息
2. 所有任务冲突的 component 信息
3. 所有任务涉及的 component 信息

每套信息都必须包含：

- attribution score
- attribution rank
- component_scores.json/pt
- component_mask.pt
- rank_pattern.json
- lora_allocation.json
- summary.json

## 12. 多任务 component score 聚合规则

### 12.1 每个 task 全量 component

对某个 task：

```text
task_component_score = (circuit_score + circuit_or_score) / 2
```

### 12.2 conflict components 与 all-task components

对冲突集合和所有任务涉及集合，component attribution score 定义为：

```text
aggregate_component_score = average(component_score_task_i over all tasks containing this component)
```

这里的平均必须基于 task-level component score，而不是 edge-level score 直接平均。

### 12.3 rank 聚合规则

对 conflict/all-task 输出，rank 同样要求与 score 分离：

1. 每个 task 内先得到该 component 的 raw attribution rank。
2. 聚合输出时，对这些 task-level raw rank value 求平均。
3. 不得用均值 score 再反推 rank。

## 13. 多任务输出目录建议

建议为 conflict 分析增加单独的输出目录，例如：

```text
output_dir/
    tasks/
        bool/
            logical_circuit/
            components/
        gender/
            logical_circuit/
            components/
        ioi_mistral/
            logical_circuit/
            components/
    conflict/
        component_scores.json
        component_scores.pt
        component_mask.pt
        rank_pattern.json
        lora_allocation.json
        summary.json
    all_tasks/
        component_scores.json
        component_scores.pt
        component_mask.pt
        rank_pattern.json
        lora_allocation.json
        summary.json
    conflict_summary.json
```

`conflict_summary.json` 至少要包含：

- task 列表
- 每个 task component 数量
- conflict component 数量
- all-task component 数量
- 每个 component 的 source_tasks
- 每个 component 的 logical assignment 解释

## 14. 对 finetuning_model.py 的新增需求

### 14.1 兼容原则

必须保持现有单任务功能不变：

- `mask_path` 单 mask 微调
- `use_lora=0/1`
- `lora_mode=standard/projection_matrix/head`
- 现有 `TargetFT`、`TargetFT+PervasivenessFT`、`TargetFT+PervasivenessKL`、`TargetFT_L1`

在此基础上增加多任务 circuit-aware 训练能力。

### 14.2 新增 CLI / config 参数

现有 `mask_path` 需改为更明确的 target 语义，并新增多任务相关路径：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `target_mask_path` | str | target task 专属 mask。取代现有 `mask_path` 的主要语义。 |
| `pervasiveness_mask_paths` | str/list | pervasiveness task mask path 列表。元素数量必须与 `pervasiveness_dataset_name` 中任务数一致。 |
| `conflict_mask_path` | str | 所有任务冲突 component 的 mask。第一版输出但训练中暂不加载。 |
| `all_component_mask_path` | str | 所有任务涉及 component 的 mask，用于第二阶段训练。 |
| `target_lora_info_dir` | str | target task 的 logical-circuit-derived LoRA 信息目录。 |
| `pervasiveness_lora_info_dirs` | str/list | 每个 pervasiveness task 对应的 LoRA 信息目录。数量必须与 pervasiveness task 数匹配。 |
| `conflict_lora_info_dir` | str | conflict component 对应的 LoRA 信息目录。第一版可不加载。 |
| `all_component_lora_info_dir` | str | 所有任务涉及 component 的 LoRA 信息目录。第二阶段加载。 |
| `multi_task_schedule` | str | 默认 `two_stage_alternating`。 |
| `stage1_num_epochs` | int | 第一阶段 epoch 数，第一版固定 1。 |
| `stage2_num_epochs` | int | 第二阶段 epoch 数，第一版固定 1。 |

为了不破坏旧配置，建议兼容策略：

1. 若新参数为空，则沿用旧单任务 `mask_path` / `lora_info_dir` 逻辑。
2. 只有当 `finetuning_method == "TargetFT+PervasivenessFT"` 且显式提供多任务参数时，才进入多任务训练策略。

## 15. 多任务 mask 训练策略需求

### 15.1 触发条件

当满足以下条件时，启用第二种训练策略：

1. `finetuning_method == "TargetFT+PervasivenessFT"`
2. `pervasiveness_dataset_name` 含 1 个或多个任务名
3. 同时提供：
   - `target_mask_path`
   - `pervasiveness_mask_paths`
   - `all_component_mask_path`

### 15.2 第一阶段训练

第一阶段目标：每个任务先在自己的 task-local component 子集上单独更新。

要求：

1. 第一阶段为 1 个 epoch。
2. 每个 step 或 mini-phase 在 target task 与 pervasiveness tasks 之间交替训练。
3. 当前任务训练时：
   - 加载该任务自己的 mask
   - loss 只计算该任务对应的 loss
4. 具体要求：
   - target task 使用 `target_mask_path`
   - 第 `i` 个 pervasiveness task 使用 `pervasiveness_mask_paths[i]`

### 15.3 第二阶段训练

第二阶段目标：在所有任务共同涉及的 component 上做联合协调。

要求：

1. 第二阶段再训练 1 个 epoch。
2. 加载 `all_component_mask_path`。
3. loss 计算所有任务的损失函数：

```text
loss = target_loss + average(pervasiveness_losses)
```

或按现有权重参数：

```text
loss = target_weight * target_loss + pervasiveness_weight * average(pervasiveness_losses)
```

### 15.4 conflict mask

`conflict_mask_path` 第一版不参与训练，但必须保留参数、校验和 summary 记录，为后续 conflict-aware 特殊策略预留接口。

## 16. 多任务 LoRA 训练策略需求

### 16.1 standard 模式

`lora_mode == "standard"` 时，保持现有 fixed PEFT LoRA 行为不变，不按任务切换 LoRA 配置。

### 16.2 projection_matrix / head 模式

当 `lora_mode in {"projection_matrix", "head"}` 且启用多任务训练策略时：

#### 第一阶段

1. target task 加载 `target_lora_info_dir`。
2. 每个 pervasiveness task 加载自己的 `pervasiveness_lora_info_dirs[i]`。
3. 当前任务训练时只计算当前任务自己的 loss。

#### 第二阶段

1. 加载 `all_component_lora_info_dir`。
2. loss 改为所有任务联合损失。

### 16.3 训练时如何“加载不同 LoRA 信息”

第一版不应在单个 step 里反复销毁/重建整个模型。需求上建议采用以下二选一实现策略，并在设计文档中明确：

1. 预构建多个 trainer / LoRA adapter 视图，按任务切换 active adapter。
2. 维护单模型，但为每个任务预先装配独立的 LoRA module state，并在阶段一交替切换可训练 LoRA 参数引用。

无论哪种方案，必须满足：

- 不重复重新加载基础模型 checkpoint。
- 每个任务的 LoRA rank 配置确实来自对应 logical circuit 输出。
- 第二阶段切换到 all-task 共享 LoRA 配置时，状态切换必须可审计。

## 17. 数据结构与配置校验需求

### 17.1 多任务 mask path 校验

必须检查：

1. `pervasiveness_mask_paths` 的元素数量是否与 `pervasiveness_dataset_name` 的任务数一致。
2. 每个 path 是否可解析到一个 mask 文件。
3. 每个 mask 与当前模型参数 shape 是否一致。

### 17.2 多任务 LoRA info 校验

必须检查：

1. `pervasiveness_lora_info_dirs` 的元素数量是否与 pervasiveness task 数一致。
2. 每个目录中必须存在与模式匹配的文件：
   - projection_matrix: `rank_pattern.json`
   - head: `component_scores.json`
3. summary 中记录的 `attention_granularity`、`score_normalization`、`rank_score_source` 应与当前训练要求兼容。

### 17.3 component 身份统一

无论 edge graph、task outputs、conflict outputs、mask、LoRA 如何转换，都必须使用一致的 component 身份字符串。推荐沿用 `EAP_forComponent` 的：

- parameter name
- module name
- component type
- head idx / row slice / col slice

否则多任务 conflict 聚合时会因命名不一致无法对齐。

## 18. 训练循环重构需求

现有 `_run_finetuning_training()` 是单 trainer、单 mask、单 LoRA 视图循环。新增多任务策略后，需要拆成两个层级：

1. `SingleTaskTrainingContext`
2. `MultiTaskTrainingSchedule`

建议新增抽象：

```python
@dataclass
class TaskTrainingSpec:
    task_name: str
    role: str  # target | pervasiveness
    dataset_key: str
    mask_path: str | None
    lora_info_dir: str | None
    weight: float


@dataclass
class StageSpec:
    name: str
    num_epochs: int
    task_specs: list[TaskTrainingSpec]
    shared_mask_path: str | None = None
    shared_lora_info_dir: str | None = None
    joint_loss: bool = False
```

这样第一阶段与第二阶段都能用统一调度器表达，而不是在单循环里硬编码大量 `if/else`。

## 19. 输出契约需求

### 19.1 EAP_forLogicalCircuit 输出

必须至少输出：

1. task-level `circuit`
2. task-level `circuit_or`
3. task-level `logical_circuit`
4. task-level component artifacts
5. 可供 conflict 分析使用的 structured logical assignment 文件

### 19.2 conflict 分析输出

必须输出：

1. 每个 task 全量 component artifacts
2. 全任务 conflict component artifacts
3. 全任务 all-component artifacts
4. 总结文件，记录 assignment 与聚合来源

### 19.3 finetuning 输出

多任务训练的 summary 需要新增：

- stage1/stage2 配置
- 每个任务使用的 mask path
- 每个任务使用的 LoRA info dir
- second-stage all-component mask / LoRA info
- conflict mask path
- 每个 epoch 当前任务与当前激活的训练视图

## 20. 验证与测试需求

### 20.1 EAP_forLogicalCircuit 单任务测试

至少需要：

1. tiny model edge current attribution smoke test
2. tiny model edge future attribution smoke test
3. K 采样 future 聚合测试：均值 score 与均值 raw-rank 语义
4. `circuit` / `circuit_or` / `logical_circuit` 输出结构测试

### 20.2 conflict 分析测试

至少需要：

1. 两任务无冲突 case
2. 两任务完全冲突 case
3. OR gate 排除 case
4. component attribution 聚合均值测试

### 20.3 finetuning 多任务测试

至少需要：

1. `pervasiveness_mask_paths` 数量校验
2. `pervasiveness_lora_info_dirs` 数量校验
3. stage1 单任务交替训练 smoke test
4. stage2 all-component 联合训练 smoke test
5. old single-task config 回归测试

## 21. 实施顺序建议

建议按以下顺序实现，避免一次性耦合过深：

### Phase 1

1. 新建 `EAP_forLogicalCircuit` 框架与 CLI/schema。
2. 完成单任务 edge-level current attribution。
3. 完成单任务 `circuit` 与 `circuit_or` 输出。

### Phase 2

1. 完成 edge-level future attribution。
2. 加入 K 采样 future 聚合。
3. 完成 logical fusion。

### Phase 3

1. 完成 edge -> component projection。
2. 输出 task-level component mask / LoRA / summary。

### Phase 4

1. 完成 conflict 分析模块。
2. 输出 task/conflict/all-task 三套 component artifacts。

### Phase 5

1. 扩展 `finetuning_model.py` 配置。
2. 实现多任务两阶段 mask 训练。
3. 实现多任务两阶段 LoRA 训练。
4. 做单任务回归与多任务 smoke test。

## 22. 当前实现中必须保留的不变量

新增功能时，以下行为不能被破坏：

1. `EAP_forComponent` 现有单任务 component 输出与 future K 采样行为保持可用。
2. `finetuning_model.py` 单任务模式不变。
3. `score_normalization` 默认仍为 `sum`，不得回退到按 neuron 数量平均。
4. `sweep_future_k.py` 仍然是 exact-K sweep 兼容路径，不使用随机 K 范围采样。
5. mask 目录可以解析到 `component_mask.pt`；训练入口对 component mask 与 logical-circuit-derived mask 的加载契约必须一致。

## 23. 需要确认但不阻塞需求文档的开放问题

以下问题不阻塞需求分析，但在编码前需要最终拍板：

1. edge -> component projection 时，某个 component 的 score 是聚合其 incident edges、outgoing edges、incoming edges，还是只按 receiver-side edges 聚合？本文档建议：默认以该 component 在 logical circuit 中被判定为“参与训练”的所有相关 edge score 绝对值平均或求和，并在实现文档中固定一种规则。
2. `circuit` 与 `circuit_or` 的 logical fusion 是否要完全复制 EAP-IG 的布尔规则，还是允许先做工程化近似版本？本文档要求语义对齐，但实现细节可先以最小可用版本落地。
3. 多任务第一阶段“交替训练”的粒度是按 batch、按 fixed number of steps、还是按 dataset round-robin。建议第一版按 batch round-robin。
4. 多任务 LoRA 切换采用 adapter-style 还是 module-state-style。建议优先选不需要重复重建基础模型的方案。
5. `TargetFT+PervasivenessKL` 是否也需要未来扩展为多任务两阶段策略。本文档第一版只强制 `TargetFT+PervasivenessFT`，但设计时不要把扩展路径堵死。

## 24. 文档结论

本次新增功能本质上是一次“可解释性层从 component/node 升级到 edge logical circuit，多任务训练层从单 mask/单 LoRA 升级到 task-aware 两阶段训练”的系统扩展。

从工程边界上，推荐拆成三层：

1. `EAP_forLogicalCircuit`：负责 edge attribution、logical circuit、task component 投影。
2. `conflict_analysis`：负责多任务 logical circuit 的 component 级融合与冲突解释。
3. `finetuning` 扩展：负责按 task-local / all-task component artifacts 执行两阶段 mask/LoRA 训练。

只要严格遵守本文档中的输入输出契约，新增系统就可以与现有 `EAP_forComponent` 和 `finetuning_model.py` 并存，而不会破坏已有单任务训练与 component-level EAP 流程。