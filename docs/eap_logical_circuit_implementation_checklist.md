# EAP_forLogicalCircuit 实现任务清单

本文档把 [docs/eap_logical_circuit_requirements.md](/ssd_users/chenhang/CSAT/docs/eap_logical_circuit_requirements.md) 细化成可执行的实现任务清单。目标是把需求拆成一系列可独立开发、验证和回归的工程任务，便于逐阶段推进。

## 1. 实施原则

### 1.1 总原则

1. 新功能必须在不破坏现有 `EAP_forComponent` 与 `finetuning_model.py` 单任务流程的前提下实现。
2. 每个阶段都要形成可运行、可验证、可回归的最小闭环。
3. 所有新增输出尽量复用 `EAP_forComponent` 的 artifact 风格，以降低 finetuning 接入成本。
4. 所有新的 component-level 训练输入都必须最终落成与当前兼容的：

```python
{parameter_name: torch.BoolTensor(shape == parameter.shape)}
```

以及：

```json
{module_name: rank}
```

### 1.2 约束

1. `score_normalization` 默认必须保持 `sum`。
2. future K 采样只在正式 runner 启用；sweep 兼容脚本保持 exact-K。
3. future trapezoid 公式不得重复乘 `K`。
4. 多任务训练扩展必须与现有 `TargetFT` 系列保持后向兼容。

## 2. 开发阶段总览

建议分 9 个阶段推进：

1. `Phase 0`：基础准备与接口冻结
2. `Phase 1`：`EAP_forLogicalCircuit` 骨架与 current edge attribution
3. `Phase 2`：future edge attribution 与 K 采样
4. `Phase 3`：`circuit`、`circuit_or`、`logical_circuit` 输出
5. `Phase 4`：component 投影、mask/LoRA artifacts 输出
6. `Phase 5`：多任务 conflict 分析
7. `Phase 6`：`finetuning_model.py` 多任务配置与兼容层扩展
8. `Phase 7`：多任务 mask 两阶段训练策略实现
9. `Phase 8`：多任务 LoRA 两阶段训练策略实现

每个阶段都包含：

- 目标
- 涉及文件
- 具体任务
- 验收标准
- 依赖关系

## 3. Phase 0：基础准备与接口冻结

### 3.1 目标

明确当前稳定基线，避免后续实现过程中混入未确认的旧逻辑。

### 3.2 涉及文件

- [EAP_forComponent/cli.py](/ssd_users/chenhang/CSAT/EAP_forComponent/cli.py)
- [EAP_forComponent/runner.py](/ssd_users/chenhang/CSAT/EAP_forComponent/runner.py)
- [EAP_forComponent/future_localization.py](/ssd_users/chenhang/CSAT/EAP_forComponent/future_localization.py)
- [EAP_forComponent/outputs.py](/ssd_users/chenhang/CSAT/EAP_forComponent/outputs.py)
- [src/exec/finetuning_model.py](/ssd_users/chenhang/CSAT/src/exec/finetuning_model.py)
- [src/model/finetuning.py](/ssd_users/chenhang/CSAT/src/model/finetuning.py)

### 3.3 具体任务

1. 记录当前稳定约定：
   - `score_normalization=sum`
   - future K 采样语义
   - `mask_path` 目录可自动解析到 `component_mask.pt`
   - probing 配置已改为 `1% / 1 epoch / 1e-7`
2. 对当前 EAP_forComponent 做一轮小范围 smoke 文档化：
   - current component attribution
   - future component attribution
   - K sampling aggregation
3. 对当前 finetuning 做一轮小范围 smoke 文档化：
   - 单任务 mask 加载
   - projection/head LoRA 接入

### 3.4 验收标准

1. 当前行为不再含糊。
2. 新增系统可以明确声明哪些内容是“复用”，哪些是“新增”。

### 3.5 依赖关系

无。

## 4. Phase 1：EAP_forLogicalCircuit 骨架与 current edge attribution

### 4.1 目标

建立新目录与最小 runner，先打通单任务 edge-level current attribution 的端到端流程。

### 4.2 建议新增文件

- [EAP_forLogicalCircuit/__init__.py](EAP_forLogicalCircuit/__init__.py)
- [EAP_forLogicalCircuit/run_eap_for_logical_circuit.py](EAP_forLogicalCircuit/run_eap_for_logical_circuit.py)
- [EAP_forLogicalCircuit/cli.py](EAP_forLogicalCircuit/cli.py)
- [EAP_forLogicalCircuit/schemas.py](EAP_forLogicalCircuit/schemas.py)
- [EAP_forLogicalCircuit/model_loader.py](EAP_forLogicalCircuit/model_loader.py)
- [EAP_forLogicalCircuit/data.py](EAP_forLogicalCircuit/data.py)
- [EAP_forLogicalCircuit/graph_registry.py](EAP_forLogicalCircuit/graph_registry.py)
- [EAP_forLogicalCircuit/edge_hooks.py](EAP_forLogicalCircuit/edge_hooks.py)
- [EAP_forLogicalCircuit/current_localization.py](EAP_forLogicalCircuit/current_localization.py)
- [EAP_forLogicalCircuit/runner.py](EAP_forLogicalCircuit/runner.py)
- [EAP_forLogicalCircuit/tests/test_current_edge_tiny.py](EAP_forLogicalCircuit/tests/test_current_edge_tiny.py)

### 4.3 具体任务

1. 定义 CLI 与 config dataclass。
2. 复用或薄封装 `EAP_forComponent/model_loader.py` 的模型加载逻辑。
3. 复用或薄封装 `EAP_forComponent/data.py` 的 pair dataset 加载逻辑。
4. 定义 graph registry：
   - source nodes
   - destination inputs
   - edge identity
5. 建立 edge hooks：
   - clean/corrupted 两侧记录 source 对 destination 的 edge contribution
   - corrupted loss backward 读取 destination input gradient
6. 实现 current edge attribution：

```text
edge_score = sum((clean_edge - corrupted_edge) * grad_destination_input)
```

7. 输出最小 edge score 文件：
   - `edge_scores.json`
   - `edge_scores.pt`
   - `summary.json`

### 4.4 验收标准

1. `--localization_mode current` 单任务可跑通。
2. tiny test 中至少能看到非零 edge score。
3. summary 中记录模型、数据集、metric、batch size、edge 数量。

### 4.5 依赖关系

依赖 `Phase 0` 的基线确认。

## 5. Phase 2：future edge attribution 与 K 采样

### 5.1 目标

把 `EAP_forComponent` 的 future 算法迁移到 edge-level，并保留 K 范围采样能力。

### 5.2 建议新增/修改文件

- [EAP_forLogicalCircuit/future_localization.py](EAP_forLogicalCircuit/future_localization.py)
- [EAP_forLogicalCircuit/runner.py](EAP_forLogicalCircuit/runner.py)
- [EAP_forLogicalCircuit/schemas.py](EAP_forLogicalCircuit/schemas.py)
- [EAP_forLogicalCircuit/tests/test_future_edge_tiny.py](EAP_forLogicalCircuit/tests/test_future_edge_tiny.py)
- [EAP_forLogicalCircuit/tests/test_future_k_sampling.py](EAP_forLogicalCircuit/tests/test_future_k_sampling.py)

### 5.3 具体任务

1. 迁移 future model loading / delta tensor 逻辑。
2. 明确 edge-level `current_score`、`direction_theta`、`direction_theta_hat` 的计算位置。
3. 复用 finite-difference 与 HVP 两条路径。
4. 保持 `future_step_k` 的当前语义：
   - `K` 在方向项内部生效
   - 最终公式外层不再额外乘 `K`
5. 复用 K 采样策略：
   - 随机四位小数
   - `future_step_k_samples > 1` 时运行多次
   - score 取均值
   - rank 取 `abs(raw_score)` 的 rank value 均值

### 5.4 验收标准

1. future 单 K 模式可跑通。
2. future K 范围采样模式可跑通。
3. 测试明确验证“不要对均值 score 重新排 rank”。
4. 文档中已固定“`K` 不重复乘”的语义。

### 5.5 依赖关系

依赖 `Phase 1`。

## 6. Phase 3：circuit、circuit_or、logical_circuit 输出

### 6.1 目标

在 edge attribution 之上，实现两套 circuit 以及逻辑融合结果。

### 6.2 建议新增文件

- [EAP_forLogicalCircuit/circuit_builder.py](EAP_forLogicalCircuit/circuit_builder.py)
- [EAP_forLogicalCircuit/logical_fusion.py](EAP_forLogicalCircuit/logical_fusion.py)
- [EAP_forLogicalCircuit/tests/test_logical_fusion.py](EAP_forLogicalCircuit/tests/test_logical_fusion.py)

### 6.3 具体任务

1. 实现 `circuit` 选择器：
   - top-n
   - threshold
2. 实现 `circuit_or`：
   - clean/corrupted 反转
   - correct/incorrect 标签反转
3. 保存：
   - `circuit_edges.json`
   - `circuit_or_edges.json`
4. 实现 logical fusion：
   - 给出每条 edge 的 logical assignment
   - 标记正向 path / OR path / shared path / unresolved
5. 输出 `logical_edges.json`

### 6.4 验收标准

1. `circuit`、`circuit_or`、`logical_circuit` 三者结构完整。
2. summary 中记录 edge selection mode、topn/threshold 与 absolute 选项。
3. tiny synthetic case 可验证 logical fusion 基本语义。

### 6.5 依赖关系

依赖 `Phase 2`。

## 7. Phase 4：component 投影、mask/LoRA artifacts 输出

### 7.1 目标

把 logical circuit 投影到 component-level，并输出当前 finetuning 能消费的 artifacts。

### 7.2 建议新增文件

- [EAP_forLogicalCircuit/component_projection.py](EAP_forLogicalCircuit/component_projection.py)
- [EAP_forLogicalCircuit/mask_builder.py](EAP_forLogicalCircuit/mask_builder.py)
- [EAP_forLogicalCircuit/rank_allocator.py](EAP_forLogicalCircuit/rank_allocator.py)
- [EAP_forLogicalCircuit/outputs.py](EAP_forLogicalCircuit/outputs.py)
- [EAP_forLogicalCircuit/tests/test_component_projection.py](EAP_forLogicalCircuit/tests/test_component_projection.py)
- [EAP_forLogicalCircuit/tests/test_rank_and_mask.py](EAP_forLogicalCircuit/tests/test_rank_and_mask.py)

### 7.3 具体任务

1. 定义 edge -> component 的聚合规则。
2. 为每个 task 分别计算：

```text
component_score_task = (score_from_circuit + score_from_circuit_or) / 2
```

3. 默认 component rank 使用 `abs(raw_score)`。
4. 生成：
   - `logical_component_scores.json`
   - `logical_component_scores.pt`
   - `component_mask.pt`
   - `rank_pattern.json`
   - `lora_allocation.json`
   - `summary.json`
5. mask 与 rank allocator 尽量复用 `EAP_forComponent` 的约定，避免训练侧再写一套解析器。

### 7.4 验收标准

1. component-level输出能被现有 mask loader 读取。
2. projection_matrix 模式下 `rank_pattern.json` 能被当前 LoRA 逻辑接受。
3. head 模式下 `component_scores.json` 能被当前 head-wise LoRA 逻辑读取。

### 7.5 依赖关系

依赖 `Phase 3`。

## 8. Phase 5：多任务 conflict 分析

### 8.1 目标

对多个任务的 logical circuit/component outputs 做冲突分析，并输出三类训练可消费的 component 集合。

### 8.2 建议新增文件

- [EAP_forLogicalCircuit/conflict_analysis.py](EAP_forLogicalCircuit/conflict_analysis.py)
- [EAP_forLogicalCircuit/run_conflict_analysis.py](EAP_forLogicalCircuit/run_conflict_analysis.py)
- [EAP_forLogicalCircuit/tests/test_conflict_analysis.py](EAP_forLogicalCircuit/tests/test_conflict_analysis.py)

### 8.3 具体任务

1. 定义 conflict 输入 schema：
   - task name
   - logical assignment
   - component score
   - rank
   - source metadata
2. 实现三类输出：
   - task-level all components
   - global conflict components
   - global all-task components
3. 实现 assignment reason 字段：
   - `positive_path`
   - `or_gate_excluded`
   - `conflict_unresolved`
   - `absent`
4. 实现 score 聚合：

```text
aggregate_component_score = average(task_component_score_i)
```

5. 实现 rank 聚合：
   - 平均 task-level raw rank value
   - 不得由均值 score 反推 rank
6. 输出三套 artifacts：
   - `component_scores.json/pt`
   - `component_mask.pt`
   - `rank_pattern.json`
   - `lora_allocation.json`
   - `summary.json`
7. 输出总览：`conflict_summary.json`

### 8.4 验收标准

1. 两任务无冲突 case 正确。
2. 两任务完全冲突 case 正确。
3. OR gate 排除 case 能区分 `or_gate_excluded`。
4. conflict/all-task 输出都可直接接入 finetuning 的 mask/LoRA 逻辑。

### 8.5 依赖关系

依赖 `Phase 4`。

## 9. Phase 6：finetuning_model.py 多任务配置扩展

### 9.1 目标

在不破坏单任务行为的前提下，为 finetuning 增加多任务 mask/LoRA 输入配置。

### 9.2 涉及文件

- [src/exec/finetuning_model.py](/ssd_users/chenhang/CSAT/src/exec/finetuning_model.py)
- [src/model/finetuning.py](/ssd_users/chenhang/CSAT/src/model/finetuning.py)

### 9.3 具体任务

1. 新增参数：
   - `target_mask_path`
   - `pervasiveness_mask_paths`
   - `conflict_mask_path`
   - `all_component_mask_path`
   - `target_lora_info_dir`
   - `pervasiveness_lora_info_dirs`
   - `conflict_lora_info_dir`
   - `all_component_lora_info_dir`
   - `multi_task_schedule`
   - `stage1_num_epochs`
   - `stage2_num_epochs`
2. 兼容旧参数：
   - 如果新参数为空，继续使用旧 `mask_path` / `lora_info_dir`
3. 解析 list 型输入并做数量校验。
4. 在 summary/config dump 中完整记录多任务参数。

### 9.4 验收标准

1. 老配置仍可直接运行。
2. 新配置可以通过参数校验。
3. list 数量不匹配时能给出清晰报错。

### 9.5 依赖关系

依赖 `Phase 5` 的输出契约固定。

## 10. Phase 7：多任务 mask 训练策略实现

### 10.1 目标

实现两阶段多任务 mask 训练逻辑。

### 10.2 涉及文件

- [src/model/finetuning.py](/ssd_users/chenhang/CSAT/src/model/finetuning.py)
- [src/finetuning/base.py](/ssd_users/chenhang/CSAT/src/finetuning/base.py)
- [src/finetuning/FT.py](/ssd_users/chenhang/CSAT/src/finetuning/FT.py)
- [src/dataset/Base.py](/ssd_users/chenhang/CSAT/src/dataset/Base.py)
- [src/dataset/__init__.py](/ssd_users/chenhang/CSAT/src/dataset/__init__.py)

### 10.3 具体任务

1. 设计 `TaskTrainingSpec` / `StageSpec` 数据结构。
2. 把现有单训练循环抽象成可切换 task context 的调度器。
3. 第一阶段：
   - target 与各 pervasiveness task 交替训练
   - 当前任务只加载自己的 mask
   - loss 只计算当前任务
4. 第二阶段：
   - 加载 `all_component_mask_path`
   - loss 计算 target + 全部 pervasiveness
5. `conflict_mask_path` 第一版只校验和记录，不参与训练。

### 10.4 验收标准

1. 单任务训练行为不变。
2. 多任务第一阶段能切换 mask。
3. 第二阶段能切换到 all-component mask。
4. 训练日志中能看到当前 stage 和当前 task。

### 10.5 依赖关系

依赖 `Phase 6`。

## 11. Phase 8：多任务 LoRA 训练策略实现

### 11.1 目标

实现 projection/head 模式下的多任务 LoRA 切换与两阶段训练。

### 11.2 涉及文件

- [src/model/finetuning.py](/ssd_users/chenhang/CSAT/src/model/finetuning.py)
- [src/model/lora_utils.py](/ssd_users/chenhang/CSAT/src/model/lora_utils.py)
- 新增：
  - [src/model/multitask_lora.py](src/model/multitask_lora.py)
  - [src/model/tests/test_multitask_lora.py](src/model/tests/test_multitask_lora.py)

### 11.3 具体任务

1. `standard` 模式保持不变。
2. `projection_matrix` / `head` 模式支持：
   - 第一阶段按任务切换 task-local LoRA info
   - 第二阶段切换到 all-task LoRA info
3. 选择实现方案：
   - active adapter 切换，或
   - 单模型多 LoRA state 切换
4. 确保不重复重载 base model。
5. 记录每个阶段、每个任务使用的 LoRA 信息来源。

### 11.4 验收标准

1. projection/head 模式都可运行多任务训练。
2. 每个任务确实命中自己的 rank/component 配置。
3. 第二阶段切换到 all-component LoRA 后日志可追踪。

### 11.5 依赖关系

依赖 `Phase 7`。

## 12. Phase 9：测试、回归与文档收尾

### 12.1 目标

补齐新增系统的测试闭环，并把文档与实现状态对齐。

### 12.2 任务清单

1. 为 `EAP_forLogicalCircuit` 建立最小 tiny test 套件。
2. 为 conflict 分析建立 synthetic test 套件。
3. 为 finetuning 多任务模式建立 smoke test。
4. 重新跑单任务回归：
   - `EAP_forComponent`
   - `sweep_future_k.py`
   - `finetuning_model.py` 单任务 mask
   - `finetuning_model.py` projection/head LoRA
5. 更新：
   - [docs/eap_logical_circuit_requirements.md](/ssd_users/chenhang/CSAT/docs/eap_logical_circuit_requirements.md)
   - [docs/eap_logical_circuit_implementation_checklist.md](/ssd_users/chenhang/CSAT/docs/eap_logical_circuit_implementation_checklist.md)
   - 新模块 README

### 12.3 验收标准

1. 新功能测试通过。
2. 单任务旧功能回归通过。
3. 文档与实现一致。

## 13. 并行开发建议

如果多人或多轮次开发，推荐这样分工：

1. A 线：`EAP_forLogicalCircuit` runner + attribution
2. B 线：logical fusion + conflict analysis
3. C 线：finetuning 多任务配置 + 两阶段训练
4. D 线：LoRA 多任务切换与测试

其中并行边界：

- `Phase 1` 与 `Phase 6` 不建议并行，接口未冻结前会反复返工。
- `Phase 4` 与 `Phase 6` 可以在 component artifact 契约冻结后并行。

## 14. 优先级排序

如果要以最短路径形成第一版可演示结果，建议优先级是：

1. `Phase 1`
2. `Phase 2`
3. `Phase 3`
4. `Phase 4`
5. `Phase 5`
6. `Phase 6`
7. `Phase 7`
8. `Phase 8`
9. `Phase 9`

原因：只有先拿到 task-level logical component artifacts，后续多任务训练扩展才有真实输入可接。

## 15. 最小里程碑定义

### Milestone A

单任务 current edge attribution + `circuit`/`circuit_or` 输出。

### Milestone B

单任务 future edge attribution + K 采样 + `logical_circuit` 输出。

### Milestone C

单任务 logical component artifacts 可直接接入当前 finetuning mask/LoRA。

### Milestone D

多任务 conflict 输出三套 component artifacts。

### Milestone E

`TargetFT+PervasivenessFT` 的两阶段多任务训练打通。

## 16. 每阶段完成后的必做验证

每个阶段完成后都必须至少执行：

1. 语法编译或 import smoke
2. 目标 slice 的 tiny/synthetic test
3. 与旧功能边界相邻的最小回归
4. 输出文件结构检查

不能跳过验证直接进入下一阶段。