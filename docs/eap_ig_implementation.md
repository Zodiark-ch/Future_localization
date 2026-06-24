# EAP-IG 实现说明

本文档总结 `/ssd_users/chenhang/CSAT/EAP-IG` 中 EAP / EAP-IG 代码的实现过程、核心依赖、主要类与函数、输入输出以及最终产物。目的是为后续在当前 finetuning gradient mask 上做 neuron 二次筛选提供实现参考。

## 1. 项目目标

EAP-IG 的目标是为自回归 Transformer LM 自动寻找与某个任务相关的 circuit。代码把模型拆成计算图中的组件节点和边，然后用 clean / corrupted 输入对、任务 metric、activation difference 和 gradient 来估计组件或边的重要性。

项目支持三种粒度：

- `edge`：给 source component 到 destination component input 的边打分。
- `node`：给每个 component 节点打分。
- `neuron`：给每个 component 输出向量的 hidden 维度打分，即每个 source node 有一个长度为 `d_model` 的 neuron score。注意这里的 neuron 是 activation/output dimension，不是参数矩阵中的单个权重值。

README 中描述的典型流程是：定义任务数据和 metric，构建 `Graph`，调用 `attribute(...)` 或 `attribute_node(...)` 计算 attribution score，再用 `Graph.apply_topn(...)` 或 `Graph.apply_threshold(...)` 选择 circuit，最后用 `evaluate_graph(...)` 验证 circuit 的性能。

## 2. 目录结构

`EAP-IG` 目录中的主要文件如下：

- `README.md`：算法能力、使用流程和兼容性说明。
- `EAPenviroment.yml`：运行环境依赖。
- `gender.py`、`induction.py`：正向 EAP-IG 任务脚本。
- `gender_or.py`、`induction_or.py`：反向 EAP-IG 任务脚本，会交换 clean/corrupted 和 correct/incorrect 标签。
- `tocircuitjson.py`：把 `Graph.to_pt(...)` 生成的 `.pt` 文件转换为 `graph.json` 和 `score.json`。
- `src/eap/graph.py`：`Node`、`Edge`、`Graph` 的定义，以及 circuit 选择、剪枝、保存和加载。
- `src/eap/attribute.py`：edge-level EAP / EAP-IG / exact / clean-corrupted / information-flow-routes 打分。
- `src/eap/attribute_node.py`：node-level 和 neuron-level EAP / EAP-IG 打分。
- `src/eap/evaluate.py`：对已选择的 circuit 做 patching / zero / mean ablation 评估。
- `src/eap/utils.py`：tokenize、hook 构造、activation difference 矩阵和 mean activation 计算。
- `src/eap/visualization.py`：graph 可视化颜色辅助，配合 `Graph.to_image(...)` 使用。

## 3. 主要依赖

核心依赖来自 `EAPenviroment.yml` 和源码 import：

- `torch`：模型运行、tensor 计算、autograd、`torch.save/load`。
- `transformers`：`AutoModelForCausalLM`、`AutoTokenizer`，用于加载本地 HF checkpoint。
- `transformer-lens`：`HookedTransformer`、hook points、`get_attention_mask`，是实现 EAP 的关键工具。
- `einops`：用 `einsum` 聚合 activation difference 和 gradient。
- `pandas`：读取任务 CSV。
- `torch.utils.data`：`Dataset`、`DataLoader`。
- `tqdm`：进度条。
- `numpy`、`matplotlib`、`pygraphviz`：图可视化相关。
- Python 标准库：`argparse`、`functools.partial`、`os`、`json`、`heapq`、`pathlib`、`re`、`typing`。

环境文件中记录的关键版本包括 `torch==2.10.0`、`transformer-lens==2.17.0`、`transformers==4.57.6`、`einops==0.8.2`、`pandas==3.0.1`、`numpy==1.26.4`、`tqdm==4.67.3`、`pygraphviz==1.14`。

## 4. 任务脚本流程

`gender.py`、`induction.py`、`gender_or.py`、`induction_or.py` 的结构基本相同。

### 4.1 CLI 输入

脚本通过 `argparse` 接收：

- `--model_dir`：本地 HF checkpoint 路径，默认指向已有 finetuned checkpoint。
- `--save_name`：输出 graph `.pt` 文件名。

脚本内部还硬编码了：

- `CUDA_VISIBLE_DEVICES`，例如 `"3"` 或 `"0"`。
- `model_name = "mistralai/Mistral-7B-v0.1"`。
- 任务 CSV 文件名：`gender.csv` 或 `induction.csv`。
- batch size：gender 为 4，induction 为 2。
- attribution 方法：`method="EAP-IG-inputs"`，`ig_steps=5`。
- circuit 大小：`g.apply_topn(20000, True)`。

### 4.2 数据输入格式

`EAPDataset` 读取 CSV，期望每一行包含：

- `clean`：任务正确语义的 prompt / text。
- `corrupted`：对应的 corrupted prompt / text。
- `correct_idx`：正确答案 token id。
- `incorrect_idx`：错误答案 token id。

`__getitem__` 返回：

```python
(row["clean"], row["corrupted"], [row["correct_idx"], row["incorrect_idx"]])
```

`collate_EAP` 把一个 batch 整理为：

```python
clean: list[str]
corrupted: list[str]
labels: torch.Tensor  # shape [batch, 2]
```

`gender_or.py` 和 `induction_or.py` 使用 `collate_EAP_OR`，它会交换 clean/corrupted，并把 label 的两列互换：

```python
clean_or = corrupted
corrupted_or = clean
labels = labels[:, [1, 0]]
```

这表示反向寻路时，把原来的错误方向当成 clean 方向。

### 4.3 模型加载

任务脚本先用 HF 加载 checkpoint：

```python
hf_model = AutoModelForCausalLM.from_pretrained(local_model_dir, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
```

再把 HF 模型包装成 TransformerLens 的 `HookedTransformer`：

```python
model = HookedTransformer.from_pretrained(
    model_name,
    hf_model=hf_model,
    tokenizer=tokenizer,
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device="cuda",
    dtype=torch.float16,
)
```

为了让 EAP hook 能工作，脚本显式设置：

```python
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True
```

这些配置让 TransformerLens 暴露 attention result、q/k/v input、MLP input 等 hook points。对 GQA 模型，`ungroup_grouped_query_attention=True` 会把 grouped query attention 展开，牺牲效率但让 head-level 图成立。

### 4.4 Metric 定义

任务脚本使用 `logit_diff` 作为 metric。它先取每个样本最后一个有效 token 位置的 logits：

```python
logits = logits[batch_indices, input_length - 1]
```

然后 gather `[correct_idx, incorrect_idx]` 两个 token 的 logit，计算：

```python
correct_logit - incorrect_logit
```

当 `loss=True` 时返回负数，即 `incorrect_logit - correct_logit`，用于 attribution 时让 backward 的目标是损失方向；当 `loss=False` 时返回原始 logit diff，用于 baseline 和 circuit 评估。

### 4.5 主流程

任务脚本的主流程是：

```python
ds = EAPDataset("gender.csv")
dataloader = ds.to_dataloader(4)

g = Graph.from_model(model)
attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method="EAP-IG-inputs", ig_steps=5)

g.apply_topn(20000, True)
g.to_pt(args.save_name)

baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
results = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results}")
```

最终得到一个 `.pt` graph 文件，并在 stdout 打印原模型 logit diff 与 circuit patch 后的 logit diff。

## 5. Graph 抽象

`src/eap/graph.py` 是 EAP-IG 的数据结构核心。

### 5.1 Node 类型

代码定义了以下节点：

- `InputNode`：名称 `input`，输出 hook 是 `hook_embed`。
- `AttentionNode`：名称形如 `a{layer}.h{head}`，输出 hook 是 `blocks.{layer}.attn.hook_result`，q/k/v input hooks 是 `blocks.{layer}.hook_q_input`、`hook_k_input`、`hook_v_input`。
- `MLPNode`：名称形如 `m{layer}`，输入 hook 是 `blocks.{layer}.hook_mlp_in`，输出 hook 是 `blocks.{layer}.hook_mlp_out`。
- `LogitNode`：名称 `logits`，输入 hook 是最后一层 `blocks.{n_layers - 1}.hook_resid_post`，始终被视为在 graph 中。

每个 `Node` 保存：`name`、`layer`、`in_hook`、`out_hook`、`index`、父子节点、父子边、qkv input hooks 等。`Node.in_graph`、`Node.score`、`Node.neurons`、`Node.neurons_scores` 本身不存值，而是映射到 `Graph` 的 tensor。

### 5.2 Edge 类型

`Edge` 表示从 parent node 到 child node input 的连接。边名形如：

- `input->m0`
- `a0.h3->m1`
- `m5->a6.h2<q>`

如果 child 是 attention head，必须指定 `qkv`，因为同一个 attention head 有 q/k/v 三个输入位置。`Edge.score` 映射到 `graph.scores[src_index, dst_index]`，`Edge.in_graph` 映射到 `graph.in_graph[src_index, dst_index]`。

### 5.3 Graph tensor

`Graph.from_model(model)` 根据 TransformerLens config 构造图。核心维度是：

```python
n_forward = 1 + n_layers * (n_heads + 1)
n_backward = n_layers * (3 * n_heads + 1) + 1
```

其中 forward 节点包含 input、每层每个 attention head、每层 MLP；backward 目标包含每层 attention head 的 q/k/v 输入、每层 MLP 输入和 logits 输入。

主要 tensor：

- `scores`: shape `[n_forward, n_backward]`，edge attribution scores。
- `in_graph`: shape `[n_forward, n_backward]`，edge 是否在当前 circuit 中。
- `real_edge_mask`: shape `[n_forward, n_backward]`，哪些 edge 在模型拓扑上真实存在。
- `nodes_in_graph`: shape `[n_forward]`，source node 是否在 circuit 中。
- `nodes_scores`: 可选 shape `[n_forward]`，node attribution scores。
- `neurons_in_graph`: 可选 shape `[n_forward, d_model]`，node 的 hidden dimension 是否在 circuit 中。
- `neurons_scores`: 可选 shape `[n_forward, d_model]`，node 的 hidden dimension attribution scores。
- `forward_to_backward`: shape `[n_forward, n_backward]`，source node 与自身 backward input 位置之间的映射。

如果需要 node score，要调用：

```python
Graph.from_model(model, node_scores=True)
```

如果需要 neuron score，要调用：

```python
Graph.from_model(model, neuron_level=True)
```

实际使用 node/neuron attribution 时通常需要：

```python
Graph.from_model(model, neuron_level=True, node_scores=True)
```

### 5.4 边的构造

`Graph.from_model` 按 residual stream 构造边：

- input 是初始 residual source。
- 每层 attention head 会接收所有之前 residual source 到 q/k/v input 的边。
- 每层 MLP 会接收所有之前 residual source 到 MLP input 的边。
- 如果 `parallel_attn_mlp=True`，attention 和 MLP 并行接收同一批之前 residual source。
- 如果不是 parallel，先加入 attention outputs，再让 MLP 接收包含 attention outputs 在内的 residual source。
- logits node 接收所有 residual source 的边。

这套边构造反映了 pre-LN Transformer 中 residual stream 是各组件输出之和的假设。

### 5.5 Circuit 选择与剪枝

`Graph` 支持三种选择方式：

- `apply_threshold(threshold, absolute=True, level="edge" | "node" | "neuron")`：按阈值选择。
- `apply_topn(n, absolute=True, level="edge" | "node" | "neuron")`：按 score 绝对值选 top-n。
- `apply_greedy(n_edges, absolute=True)`：从 logits 反向贪心选择 edge。

`prune()` 会确保 circuit 连通，删除没有入边或出边的节点/边；如果存在 `neurons_in_graph`，还会删除没有任何保留 neuron 的节点。

### 5.6 保存和加载

`Graph.to_pt(filename)` 保存 PyTorch dict，关键字段包括：

- `cfg`
- `src_nodes`
- `dst_nodes`
- `edges_scores`
- `edges_in_graph`
- `nodes_in_graph`
- 可选 `nodes_scores`
- 可选 `neurons_in_graph`
- 可选 `neurons_scores`

`Graph.from_pt(pt_path)` 可以恢复 graph。`Graph.to_json(...)` 保存可读 JSON；`tocircuitjson.py` 则把 `.pt` 文件转成两个 JSON：

- `graph.json`：只包含 `edges_in_graph=True` 的边，格式是 `[[src, dst], ...]`。
- `score.json`：包含 `src_nodes`、`dst_nodes` 和完整 `edges_scores` 二维浮点列表。

导出 JSON 时会重命名：`input -> embeds`，`logits -> resid_post`，`a0.h31<q> -> a0.h31.q`。

## 6. Edge-level EAP / EAP-IG 实现

edge-level 算法在 `src/eap/attribute.py` 和 `src/eap/utils.py`。

### 6.1 tokenize_plus

`tokenize_plus(model, inputs, max_length=None)` 调用 TransformerLens tokenizer：

```python
tokens = model.to_tokens(inputs, prepend_bos=True, padding_side="right", truncate=...)
attention_mask = get_attention_mask(model.tokenizer, tokens, True)
input_lengths = attention_mask.sum(1)
n_pos = attention_mask.size(1)
```

返回 `(tokens, attention_mask, input_lengths, n_pos)`。

### 6.2 make_hooks_and_matrices

`utils.make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)` 构建三类 hook 和一个 activation difference 矩阵：

```python
activation_difference: [batch, pos, n_forward, d_model]
fwd_hooks_corrupted
fwd_hooks_clean
bwd_hooks
```

forward hook 的逻辑是：

- 在 corrupted 输入上运行 `fwd_hooks_corrupted`，把 source node 输出 activation 加到 `activation_difference`。
- 在 clean 输入上运行 `fwd_hooks_clean`，把 source node 输出 activation 从 `activation_difference` 中减去。
- 因此矩阵里保存的是 `corrupted_activation - clean_activation`。

backward hook 的逻辑是：

```python
s = einsum(
    activation_difference[:, :, :prev_index],
    grads,
    "batch pos forward hidden, batch pos backward hidden -> forward backward",
)
scores[:prev_index, bwd_index] += s
```

也就是说，对每个 destination input 的梯度 `grads`，用所有之前 source node 的 activation difference 与该梯度逐位置、逐 hidden 维相乘并求和，得到 source-to-destination edge 的 attribution score。

hook 位置包括：

- source output hooks：`hook_embed`、每层 `attn.hook_result`、每层 `hook_mlp_out`。
- destination input backward hooks：每层 `hook_q_input`、`hook_k_input`、`hook_v_input`、每层 `hook_mlp_in`、最后 `hook_resid_post`。

### 6.3 EAP

`get_scores_eap(...)` 的流程是：

1. 初始化 `scores = zeros([n_forward, n_backward], device="cuda")`。
2. 对每个 batch 取 clean/corrupted tokens。
3. 用 corrupted forward hooks 记录 corrupted activation。
4. 用 clean forward hooks 记录 clean activation，形成 activation difference。
5. 计算 clean logits。
6. 再对 clean tokens 做一次带 forward/backward hooks 的 forward。
7. 计算 metric，例如负 logit diff。
8. 调用 `metric_value.backward()`，backward hooks 读取梯度并更新 `scores`。
9. 最后 `scores /= total_items`。

公式上接近：

```text
score(edge src -> dst) = sum_{batch,pos,hidden} (act_corrupted[src] - act_clean[src]) * grad_clean[dst_input]
```

这是对“如果把该 edge 从 clean 换成 corrupted，会使 metric 改变多少”的一阶近似。

### 6.4 EAP-IG inputs

`get_scores_eap_ig(...)` 是任务脚本实际调用的方法。它仍然使用 activation difference 和 backward hooks，但 gradient 不是只在 clean input 上取一次，而是在 corrupted input embedding 到 clean input embedding 的积分路径上取多步平均。

流程是：

1. 用 corrupted forward hooks 得到 corrupted source activations。
2. 提取 corrupted input embedding activation。
3. 用 clean forward hooks 得到 clean logits，并形成 activation difference。
4. 计算 clean input embedding activation。
5. 对 `step in range(0, steps)` 构造 input interpolation hook：

```python
new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted)
```

6. 在每个 interpolation step 上运行 clean tokens，但 input embedding 被 hook 替换为插值 activation。
7. 对 metric backward，bwd hooks 累加 score。
8. 最后除以样本数和 `steps`。

`attribute_node.py` 里的 EAP-IG inputs 版本使用 `range(1, steps + 1)`，并通过 `+ activations * 0` 保持 autograd 连接；edge-level `attribute.py` 版本使用 `range(0, steps)`。

### 6.5 EAP-IG activations

`get_scores_ig_activations(...)` 不只插值 input embedding，而是逐个 component 对 clean/corrupted activation 做插值：

```python
new_output = alpha * clean_activation + (1 - alpha) * corrupted_activation
```

然后在每个 node、每个 step 上 backward 累加 score。这比 inputs 版本更慢，复杂度约为 `O(steps * layers)`，但支持更一般的 activation-level intervention。

### 6.6 clean-corrupted 和 exact

- `get_scores_clean_corrupted(...)` 在 clean 和 corrupted 两端各做一次 backward，近似积分路径的两端平均。
- `get_scores_exact(...)` 不是一阶近似，而是逐条边关闭后调用 `evaluate_graph(...)`，用性能差作为 score，计算非常慢。

### 6.7 attribute 调度器

`attribute(...)` 是 edge-level 统一入口。它先检查模型配置：

```python
assert model.cfg.use_attn_result
assert model.cfg.use_split_qkv_input
assert model.cfg.use_hook_mlp_in
if model.cfg.n_key_value_heads is not None:
    assert model.cfg.ungroup_grouped_query_attention
```

然后根据 `method` 调用对应 scoring 函数。支持的方法包括：

- `EAP`
- `EAP-IG-inputs`
- `clean-corrupted`
- `EAP-IG-activations`
- `information-flow-routes`
- `exact`

如果 `aggregation="mean"`，会再除以 `d_model`。最后写入：

```python
graph.scores[:] = scores.to(graph.scores.device)
```

## 7. Node / Neuron-level EAP-IG 实现

`src/eap/attribute_node.py` 是后续 neuron 二次筛选最直接相关的参考。

### 7.1 与 edge-level 的区别

edge-level 的 `scores` shape 是：

```python
[n_forward, n_backward]
```

node-level 的 `scores` shape 是：

```python
[n_forward]
```

neuron-level 的 `scores` shape 是：

```python
[n_forward, d_model]
```

这里的 `neuron=True` 会让 gradient hook 使用：

```python
s = einsum(
    activation_difference[:, :, fwd_index],
    grads,
    "batch pos ... hidden, batch pos ... hidden -> ... hidden",
)
scores[fwd_index] += s
```

因此每个 source component 的每个 hidden/output dimension 都会得到一个 attribution score。

### 7.2 make_hooks_and_matrices in attribute_node

`attribute_node.py` 自己定义了一个 `make_hooks_and_matrices(...)`。它不构造 source-to-destination 的完整 edge score，而是把每个 source node 的 activation difference 和该 node 相关 hook 的 gradient 相乘。

关键点：

- `activation_difference` 仍是 `[batch, pos, n_forward, d_model]`。
- 对 input node，在 `hook_embed` 上记录 activation difference 和 gradient。
- 对 attention，每层用 `a{layer}.h0` 作为 hook 入口，但 `graph.forward_index(node)` 默认会返回整个 attention layer 的 head slice，因此可以覆盖所有 head。
- 对 MLP，在 `hook_mlp_out` 记录 output activation difference，在 `hook_mlp_in` 记录 gradient。
- 当 `neuron=False` 时，hidden 维被求和，得到每个 node 一个标量。
- 当 `neuron=True` 时，hidden 维保留，得到 `[n_forward, d_model]`。

### 7.3 attribute_node 调度器

统一入口是：

```python
attribute_node(
    model,
    graph,
    dataloader,
    metric,
    method="EAP-IG-inputs",
    ig_steps=5,
    neuron=True,
)
```

它支持：

- `EAP`
- `EAP-IG-inputs`
- `EAP-IG-activations`
- `exact`

完成后根据 `neuron` 写入不同 graph 字段：

```python
if neuron:
    graph.neurons_scores[:] = scores.to(graph.scores.device)
else:
    graph.nodes_scores[:] = scores.to(graph.scores.device)
```

要使用 neuron-level，必须先让 graph 初始化 `neurons_scores` 和 `neurons_in_graph`：

```python
g = Graph.from_model(model, neuron_level=True, node_scores=True)
attribute_node(model, g, dataloader, metric, method="EAP-IG-inputs", ig_steps=5, neuron=True)
g.apply_topn(n, absolute=True, level="neuron")
```

当前四个任务脚本没有调用 `attribute_node`，只用了 `attribute` 的 edge-level scoring。

## 8. Circuit 评估实现

`src/eap/evaluate.py` 负责验证选择出的 circuit。

### 8.1 evaluate_baseline

`evaluate_baseline(model, dataloader, metrics, run_corrupted=False)` 不做任何 intervention：

- tokenizes clean/corrupted。
- forward clean 和 corrupted。
- 调用 metric。
- 返回每个样本的 metric tensor。

如果 `run_corrupted=True`，metric 使用 corrupted logits 作为主输出。

### 8.2 evaluate_graph

`evaluate_graph(...)` 会实际模拟“只保留 graph 中的 edge/node/neuron，其余被 corrupted/zero/mean 替换”。核心流程：

1. `graph.prune()`，确保 circuit 连通。
2. 把 `graph.in_graph` 转到模型设备，得到 edge mask。
3. 如果有 `graph.neurons_in_graph`，再构造 neuron mask；edge 在 graph 中但 neuron 没全保留时，也会按 neuron mask 局部 patch。
4. 对 mask 取反，因为 hook 里要操作的是“不在 graph 中、需要 corrupted 的部分”。
5. 对每个 batch，用 forward hooks 得到 activation difference。
6. 构造 input construction hooks，把不在 circuit 的 source activation difference 加到 destination input 上。
7. 在 clean tokens 上运行模型，但指定 destination input 被 patch 成“部分 clean + 部分 corrupted”。
8. 调用 metric 得到 circuit performance。

在非 Gemma 一类普通模型中，核心更新是：

```python
update = einsum(
    activation_differences[:, :, :len(in_graph_vector)],
    in_graph_vector,
    "batch pos previous hidden, previous ... -> batch pos ... hidden",
)
activations += update
```

如果有 neuron mask，则变成：

```python
update = einsum(
    activation_differences[:, :, :len(in_graph_vector)],
    neuron_matrix[:len(in_graph_vector)],
    in_graph_vector,
    "batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden",
)
```

这说明 EAP-IG 的 neuron-level circuit 是在 activation hidden dimension 上做局部 patch，不是直接冻结或修改参数。

## 9. 输入、输出和最终结果

### 9.1 输入

运行任务脚本需要：

- 一个 HF 格式本地 checkpoint：`--model_dir`。
- 与 checkpoint 对应的 tokenizer。
- TransformerLens 支持的 base model 名称，例如 `mistralai/Mistral-7B-v0.1`。
- 一个 clean/corrupted CSV：`gender.csv` 或 `induction.csv`，列为 `clean`、`corrupted`、`correct_idx`、`incorrect_idx`。
- 一个 metric 函数：当前脚本使用最后 token 的 correct/incorrect logit difference。
- GPU/CUDA 环境：代码多处 hard-code `device="cuda"` 或 `device="cuda"` tensor。

### 9.2 中间状态

主要中间状态包括：

- tokenized clean/corrupted inputs。
- `attention_mask` 和 `input_lengths`。
- `activation_difference`，edge-level 为 `[batch, pos, n_forward, d_model]`。
- backward hook 中的 gradients。
- edge/node/neuron score tensors。

### 9.3 输出

任务脚本最终输出：

- `args.save_name` 指向的 `.pt` 文件，包含 graph config、edge scores、selected circuit mask 等。
- stdout 中的 baseline/circuit performance：

```text
Original performance was {baseline}; the circuit's performance is {results}
```

可选输出：

- `Graph.to_image(...)` 可导出 PNG circuit 图，但任务脚本中默认注释掉。
- `tocircuitjson.py` 可导出 `graph.json` 和 `score.json`。

## 10. 对当前 gradient mask 二次筛选的参考价值

对当前项目的 gradient mask 二次筛选，EAP-IG 中最有参考价值的是以下机制：

- 用 clean/corrupted 数据对定义“任务相关方向”。当前 finetuning target/pervasiveness 数据也可以类比构造正负或 target/reference 输入。
- 用 hook 记录 component activation，并用 backward hook 记录 metric gradient。
- 用 `activation_difference * gradient` 得到重要性分数，而不是只看参数梯度大小。
- `attribute_node.py` 的 `neuron=True` 已经实现了 per-component hidden dimension score，可作为“component 内 neuron 二次筛选”的直接模板。
- `Graph.neurons_scores` 和 `Graph.neurons_in_graph` 提供了保存 per-component neuron score/mask 的格式参考。
- `evaluate_graph` 展示了如何用 neuron mask 在 activation 层验证 circuit faithfulness。

但需要注意一个概念差异：

- EAP-IG 的 neuron 是 component output activation 的 hidden dimension，形状通常是 `[n_forward, d_model]`。
- 当前 finetuning mask 的 neuron 是参数矩阵中的单个权重值，形状和每个 parameter tensor 完全一致。

因此后续如果要筛选 gradient mask 中每个为 true 的参数值，不能直接复用 `Graph.neurons_scores` 的 shape；需要把 EAP-IG 的“activation hidden dimension score”映射到参数矩阵元素，或者重新设计 parameter-level hook/gradient score。对 MLP 和 attention projection 矩阵，可以考虑把 component activation score 广播/组合到对应 weight 的输入维和输出维；如果需要真正 parameter-level EAP，则需要在参数或模块输出上构造与当前 mask 同形状的 score。

## 11. 重要限制和实现注意点

- 代码强依赖 TransformerLens hook API，不是普通 HF `nn.Module` 直接可用的形式。
- 模型最好是 pre-LayerNorm，因为 residual stream 需要可以被分解为各组件输出之和。
- 对 GQA 模型必须设置 `ungroup_grouped_query_attention=True`。
- 多处 tensor 初始化直接使用 `cuda`，CPU 或自动 device_map 不适配。
- 任务脚本中 CSV 文件名是相对路径，运行时需要在包含 `gender.csv` 或 `induction.csv` 的目录执行，或者改成绝对路径。
- 当前包的 `src/eap/__init__.py` 只定义了 `hello()`，没有统一导出 `Graph`、`attribute` 等对象，所以脚本用的是显式模块 import。
- 当前示例脚本只使用 edge-level `attribute(...)`，没有实际调用 node/neuron-level 的 `attribute_node(...)`。
- `Graph.apply_topn(level="neuron")` 内部会全量 `torch.argsort` `neurons_scores`，如果迁移到 7B parameter-level mask，需要改成更节省内存的阈值/top-k 实现。

## 12. 第二阶段需求文档：EAP_forNeuron

本阶段新增一个独立文件夹 `EAP_forNeuron/`。它的目标不是复现 EAP-IG 的 edge circuit discovery，而是在现有 gradient mask 的基础上，对 mask 中已经为 `True` 的参数级 neuron 重新计算 attribution score，再按比例筛选，输出一个新的 finetuning mask。

这里的 neuron 沿用当前 finetuning mask 的定义：一个 neuron 就是某个参数矩阵中的一个标量权重值。例如 `model.layers.3.self_attn.q_proj.weight[17, 2048]` 是一个 neuron。它不是 EAP-IG 中 `[component, d_model]` 形式的 activation hidden dimension。

### 12.1 功能目标

`EAP_forNeuron` 必须完成以下功能：

1. 从命令行接收一个已有 mask 文件路径。
2. 读取 mask，mask 格式保持当前项目约定：

```python
{
    parameter_name: torch.BoolTensor(shape == model_parameter.shape)
}
```

3. 只对 `mask[parameter_name] == True` 的 neuron 计算 attribution score。
4. 不计算 component-level score，也不计算 edge attribution score。
5. 用类似 EAP 的思想计算每个参数级 neuron 的重要性：clean/corrupted 输入带来的该 neuron 激活贡献差异，乘以 clean loss 对该 neuron 输出位置的梯度，再聚合为一个标量。
6. 按 score 从高到低排序，只保留满足指定比例的 neuron 为 `True`。默认比例 `output_ratio` 以所有可筛选 neuron 总数为分母；输出 mask 仍然必须是输入 mask 的子集，所以实际保留数会被输入 mask 中的 `True` 候选数上限截断。
7. 生成新 mask，保存到给定输出文件夹。
8. 输出 score 统计、保留比例统计和日志，方便复查。

### 12.2 与现有 gradient mask 的关系

当前 `src/unlearn/generate_mask.py` 生成的 gradient mask 已经是参数同形状 bool mask，并在 finetuning 中通过 `src/finetuning/base.py::mask_gradient(...)` 使用：

```python
for key, tensor in model.named_parameters():
    if tensor.grad is not None and key in self.mask:
        tensor.grad *= self.mask[key].to(tensor.grad.device)
```

因此 EAP_forNeuron 的输出 mask 必须保持同样格式和 key，使它能被 `finetuning_model.py` 通过 `mask_path` 直接加载。新的 mask 应该是旧 mask 的子集：

```python
new_mask[key] = old_mask[key] & selected_by_eap[key]
```

也就是说，旧 mask 中为 `False` 的位置永远不能被重新打开；EAP_forNeuron 只负责在旧 mask 的 `True` 候选集合里继续筛掉一部分。

### 12.3 输入输出约定

#### 12.3.1 命令行输入

建议新增入口脚本：

```text
EAP_forNeuron/run_eap_for_neuron.py
```

建议参数：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `--model_name_or_path` | str | HF 模型或本地 checkpoint 路径，默认可与 finetuning 使用的 `mistralai/Mistral-7B-v0.1` 对齐。 |
| `--tokenizer_name_or_path` | str | tokenizer 路径，默认等于 model path。 |
| `--mask_path` | str | 输入 gradient mask 文件路径。必填。 |
| `--output_dir` | str | 新 mask、score、summary 保存目录。必填。 |
| `--dataset_name` | str | `bool`、`gender`、`ioi_mistral` 三选一。 |
| `--data_path` | str | 可选，pair CSV 路径。默认从 `EAP_forNeuron/data/{dataset_name}.csv` 或复制后的默认路径读取。 |
| `--output_ratio` | float | 输出 mask 的目标 `True` 比例。默认以所有可筛选 neuron 总数为分母。 |
| `--ratio_base` | str | `all` 或 `candidate`。默认 `all`；`candidate` 表示比例只相对于输入 mask 中的 `True` 候选集合。 |
| `--max_samples` | int | attribution 使用的最大样本数。用于显存/时间控制。 |
| `--batch_size` | int | dataloader batch size。 |
| `--max_length` | int | tokenizer 最大长度。 |
| `--score_abs` | bool | 是否按 attribution score 的绝对值排序。默认建议 `True`。 |
| `--target_modules` | str | 逗号分隔的模块后缀。默认 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。 |
| `--include_lm_head` | bool | 是否处理 `lm_head.weight`。默认 `False`。 |
| `--include_embed_tokens` | bool | 是否处理 embedding 权重。默认 `False`，第一版不建议开启。 |
| `--score_dtype` | str | score 保存 dtype，建议 `float32` 或 `float16`。 |
| `--device` | str | 默认 `cuda:0`。 |
| `--use_bfloat16` | bool | 默认 `True`，与当前 Blackwell/Mistral 训练路径一致。 |

#### 12.3.2 输入 mask

输入 mask 由 `torch.load(mask_path, map_location="cpu")` 读取。加载后必须校验：

- 对每个 `model.named_parameters()` 中的 key，如果 mask 存在，则 `mask[key].shape == parameter.shape`。
- mask dtype 必须能转换为 bool。
- 对 unsupported 参数，如果 mask 中存在 `True`，默认不计算 score，输出 mask 对这些位置置为 `False`，同时记录 warning。也可以通过 `--preserve_unsupported_true` 改为保留，但默认不建议。

#### 12.3.3 输出文件

输出目录建议包含：

```text
output_dir/
    with_{output_ratio}.pt
    scores.pt
    selected_scores.pt
    summary.json
    skipped_parameters.json
```

其中：

- `with_{output_ratio}.pt`：新的 bool mask，格式与输入 mask 一致，可直接给 finetuning `mask_path` 使用。
- `scores.pt`：和参数同形状的 score dict。只要求候选位置有有效 score；非候选位置可以是 `nan` 或 0，但推荐 `nan`，避免误读。
- `selected_scores.pt`：可选，保存被保留 neuron 的 `(parameter_name, flat_index, score)` 紧凑列表，便于分析。
- `summary.json`：保存所有可筛选 neuron 总数、输入候选数、最终保留数、比例分母、各参数保留比例、阈值、数据集、模型、运行参数。
- `skipped_parameters.json`：记录因为不是 supported matrix、shape 不匹配、没有梯度、没有 hook 等原因被跳过的参数。

### 12.4 数据集需求

本阶段暂时复制并支持三个 pair 数据集：`bool`、`gender`、`ioi_mistral`。

`EAP_forNeuron` 不应直接使用当前 finetuning wrapper 作为 clean/corrupted pair 的唯一来源，因为当前 `src/dataset` 的 `IOI`、`gender`、`bool` wrapper 主要输出 `input_ids/attention_mask/labels`，没有显式 corrupted prompt。如果 clean 和 corrupted prompt 相同，只交换 label，则 activation difference 为 0，EAP 分数失效。

建议新增：

```text
EAP_forNeuron/data/bool.csv
EAP_forNeuron/data/gender.csv
EAP_forNeuron/data/ioi_mistral.csv
```

第一版可从 `EAP-IG/bool.csv`、`EAP-IG/gender.csv`、`EAP-IG/ioi_mistral.csv` 复制。CSV 需要至少包含：

```text
clean,corrupted,correct_idx,incorrect_idx
```

`ioi_mistral.csv` 还包含 `corrupted_hard`，第一版默认不用；可以在参数 `--corruption_column corrupted|corrupted_hard` 中选择。

#### 12.4.1 数据类设计

新增文件：

```text
EAP_forNeuron/datasets.py
```

建议定义：

```python
class PairExample:
    clean: str
    corrupted: str
    correct_idx: int
    incorrect_idx: int
```

```python
class EAPNeuronPairDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, tokenizer, max_length, corruption_column="corrupted")
    def __len__(self)
    def __getitem__(self, index)
```

`__getitem__` 返回原始 string 和 label token id，不在这里做 tensor padding，便于 collator 统一处理。

```python
class EAPNeuronCollator:
    def __init__(self, tokenizer, max_length)
    def __call__(self, examples)
```

collator 输出：

```python
{
    "clean_input_ids": LongTensor[batch, seq],
    "clean_attention_mask": LongTensor[batch, seq],
    "corrupted_input_ids": LongTensor[batch, seq],
    "corrupted_attention_mask": LongTensor[batch, seq],
    "labels": LongTensor[batch, seq],
    "correct_idx": LongTensor[batch],
    "incorrect_idx": LongTensor[batch],
    "label_positions": LongTensor[batch],
}
```

`labels` 只在 clean prompt 最后一个有效 token 位置写入 `correct_idx`，其余为 `-100`。这与 `training_losses.task_loss(...)` 的单 token task 语义一致。

#### 12.4.2 metric / loss 设计

第一版建议使用和当前训练一致的 target loss：

```python
loss, outputs = task_loss(model, (clean_input_ids, clean_attention_mask, labels))
```

也可以额外实现 `logit_diff_loss`，用于和 EAP-IG CSV 中的 `correct_idx/incorrect_idx` 对齐：

```python
loss = -(logit[correct_idx] - logit[incorrect_idx]).mean()
```

推荐设计为可选参数：

| `--metric` | 含义 |
| --- | --- |
| `task_loss` | 默认，复用当前项目的 local causal LM loss。 |
| `logit_diff` | 复用 EAP-IG 风格的 correct/incorrect logit diff。 |

注意：不要把 `labels` 传给 Mistral forward；必须沿用 `training_losses.task_loss` 或本地 logits loss，避免恢复之前 Blackwell/Mistral 上的 CUDA assert 问题。

### 12.5 参数级 neuron attribution 公式

只考虑 `nn.Linear` 权重矩阵时，设某个参数矩阵：

```text
W shape = [out_features, in_features]
Y = X @ W.T
```

其中 `X` 是该 Linear 的输入 activation，flatten 后：

```text
X_clean shape = [D, in_features]
X_corrupted shape = [D, in_features]
grad_Y_clean shape = [D, out_features]
```

`D` 是参与统计的 activation 行数，通常是 `batch * seq_len` 中 attention mask 为 1 的 token 行。对于一个参数级 neuron `W[o, i]`，它在每个 token 行上的 clean/corrupted 输出贡献是：

```text
contrib_clean[d] = X_clean[d, i] * W[o, i]
contrib_corrupted[d] = X_corrupted[d, i] * W[o, i]
```

它的 contribution difference 是一个 `D * 1` 向量：

```text
delta_contrib[d] = (X_corrupted[d, i] - X_clean[d, i]) * W[o, i]
```

对应梯度是 clean loss 对该 Linear 输出维度 `o` 的梯度：

```text
grad[d] = d loss / d Y_clean[d, o]
```

最终 attribution score 聚合为一个标量：

```text
score[W[o, i]] = sum_d delta_contrib[d] * grad[d]
```

如果使用 `score_abs=True`，筛选时使用 `abs(score)`，但保存的原始 score 应保留符号。

这个公式只计算与 `W[o, i]` 相关的乘积，不需要构造完整 `Y` 或完整 `[out_features, in_features]` 的中间贡献矩阵。实现时应只在 mask 为 `True` 的候选坐标上计算。

### 12.6 核心算法流程

对每个 batch：

1. 从 dataloader 取 clean/corrupted input。
2. 在 corrupted 输入上 forward 一次，只记录目标 Linear 模块的输入 activation `X_corrupted`。这个 forward 用 `torch.no_grad()`。
3. 在 clean 输入上 forward 一次，记录目标 Linear 模块的输入 activation `X_clean`，同时对目标 Linear 模块输出注册 gradient hook，记录 `grad_Y_clean`。
4. 用 `task_loss` 或 `logit_diff_loss` 计算 loss。
5. 执行 backward，使每个目标模块拿到 output gradient。
6. 对每个参数矩阵的 mask true 坐标，计算并累加 `score[W[o, i]] += sum_d ((X_corrupted[:, i] - X_clean[:, i]) * W[o, i] * grad_Y_clean[:, o])`。
7. 清空梯度、释放 activation 缓存。
8. 所有 batch 完成后，按样本数或有效 token 数归一化 score。
9. 在所有候选 score 上按比例筛选，生成新 mask。

为了节省显存，activation 缓存应该只保存当前 batch，不跨 batch 保存。score dict 放 CPU 上累加。

### 12.7 支持的模块范围

第一版建议只支持 Mistral/Llama 类模型中的 Linear 权重：

- `self_attn.q_proj.weight`
- `self_attn.k_proj.weight`
- `self_attn.v_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_proj.weight`
- `mlp.up_proj.weight`
- `mlp.down_proj.weight`

可选支持：

- `lm_head.weight`：输入 activation 是 final hidden states，输出是 vocab logits；参数很大，默认关闭。
- `model.embed_tokens.weight`：不是普通 Linear forward，第一版默认不支持。

默认不支持：

- LayerNorm/RMSNorm 参数。
- bias 参数。
- LoRA adapter 参数。
- 非二维参数。
- 任何无法从 `named_modules()` 映射到 `nn.Linear.weight` 的 key。

对于 unsupported 参数，输出策略必须明确。建议默认：

```python
new_mask[key] = torch.zeros_like(old_mask[key], dtype=torch.bool)
```

并在 `skipped_parameters.json` 中记录原因。这样最终 mask 的真实保留比例可控，不会因为跳过参数而意外保留未打分 neuron。

### 12.8 关键类设计

建议文件结构：

```text
EAP_forNeuron/
    README.md
    __init__.py
    run_eap_for_neuron.py
    datasets.py
    model_loader.py
    mask_io.py
    hooks.py
    scorer.py
    selector.py
    utils.py
    data/
        bool.csv
        gender.csv
        ioi_mistral.csv
```

#### 12.8.1 `NeuronCoordinate`

文件：`EAP_forNeuron/mask_io.py`

用途：描述一个参数级 neuron。

```python
@dataclass(frozen=True)
class NeuronCoordinate:
    parameter_name: str
    module_name: str
    row: int
    col: int
    flat_index: int
```

对于 Linear weight：`row` 对应 output dimension，`col` 对应 input dimension。

#### 12.8.2 `MaskSpec`

文件：`EAP_forNeuron/mask_io.py`

用途：加载和校验输入 mask。

```python
class MaskSpec:
    @classmethod
    def load(cls, mask_path: str, model: nn.Module, target_modules: list[str]) -> "MaskSpec"
    def candidate_count(self) -> int
    def iter_parameter_masks(self)
    def empty_output_mask(self) -> dict[str, torch.Tensor]
```

职责：

- 读取 mask。
- 校验 key/shape/dtype。
- 建立 `parameter_name -> module_name` 映射。
- 筛选支持的 Linear weight 参数。
- 统计候选数量。

#### 12.8.3 `LinearActivationCache`

文件：`EAP_forNeuron/hooks.py`

用途：管理 forward hook 和 backward hook。

```python
class LinearActivationCache:
    def __init__(self, modules: dict[str, nn.Linear])
    def collect_corrupted_inputs(self, model, batch) -> dict[str, torch.Tensor]
    def collect_clean_inputs_and_grads(self, model, batch, loss_fn) -> tuple[dict, dict, torch.Tensor]
    def clear(self)
```

需要捕获：

- `clean_inputs[module_name]`: Linear input activation，shape `[batch, seq, in_features]`。
- `corrupted_inputs[module_name]`: 同形状。
- `clean_output_grads[module_name]`: Linear output gradient，shape `[batch, seq, out_features]`。

实现要求：

- corrupted forward 使用 `torch.no_grad()`。
- clean forward 需要 autograd。
- hook 返回原始 activation，不修改模型行为。
- 如果模块 input 是 tuple，只取第一个 tensor。
- 每个 batch 结束后移除 hook 或清空缓存，避免内存泄漏。

#### 12.8.4 `ParameterNeuronScorer`

文件：`EAP_forNeuron/scorer.py`

用途：计算并累加 parameter-level attribution score。

```python
class ParameterNeuronScorer:
    def __init__(self, model, mask_spec, score_dtype=torch.float32, device="cuda")
    def score_batch(self, clean_inputs, corrupted_inputs, clean_output_grads, attention_mask)
    def finalize(self, normalizer: float | None = None) -> dict[str, torch.Tensor]
```

内部状态：

```python
self.scores = {
    parameter_name: torch.full(parameter.shape, torch.nan, dtype=score_dtype, device="cpu")
}
```

非候选位置保持 `nan`。候选位置初始化为 0 后累加。

核心函数：

```python
def score_linear_weight(
    weight: torch.Tensor,
    candidate_mask: torch.BoolTensor,
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    output_grad: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    ...
```

内存友好计算策略：

- 将 `[batch, seq, hidden]` flatten 为 `[D, hidden]`。
- 如果给了 `attention_mask`，只保留有效 token 行。
- 对候选坐标按 row 分组。
- 对同一个 output row `o`，取 `grad_o = output_grad[:, o]`。
- 对该 row 的候选 input cols `cols`，计算：

```python
delta_x = corrupted_input[:, cols] - clean_input[:, cols]
scores[o, cols] += (delta_x * grad_o[:, None]).sum(dim=0) * weight[o, cols]
```

这样避免构造 `[D, out_features, in_features]` 的巨大张量。

#### 12.8.5 `NeuronMaskSelector`

文件：`EAP_forNeuron/selector.py`

用途：按比例从候选 score 中生成新 mask。

```python
class NeuronMaskSelector:
    def __init__(self, output_ratio: float, ratio_base: str = "all", score_abs: bool = True)
    def select(self, old_mask: dict[str, Tensor], scores: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict]
```

要求：

- `output_ratio` 表示输出 mask 的目标 `True` 比例。
- `ratio_base="all"` 时，分母是所有支持打分的参数级 neuron 总数，`target_keep_count = int(total_neuron_count * output_ratio)`。
- `ratio_base="candidate"` 时，分母是输入 mask 中已经为 `True` 的候选数，`target_keep_count = int(candidate_count * output_ratio)`。
- 实际保留数必须满足 `actual_keep_count <= candidate_count`，即 `actual_keep_count = min(candidate_count, target_keep_count)`。
- 只在 `old_mask == True` 且 score 有效的位置中选择。
- 按 score 降序选择。
- 对 7B 参数量，不能使用会产生多个全量排序副本的实现；建议用阈值或分块 top-k。

第一版可采用两级策略：

1. 如果候选数量低于安全阈值，例如 50M，可以 concat 有效 score 后 `torch.topk`。
2. 如果候选数量更大，用二分阈值或 chunk top-k，避免 OOM。

输出 summary 至少包含：

```python
{
    "total_neuron_count": int,
    "candidate_count": int,
    "target_keep_count": int,
    "actual_keep_count": int,
    "output_ratio": float,
    "ratio_base": "all" | "candidate",
    "threshold": float,
    "per_parameter": {
        parameter_name: {
            "candidates": int,
            "kept": int,
            "skipped": int,
        }
    }
}
```

#### 12.8.6 `EAPForNeuronRunner`

文件：`EAP_forNeuron/runner.py` 或直接在 `run_eap_for_neuron.py` 中定义。

```python
class EAPForNeuronRunner:
    def __init__(self, config)
    def load_model_and_tokenizer(self)
    def load_dataset(self)
    def load_mask(self)
    def run_attribution(self)
    def select_and_save(self)
    def run(self)
```

职责：串联模型、数据、mask、hook、score、selector、保存。

### 12.9 函数设计清单

建议最小函数集合：

```python
def parse_args() -> argparse.Namespace
def load_model(model_name_or_path, tokenizer_name_or_path, device, use_bfloat16)
def load_pair_dataset(dataset_name, data_path, tokenizer, max_length, max_samples)
def load_mask(mask_path, model) -> dict[str, torch.BoolTensor]
def find_supported_linear_modules(model, target_modules) -> dict[str, nn.Linear]
def parameter_name_to_module_name(parameter_name: str) -> str
def build_candidate_masks(mask, model, supported_modules) -> tuple[dict, dict]
def make_task_loss_fn(metric: str)
def score_linear_weight(...)
def merge_scores_into_mask(old_mask, scores, output_ratio, ratio_base, score_abs)
def save_outputs(output_dir, new_mask, scores, summary, skipped)
```

### 12.10 与现有代码的新增、修改、删除

#### 12.10.1 新增

必须新增：

- `EAP_forNeuron/` 文件夹。
- `EAP_forNeuron/run_eap_for_neuron.py`：命令行入口。
- `EAP_forNeuron/datasets.py`：复制并规范 `bool/gender/ioi_mistral` pair 数据集读取。
- `EAP_forNeuron/model_loader.py`：加载 HF model/tokenizer，复用当前 Mistral patch 和 dtype/device 约定。
- `EAP_forNeuron/mask_io.py`：读取、校验、保存 mask 和 score。
- `EAP_forNeuron/hooks.py`：Linear input activation 和 output gradient hook。
- `EAP_forNeuron/scorer.py`：参数级 neuron attribution 计算。
- `EAP_forNeuron/selector.py`：按比例筛选 mask。
- `EAP_forNeuron/README.md`：运行说明、输入输出格式和示例命令。
- `EAP_forNeuron/data/bool.csv`、`gender.csv`、`ioi_mistral.csv`：第一版可从 `EAP-IG` 复制。

建议新增测试或 smoke 脚本：

- `EAP_forNeuron/tests/test_selector.py`：验证比例筛选、tie、nan、旧 mask false 不会被打开。
- `EAP_forNeuron/tests/test_scorer_tiny.py`：用 tiny Linear 模型验证公式。
- `EAP_forNeuron/tests/test_mask_io.py`：验证 mask shape/dtype/key 校验。

#### 12.10.2 修改

第一版可以不修改训练主流程，只要输出 mask 格式保持兼容即可。后续可选修改：

- `src/exec/finetuning_model.py`：新增一个可选参数，例如 `mask_refine_path` 或 `eap_neuron_mask_path`，用于自动调用 EAP_forNeuron；第一版不建议耦合，先独立脚本运行。
- `src/model/finetuning.py`：如果未来需要在 mask 不存在时自动先 gradient mask 再 EAP_forNeuron refine，可以在 `_generate_mask` 后接一个 refine step；第一版不做，避免训练入口复杂化。
- `current_functionality_requirements.md`：后续实现完成后应补充新功能说明。
- `enviroment.md`：如果引入新依赖才修改；按当前设计只用已有 `torch/transformers/pandas/tqdm`，无需新增依赖。

#### 12.10.3 删除

本阶段不删除现有代码。尤其不要删除或重命名：

- `src/unlearn/generate_mask.py` 中现有 gradient mask 生成功能。
- `src/finetuning/base.py::mask_gradient(...)`。
- `src/dataset` 中已有 `IOI/gender/bool` wrapper。
- `EAP-IG/` 原始参考实现。

EAP_forNeuron 应作为独立 refine 工具存在，输出兼容 mask 即可。

### 12.11 运行示例

建议命令形态：

```bash
cd /ssd_users/chenhang/CSAT
/home/chenhang/.conda/envs/LLMSFT_BW/bin/python EAP_forNeuron/run_eap_for_neuron.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --tokenizer_name_or_path mistralai/Mistral-7B-v0.1 \
  --mask_path files/masks/gradient/IOI/with_0.2.pt \
  --dataset_name ioi_mistral \
  --data_path EAP_forNeuron/data/ioi_mistral.csv \
  --output_dir files/masks/eap_for_neuron/IOI \
  --output_ratio 0.1 \
  --ratio_base all \
  --batch_size 2 \
  --max_samples 128
```

如果输入 mask 中有 20% 可筛选 neuron 为 `True`，`--output_ratio 0.1 --ratio_base all` 的含义是最终输出大约 10% 可筛选 neuron 为 `True`，并且这些 `True` 全部来自输入 mask 的候选集合。若输入候选数不足 10%，则输出最多只能保留输入 mask 中已有的 `True`。

如果希望从输入候选集合中再保留一半，则使用：

```text
--output_ratio 0.5 --ratio_base candidate
```

### 12.12 验证标准

实现完成后至少需要以下验证：

1. Tiny Linear 模型公式测试：手工计算 `sum((x_corr - x_clean) * w * grad)`，与 `score_linear_weight` 输出一致。
2. Mask 子集测试：输出 mask 任意位置满足 `new_mask[key] <= old_mask[key]`。
3. 比例测试：`ratio_base="all"` 时输出 True 数等于 `min(candidate_count, int(total_neuron_count * output_ratio))`；`ratio_base="candidate"` 时输出 True 数等于 `int(candidate_count * output_ratio)`，除非有效 score 数不足。
4. 数据集 smoke test：`bool/gender/ioi_mistral` 都能读取一个 batch，clean/corrupted input 不完全相同。
5. Mistral tiny batch smoke test：在 `max_samples=1`、`batch_size=1` 下跑通 hook、loss、backward、score 保存。
6. 与 finetuning 兼容测试：生成的 `with_{output_ratio}.pt` 能被 `src/finetuning/base.py::mask_gradient(...)` 正常加载和应用。
7. 内存测试：不得在 7B 全模型候选上构造 `[D, out_features, in_features]` 或多个全量排序副本。

### 12.13 风险和注意事项

- 如果 clean/corrupted token 长度差异很大，按同一 position 做 activation difference 可能语义不完全对齐。第一版要求 pair CSV 尽量保持 clean/corrupted 模板长度接近，并用 attention mask 排除 padding。
- bool.csv 中可能存在 `correct_idx == incorrect_idx` 的样本；如果使用 `logit_diff` metric，这类样本没有区分度，应过滤或改用 `task_loss`。
- 对 `o_proj`、`down_proj` 这类输入来自前面非线性或 attention 聚合后的矩阵，公式仍成立，因为它们本身是 Linear；但 score 解释是该权重对该 Linear 输出的局部贡献，不是完整 causal path attribution。
- 对 `gate_proj/up_proj/down_proj`，Mistral MLP 中存在 `act(gate_proj(x)) * up_proj(x)` 的非线性乘积。第一版按每个 Linear 的局部输入和输出梯度计算 attribution，梯度会自动包含后续非线性影响，因此不需要手写 MLP 公式。
- 对 q/k/v，后续 attention softmax 的影响也通过 output gradient 反传体现；不需要显式重算 attention。
- 输出 score 的符号代表该 neuron 的 corrupted-clean contribution 对 loss 的方向影响。筛选时默认用绝对值，因为“重要性”通常不关心方向；如果只想保留使 target loss 降低或升高的 neuron，可以关闭 `score_abs` 并按有符号 score 选择。
- 大 mask 上的排序必须沿用上一次 gradient mask OOM 修复的经验：使用阈值、chunk top-k 或流式统计，不要对全模型候选构造多个巨大 `argsort` 张量。

## 13. 第三阶段需求文档：EAP_forComponent

`EAP_forNeuron` 的问题在于粒度过细：即使输入 mask 已经冻结 80% 参数，剩下 20% 对 Mistral-7B 来说仍然可能是十亿级别的参数标量。逐 neuron 计算 attribution score 会导致第一批样本就非常慢，且 CPU/GPU 数据搬运和 score buffer 都很重。

因此第三阶段新增 `EAP_forComponent/`。它把粒度从“参数矩阵中的单个标量权重”提升到更粗的 component。对 Mistral/LLaMA 类模型，第一版仍然只处理每层 attention 的 `q_proj/k_proj/v_proj/o_proj` 和 MLP 的 `gate_proj/up_proj/down_proj` 七类 Linear，但 attention 部分必须支持两种 attribution 粒度：

- `projection_matrix`：对同一层同一个 attention projection 中的所有 head 一起计算一个 attribution score。例如整个 `model.layers.12.self_attn.q_proj.weight` 得到一个 score。这是最容易直接对接 PEFT `rank_pattern` 的矩阵级模式。
- `head`：在同一层同一个 attention projection 内按 head slice 计算 attribution score。例如 `model.layers.12.self_attn.q_proj.head_0`、`head_1` 分别得到 score。MLP 的 `gate/up/down` 不存在 attention head，仍然保持矩阵级 component。

Mistral-7B 通常是 32 层。`projection_matrix` 模式下 component 数量约为 `32 * 7 = 224`；`head` 模式下 attention component 会增加到每层按 head 展开的 q/k/v/o slice，但仍远小于参数级 neuron 数量。

EAP_forComponent 的直接用途有两个：

1. 根据 component attribution score 给 LoRA 分配不同 rank：score 越高，对应矩阵或聚合后的矩阵 LoRA rank 越大。
2. 生成一个与现有 mask 格式兼容的 component-rank mask，用作“不使用 LoRA，只用 gradient mask 控制训练参数比例”的对照实验。

### 13.1 功能目标

`EAP_forComponent` 必须完成以下功能：

1. 加载 Mistral 模型本身，不需要输入已有 neuron mask。
2. 枚举模型中的目标 Linear，只包含：

```text
model.layers.*.self_attn.q_proj.weight
model.layers.*.self_attn.k_proj.weight
model.layers.*.self_attn.v_proj.weight
model.layers.*.self_attn.o_proj.weight
model.layers.*.mlp.gate_proj.weight
model.layers.*.mlp.up_proj.weight
model.layers.*.mlp.down_proj.weight
```

3. 不计算 edge attribution，不构建 EAP-IG 的 edge graph。
4. 不计算参数级 neuron attribution，只给粗粒度 component 生成标量 score。
5. Attention component 必须支持两种粒度：
    - `projection_matrix`：同一层同一 projection 的所有 head 一起算一个 score。
    - `head`：同一层同一 projection 内按 head slice 分别算 score。
6. MLP component 暂时只支持矩阵级，即每个 `gate_proj/up_proj/down_proj` 矩阵一个 score。
7. 使用 clean/corrupted 输入对，提取所需的 clean activation、corrupted activation，以及目标 metric 对对应 activation 的梯度。
8. Score token 聚合必须支持两种模式：
    - `all_active`：所有 attention mask 为 1 的 token 位置参与 attribution 聚合。
    - `label_position`：只在 label 对应的最后有效 token 位置参与 attribution 聚合。
9. Attribution 公式必须支持两种 localization 模式：
    - `current` / `current_localization`：只使用当前基础参数 `theta` 下的 clean/corrupted activation 差值和局部 activation 梯度。
    - `future` / `future_localization`：额外输入简单微调后的参数 `theta'`，根据 `theta' - theta` 构造理想参数 `theta_hat^I = theta + K * (theta' - theta)`，并使用梯形积分形式的一阶参数扰动修正 score。
10. 对每个 component 计算 attribution scalar，保存并排序。
11. 根据 score 为目标 component 分配 LoRA rank。矩阵级 score 可以直接输出 PEFT `LoraConfig(rank_pattern=...)`；head 级 score 需要额外聚合回矩阵级 rank pattern，或配合自定义 head-wise LoRA 使用。
12. 根据 score 排序生成一个 bool mask：最高 score 的 component 100% 参数为 `True`，最低 score 的 component 0% 参数为 `True`，中间 component 的 `True` 比例按排序线性递减。
13. 输出所有结果到指定目录，包括 component score、rank pattern、mask 和 summary。

这里的 component 不是 EAP_forNeuron 中的单个参数标量。它在 `projection_matrix` 模式下是一个完整参数矩阵，在 `head` 模式下是 attention projection 矩阵中的一个 head slice。

### 13.2 与 EAP-IG 的关系

EAP-IG 原始 node/neuron scoring 的核心思想是：

```text
score = sum((activation_corrupted - activation_clean) * gradient_clean)
```

EAP_forComponent 继续使用这个思想，但不使用 TransformerLens `Graph`。它直接对 HF `nn.Linear` 模块注册 hook：

- corrupted forward：保存目标 Linear 的 output activation。
- clean forward：保存目标 Linear 的 output activation。
- metric backward：根据 `localization_mode` 对指定 activation 注册 backward hook。按照本阶段公式，`current_localization` 和 `future_localization` 都需要得到目标 metric 对 corrupted-side activation `A^theta(R)` 的局部梯度；如果为了复现实验保留旧 clean-gradient 路径，必须在 summary 中显式记录 gradient 来源。

因此它更接近 EAP-IG 中的 node-level attribution，而不是 edge-level attribution。差异是 EAP-IG 的 node 通常是 attention head / MLP output，EAP_forComponent 的 node 是当前项目可用于 LoRA/mask 分配的训练 component：可以是完整 Linear 矩阵，也可以是 attention projection 内部的 head slice。

### 13.3 Component attribution 公式

EAP_forComponent 需要把 attribution score 的计算方式显式命名为 localization mode。第一版已有公式命名为 `current_localization`，新增公式命名为 `future_localization`。命令行通过 `--localization_mode current|future` 切换；默认必须保持 `current`，以保证与已有实验结果兼容。

本节统一记号：

- `A`：当前正在计算 score 的 component，可以是完整 projection 矩阵，也可以是 attention head slice。
- `theta`：基础 Mistral 参数，即 `--model_name_or_path` 加载的默认参数。
- `theta'`：一个经过简单 finetuning 后得到的 Mistral 参数，由新增命令行参数提供。
- `C`：clean 输入。
- `R`：corrupted 输入。
- `A^theta(C)`：参数为 `theta` 时 component `A` 在 clean 输入上的 activation。
- `A^theta(R)`：参数为 `theta` 时 component `A` 在 corrupted 输入上的 activation。
- `g_A^theta(R)`：目标 metric `f_A^theta` 对 corrupted activation `A^theta(R)` 反传得到的局部梯度，即 `partial f_A^theta / partial A^theta(R)`。

注意：现有实现里符号可能写作 `corrupted - clean` 再乘梯度；需求文档从本次开始统一采用用户推导中的 `clean - corrupted` 方向。实现时必须把符号约定写入 `summary.json`，例如 `delta_activation_convention="clean_minus_corrupted"`，避免不同 run 的 signed score 被混用。rank/mask 默认仍然使用绝对值重要性，所以符号改变不会影响只看 `abs/rank_score` 的排序实验，但会影响 signed score 分析。

对一个目标 Linear module：

```python
Y = X @ W.T + b
```

EAP_forComponent 不再展开 `W[o, i]` 的每个元素，而是把一个完整矩阵或一个 head slice 视作 component。

#### 13.3.1 `projection_matrix` 粒度

以下公式描述 `current_localization` 下的矩阵级计算。`projection_matrix` 模式把整个 Linear output `Y` 作为 component activation。对 batch 中选中的 token 行 flatten 后：

```text
Y_clean shape = [D, out_features]
Y_corrupted shape = [D, out_features]
grad_Y_corrupted shape = [D, out_features]
```

矩阵级 raw score 定义为：

```text
raw_score_current[W] = sum_{d,h} (Y_clean[d,h] - Y_corrupted[d,h]) * grad_Y_corrupted[d,h]
```

#### 13.3.2 `head` 粒度

`head` 模式只对 attention projection 拆 head。对 `q_proj/k_proj/v_proj`，head component 是 output hidden 维上的连续 row slice：

```text
head_rows(h) = [h * head_dim, (h + 1) * head_dim)
raw_score_current[W.head_h] = sum_{d,hid in head_rows(h)} (Y_clean[d,hid] - Y_corrupted[d,hid]) * grad_Y_corrupted[d,hid]
```

对 `o_proj`，head component 更自然地对应 attention result 拼接后的 input head slice，而不是 output row slice。因为 `o_proj.weight` 会把所有 head 的输入混合到 residual hidden 维，建议按 input column slice 计算每个 head 对 `o_proj` output 的贡献：

```text
head_cols(h) = [h * head_dim, (h + 1) * head_dim)
Y_h = X[:, head_cols(h)] @ W[:, head_cols(h)].T
raw_score_current[o_proj.head_h] = sum_{d,hid} (Y_h_clean[d,hid] - Y_h_corrupted[d,hid]) * grad_Y_corrupted[d,hid]
```

因此 head 模式下 hook 需要保存 attention projection 的 output activation，也需要在 `o_proj` 上保存 input activation 和 output gradient。

对 Mistral 这类 GQA 模型需要单独记录 head 类型：

- `q_proj` 和 `o_proj` 按 query head 数 `num_attention_heads` 切分。
- `k_proj` 和 `v_proj` 按 key/value head 数 `num_key_value_heads` 切分。
- 如果后续希望把 k/v score 对齐到 query head，需要在 summary 中记录 `query_heads_per_kv_head`，但第一版默认保留真实的 KV head 粒度。

MLP 的 `gate_proj/up_proj/down_proj` 没有 head 概念，即使 `--attention_granularity head`，它们仍然按完整矩阵计算。

#### 13.3.3 Score token 模式

`D` 是参与 attribution 聚合的 token 行数。EAP_forComponent 必须实现两种 token 模式：

| `--score_token_mode` | 参与 attribution 聚合的 token 位置 |
| --- | --- |
| `all_active` | 所有 `attention_mask == 1` 的 token 位置。这个模式最接近 EAP-IG 原始实现。 |
| `label_position` | 每个样本 label 对应的最后有效 token 位置。这个模式更贴近当前准确率只看最后答案 token 的评估语义，也更省显存和时间。 |

注意：`score_token_mode` 控制的是 attribution 聚合时使用哪些 activation/gradient 行；loss/metric 本身仍然可以只在最后答案 token 上定义，然后梯度会通过模型反传到相关 token 位置。

筛选和 rank 分配通常使用重要性强度：

```text
rank_score[component] = abs(raw_score[component])
```

由于 `q/k/v/o/gate/up/down` 的输出维度和 component 粒度差异很大，必须同时保存 raw score 和 normalized score。建议第一版支持：

| `--score_normalization` | 含义 |
| --- | --- |
| `sum` | 使用原始总和，保留矩阵总体影响大小。 |
| `mean` | 除以参与统计的元素数 `D * out_features`，减少大矩阵维度偏置。 |
| `sqrt_numel` | 除以 `sqrt(parameter_numel)`，在总影响和维度归一之间折中。 |

默认建议 rank 分配使用 `mean` 或 `sqrt_numel` 的绝对值，同时在输出文件中保存所有版本，便于后续分析。输出中必须记录当前使用的 `attention_granularity` 和 `score_token_mode`，避免把不同粒度或不同 token 聚合模式的 score 混在一起比较。

#### 13.3.4 `current_localization` 与 `future_localization`

`current_localization` 是当前已有方法。对 component `A`，先在基础参数 `theta` 下得到：

```text
Delta_A(theta) = A^theta(C) - A^theta(R)
g_A(theta) = partial f_A^theta / partial A^theta(R)
```

则当前 localization 的 scalar attribution 为：

```text
Attr_current(A; theta) = Delta_A(theta) dot g_A(theta)
```

这里的 `dot` 表示对 `score_token_mode` 选中的 token 行和 component feature 维度求和。`projection_matrix/head/o_proj head input slice` 的切片规则仍然沿用 13.3.1 和 13.3.2。

新增 `future_localization` 需要额外输入一个简单 finetuning 后的模型参数 `theta'`，并给定超参数 `K`。整体流程如下：

1. 加载基础模型参数 `theta`，即默认 Mistral 参数。
2. 加载 finetuned 模型参数 `theta'`。`theta'` 必须与 `theta` 架构一致，并且目标参数名和 shape 必须能逐项对齐。
3. 在 `theta` 下分别计算 clean/corrupted activation，得到：

```text
Delta_A(theta) = A^theta(C) - A^theta(R)
```

4. 在 `theta` 下计算当前 localization 对参数的方向导数项 `S(theta)`：

```text
S(theta) = [
    (partial A^theta(C) / partial theta - partial A^theta(R) / partial theta)
        dot (partial f_A^theta / partial A^theta(R))
    +
    (A^theta(C) - A^theta(R))
        dot (partial^2 f_A^theta / partial A^theta(R) partial theta)
]
```

从数学上看，`S(theta)` 是 `Attr_current(A; theta)` 对模型参数 `theta` 的梯度或方向导数形式。实现时不能显式构造完整 Hessian；必须优先使用 autograd 的 Hessian-vector product / vector-Jacobian product 来计算与 `Delta theta` 相关的收缩项。

5. 计算参数差值：

```text
Delta_theta = theta' - theta
```

6. 根据超参数 `K` 构造理想未来参数：

```text
theta_hat^I = theta + K * Delta_theta
```

7. 加载或临时应用 `theta_hat^I`，并按照同样公式计算：

```text
S(theta_hat^I)
```

8. 最终 future localization attribution 使用梯形积分形式：

```text
Attr_future(A) = Attr_current(A; theta)
    + 0.5 * [S(theta) + S(theta_hat^I)] dot [K * Delta_theta]
```

如果中间推导中把 `Delta_A` 写成最终公式第一项，工程实现时应将其解释为已经与当前局部梯度收缩后的 `Attr_current(A; theta)` scalar；否则 `Delta_A` 仍是 activation tensor，无法与参数方向导数项直接相加。

Future localization 的输出仍然必须是每个 component 一个 scalar score。得到 `Attr_future(A)` 后，后续 normalized score、rank allocation、component mask、LoRA rank pattern 的流程与 current localization 完全一致。

#### 13.3.5 Future localization 输入与计算要求

新增输入参数：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `--localization_mode` | str | `current` 或 `future`，默认 `current`。 |
| `--future_model_name_or_path` | str | `theta'` 的模型目录或 HF 名称。可以是完整 finetuned checkpoint。 |
| `--future_model_cache_dir` | str | 可选，`theta'` 的 cache dir；默认沿用 `--cache_dir`。 |
| `--future_step_k` | float | 公式中的 `K`，默认建议 `1.0`。 |
| `--future_delta_parameter_filter` | str | 可选，只对匹配的参数计算 `Delta_theta`，默认覆盖所有与目标 component 相关的 Linear 权重。 |
| `--future_hvp_strategy` | str | `hvp` 或 `finite_difference`。默认 `hvp`；`finite_difference` 只用于 debug。 |

`theta'` 的加载规则：

- 如果 `theta'` 是完整模型 checkpoint，直接用 `AutoModelForCausalLM.from_pretrained(...)` 加载。
- 如果 `theta'` 是 LoRA adapter checkpoint，必须先和基础模型 merge，或者在文档/CLI 中明确要求用户传入已 merge 的完整模型目录。
- `theta` 与 `theta'` 的 tokenizer vocab size、hidden size、layer 数、attention head 配置必须一致。
- 对每个用于 `Delta_theta` 的参数，必须检查 `name` 和 `shape` 一致；不一致时应报错而不是静默跳过。

计算实现建议：

- 第一版不建议同时常驻两个 7B 模型和一个 `theta_hat^I` 模型在 GPU 上。可以将 `theta'` state dict 放在 CPU，按目标参数计算 `Delta_theta`，并在需要计算 `theta_hat^I` 时临时加载/插值。
- `theta_hat^I` 不需要默认保存到磁盘；只有设置 debug 参数时才输出。
- `S(theta) dot Delta_theta` 和 `S(theta_hat^I) dot Delta_theta` 应作为方向导数直接计算，避免保存每个参数同形状的完整 `S(theta)` 张量。
- 对 `projection_matrix` 和 `head` 两种 attention 粒度都必须支持 future localization；差异仍只体现在 component slice 的选择上。
- 对 `all_active` 和 `label_position` 两种 token 模式都必须支持 future localization；token 选择规则不能因为 future 模式改变。

Future localization 的计算量显著高于 current localization。实现时需要在 `summary.json` 中记录：

```json
{
    "localization_mode": "future",
    "future_model_name_or_path": "...",
    "future_step_k": 1.0,
    "future_hvp_strategy": "hvp",
    "delta_activation_convention": "clean_minus_corrupted",
    "gradient_source": "corrupted_activation",
    "future_uses_directional_derivative": true
}
```

### 13.4 数据集需求

数据集仍然暂时支持并复制三份 pair CSV：

```text
EAP_forComponent/data/bool.csv
EAP_forComponent/data/gender.csv
EAP_forComponent/data/ioi_mistral.csv
```

第一版可以从现有路径复制：

```text
EAP-IG/bool.csv
EAP-IG/gender.csv
EAP-IG/ioi_mistral.csv
```

字段要求与 EAP_forNeuron 一致：

```text
clean,corrupted,correct_idx,incorrect_idx
```

`ioi_mistral.csv` 可继续支持 `corrupted_hard`，通过 `--corruption_column corrupted|corrupted_hard` 选择。bool 数据仍然需要注意：如果 CSV 里保存的是原始表达式而不是完整 Mistral instruction prompt，就要复用 EAP_forNeuron 中 bool prompt 的格式化逻辑。

### 13.5 输出文件

输出目录建议为：

```text
output_dir/
    component_scores.json
    component_scores.pt
    rank_pattern.json
    lora_allocation.json
    component_mask.pt
    summary.json
```

文件说明：

- `component_scores.json`：可读 JSON，按 score 降序列出每个 component 的 raw score、normalized score、rank score、层号、模块类型、参数名、shape、`attention_granularity`、`score_token_mode`、`localization_mode`。head 粒度还要包含 `head_idx`、`head_kind` 和 slice 信息；future localization 还要记录 `future_step_k` 和用于方向导数的 score 分量。
- `component_scores.pt`：PyTorch 格式，保存完整 tensor / dict 结果，便于后续脚本读取。
- `rank_pattern.json`：用于 PEFT LoRA 的 per-module rank 配置，key 为 module name，不带 `.weight` 后缀，例如 `model.layers.0.self_attn.q_proj`。如果 attribution 使用 head 粒度，默认需要先把 head score 聚合回矩阵级 rank。
- `lora_allocation.json`：保存 rank 分配的完整元数据，包括 `min_rank/max_rank/rank_budget/rank_multiple/rank_score_source/attention_granularity/score_token_mode/head_to_matrix_aggregation` 等。
- `component_mask.pt`：现有 finetuning mask 格式 `{parameter_name: BoolTensor(shape == parameter.shape)}`，用于非 LoRA 对照实验。head 粒度下仍然输出完整参数矩阵同形状 mask。
- `summary.json`：保存模型、数据集、样本数、score 归一化方式、rank 分配方式、mask 生成方式、attention 粒度、token 模式、localization 模式、future model 路径、`K`、HVP 策略和运行耗时。

### 13.6 LoRA rank 分配规则

EAP_forComponent 的核心输出之一是 LoRA rank allocation。Rank 分配必须明确使用哪一个 score view：`attention_granularity`、`score_token_mode`、`score_normalization` 和 `rank_score_source` 都要记录到输出元数据中。

需要特别注意：标准 PEFT `LoraConfig(rank_pattern=...)` 的粒度是 `nn.Linear` module，不是同一个 Linear 里的 head slice。因此：

- `projection_matrix` score 可以直接生成 PEFT rank pattern。
- `head` score 不能直接表达为标准 PEFT per-head rank。第一版必须提供一种 `head_to_matrix_aggregation`，例如 `mean/max/sum`，把同一矩阵内的 head score 聚合回矩阵级 rank pattern。
- 如果后续实现自定义 head-wise LoRA，可以额外输出 `head_rank_allocation.json`，但这不是标准 PEFT rank pattern。

设计类：

```python
class LoraRankAllocator:
    def __init__(self, min_rank: int, max_rank: int, rank_multiple: int = 1, rank_budget: int | None = None, head_to_matrix_aggregation: str = "mean")
    def allocate(self, component_scores: list[ComponentScore]) -> dict[str, int]
```

支持两种 rank 分配模式：

#### 13.6.1 无总 rank budget

如果没有设置 `rank_budget`，对 score 做 min-max normalization：

```text
score_norm_i = (rank_score_i - min_score) / (max_score - min_score)
rank_i = round_to_multiple(min_rank + score_norm_i * (max_rank - min_rank), rank_multiple)
```

其中 score 最高的矩阵得到 `max_rank`，score 最低的矩阵得到 `min_rank`。如果 `min_rank=0`，rank 为 0 的矩阵不加入 LoRA target module；如果 PEFT 不支持 rank 0，则实现时应从 `target_modules` 或 `rank_pattern` 中删除该 module。

#### 13.6.2 有总 rank budget

如果设置 `rank_budget`，则在所有 component 之间分配一个总 rank 数量。推荐使用 water-filling / clipping：

1. 先按 `rank_score` 比例分配连续值。
2. 对每个 component 应用 `min_rank/max_rank` 约束。
3. 按 `rank_multiple` 取整。
4. 如果取整后超出 budget，从低 score component 开始减少；如果低于 budget，从高 score component 开始增加。

输出必须保证：

```text
min_rank <= rank_i <= max_rank
sum(rank_i) <= rank_budget   # 如果设置 budget
rank_i % rank_multiple == 0  # 如果 rank_multiple > 1
```

### 13.7 LoRA finetuning 集成需求

当前 `src/model/finetuning.py` 中 LoRA 配置是固定的：

```python
LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    ...
)
```

为了使用 EAP_forComponent 的 rank allocation，需要后续修改 finetuning 入口，支持 per-module rank：

#### 13.7.1 `src/exec/finetuning_model.py` 新增参数

建议新增：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `lora_rank_pattern_path` | str | EAP_forComponent 输出的 `rank_pattern.json` 路径。 |
| `lora_alpha_pattern_path` | str | 可选，per-module alpha 配置路径。 |
| `lora_default_rank` | int | 没有 pattern 时的默认 rank。 |
| `lora_target_modules` | str | 默认 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。 |
| `lora_alpha_strategy` | str | `constant` 或 `twice_rank`。 |

#### 13.7.2 `src/model/finetuning.py` 修改 LoRA 初始化

建议改为：

```python
rank_pattern = load_rank_pattern(lora_rank_pattern_path)
lora_config = LoraConfig(
    r=lora_default_rank,
    lora_alpha=32,
    target_modules=lora_target_modules,
    rank_pattern=rank_pattern,
    alpha_pattern=alpha_pattern,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

注意：`rank_pattern.json` 的 key 应该使用 module name，例如：

```json
{
  "model.layers.0.self_attn.q_proj": 16,
  "model.layers.0.self_attn.k_proj": 8,
  "model.layers.0.mlp.down_proj": 24
}
```

如果 PEFT 版本对 exact module name / regex key 有差异，需要在实现时加一个 small smoke test：加载一个 tiny model 或 Mistral config，确认 `rank_pattern` 实际命中了目标 module。

### 13.8 Component mask 生成规则

除了 LoRA rank pattern，EAP_forComponent 还需要生成一个普通 bool mask，用于不用 LoRA 的对照实验。

输出 mask 仍然沿用当前项目格式：

```python
{
    parameter_name: torch.BoolTensor(shape == model_parameter.shape)
}
```

`projection_matrix` 粒度的生成规则：

1. 对目标矩阵按 `rank_score` 从高到低排序。
2. 假设共有 `N` 个 component。第 `i` 个 component 的 true ratio 为：

```text
component_true_ratio_i = 1 - i / (N - 1)   # i 从 0 开始
```

3. 如果 `N == 1`，该 component ratio 为 1。
4. 最高分 component 的全部参数为 `True`。
5. 最低分 component 的全部参数为 `False`。
6. 中间 component 的 true 数量为：

```text
true_count_i = int(parameter_numel_i * component_true_ratio_i)
```

7. 因为 component-level score 不提供矩阵内部每个 neuron 的重要性，矩阵内部 true 位置不应伪装成 attribution 结果。默认建议使用固定 seed 的随机排列选择 true 位置：

```python
indices = torch.randperm(parameter_numel, generator=generator)[:true_count]
mask.flatten()[indices] = True
```

8. 可选提供 `--mask_fill_strategy magnitude|random|first`：
   - `random`：默认，避免引入额外启发式。
   - `magnitude`：按参数绝对值选择，用于探索性实验，但要在 summary 中明确记录。
   - `first`：只用于 deterministic debug，不建议正式实验。

`head` 粒度的生成规则类似，但排序单位变为 head component：

- `q_proj/k_proj/v_proj`：按 output row slice 写入 mask，即只在对应 head rows 内分配 true/false。
- `o_proj`：按 input column slice 写入 mask，即只在对应 head cols 内分配 true/false。
- `gate_proj/up_proj/down_proj`：仍按完整矩阵写入 mask。

这个 mask 只是 LoRA rank allocation 的对照实验，不代表矩阵内部 neuron 的 EAP attribution。head 粒度下，mask 表示“这个 head slice 所属参数区域按 component score 得到某个保留比例”，并不表示 slice 内每个参数都有单独 attribution score。

### 13.9 核心类设计

建议新增目录：

```text
EAP_forComponent/
    README.md
    __init__.py
    run_eap_for_component.py
    cli.py
    schemas.py
    data.py
    model_loader.py
    components.py
    hooks.py
    scorer.py
    rank_allocator.py
    mask_builder.py
    outputs.py
    runner.py
    data/
        bool.csv
        gender.csv
        ioi_mistral.csv
    tests/
```

第一版可以复用 EAP_forNeuron 的 `data.py` 和 `model_loader.py` 逻辑；如果复制到 `EAP_forComponent`，必须保持 cache_dir、Mistral RoPE patch、Blackwell bf16、`device_map={"": 0}` 和 tokenizer pad token 处理一致。

#### 13.9.1 `ComponentTarget`

文件：`EAP_forComponent/schemas.py`

```python
@dataclass
class ComponentTarget:
    parameter_name: str
    module_name: str
    layer_idx: int
    component_type: str  # q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj
    granularity: str  # projection_matrix/head
    head_idx: int | None
    head_kind: str | None  # query/key_value/output_input_slice
    row_slice: tuple[int, int] | None
    col_slice: tuple[int, int] | None
    module: nn.Linear
    shape: torch.Size
    numel: int
```

#### 13.9.2 `ComponentScore`

文件：`EAP_forComponent/schemas.py`

```python
@dataclass
class ComponentScore:
    parameter_name: str
    module_name: str
    layer_idx: int
    component_type: str
    granularity: str
    score_token_mode: str
    localization_mode: str
    head_idx: int | None
    head_kind: str | None
    raw_score: float
    abs_score: float
    mean_score: float
    sqrt_numel_score: float
    rank_score: float
    current_score: float | None
    future_directional_score_theta: float | None
    future_directional_score_theta_hat: float | None
    future_correction: float | None
    shape: tuple[int, ...]
    numel: int
```

#### 13.9.3 `ComponentRegistry`

文件：`EAP_forComponent/components.py`

职责：从 HF model 中枚举目标 component。

```python
class ComponentRegistry:
    @classmethod
    def from_model(cls, model, target_modules: list[str], attention_granularity: str) -> "ComponentRegistry"
    def targets(self) -> list[ComponentTarget]
    def parameter_names(self) -> list[str]
```

枚举规则：

- 只接受 `nn.Linear`。
- 参数名必须以七类目标后缀之一结尾。
- `attention_granularity="projection_matrix"` 时，每个目标矩阵生成一个 `ComponentTarget`。
- `attention_granularity="head"` 时，attention projection 按 head slice 生成多个 `ComponentTarget`，MLP projection 仍生成一个矩阵级 `ComponentTarget`。
- 对 GQA 模型，`q_proj/o_proj` 使用 `config.num_attention_heads`，`k_proj/v_proj` 使用 `config.num_key_value_heads`。
- 默认不包含 `lm_head`、embedding、LayerNorm、bias、LoRA adapter 参数。

#### 13.9.4 `ComponentActivationCache`

文件：`EAP_forComponent/hooks.py`

职责：捕获目标 Linear activation 和 output gradient。

```python
class ComponentActivationCache:
    def __init__(self, targets: list[ComponentTarget], capture_device="cpu")
    def register(self)
    def remove(self)
    def clear_batch(self)
    def capture(self, mode: Literal["clean", "corrupted"])
```

需要保存：

```python
clean_outputs[parameter_name]
corrupted_outputs[parameter_name]
output_grads[parameter_name]
clean_inputs[parameter_name]      # head 粒度下 o_proj 需要
corrupted_inputs[parameter_name]  # head 粒度下 o_proj 需要
```

与 EAP_forNeuron 不同，这里默认 hook 的是 Linear output；但为了支持 `o_proj` head 粒度，也要能保存 Linear input。

#### 13.9.5 `ComponentAttributionScorer`

文件：`EAP_forComponent/scorer.py`

```python
class ComponentAttributionScorer:
    def __init__(self, targets, score_token_mode="all_active", score_normalization="mean", localization_mode="current")
    def score_batch(clean_outputs, corrupted_outputs, output_grads, attention_mask, label_positions)
    def finalize() -> list[ComponentScore]
```

核心函数：

```python
def score_component_output(clean_output, corrupted_output, output_grad, attention_mask, label_positions, token_mode):
    delta = clean_output - corrupted_output
    return (delta * output_grad).sum()

def score_o_proj_head_input_slice(clean_input, corrupted_input, weight, output_grad, col_slice, attention_mask, label_positions, token_mode):
    clean_contrib = clean_input[..., col_slice] @ weight[:, col_slice].T
    corrupted_contrib = corrupted_input[..., col_slice] @ weight[:, col_slice].T
    return ((clean_contrib - corrupted_contrib) * output_grad).sum()
```

当 `localization_mode="future"` 时，`ComponentAttributionScorer` 不应只依赖缓存的 activation/output gradient。建议新增 `FutureLocalizationScorer` 或在 scorer 内拆出独立分支，负责：

```python
class FutureLocalizationScorer:
    def __init__(self, base_model, future_model_path, targets, future_step_k, hvp_strategy="hvp", delta_parameter_filter=None)
    def compute_delta_theta(self) -> dict[str, Tensor | LazyTensor]
    def build_theta_hat(self)
    def directional_score(self, theta_state, delta_theta, batch, target) -> float
    def score_batch(...) -> dict[str, float]
```

其中 `directional_score(...)` 需要直接计算 `S(theta) dot Delta_theta`，不要 materialize 完整 Hessian。Current localization 可以继续沿用已有 activation cache；future localization 可以复用 hook 捕获 activation，但必须打开 `create_graph=True` 以支持二阶导或 HVP。

#### 13.9.6 `LoraRankAllocator`

文件：`EAP_forComponent/rank_allocator.py`

职责：把 component score 转成 LoRA rank pattern。

```python
class LoraRankAllocator:
    def allocate(self, scores: list[ComponentScore]) -> dict[str, int]
    def to_rank_pattern(self, ranks: dict[str, int]) -> dict[str, int]
```

其中 `to_rank_pattern` 要把参数名去掉 `.weight` 后缀，得到 PEFT module key。

#### 13.9.7 `ComponentMaskBuilder`

文件：`EAP_forComponent/mask_builder.py`

职责：根据 component score 排序生成对照 mask。

```python
class ComponentMaskBuilder:
    def __init__(self, mask_fill_strategy="random", seed=0)
    def build(self, model, scores: list[ComponentScore]) -> dict[str, torch.BoolTensor]
```

只给目标参数矩阵生成 key；如果为了与现有 finetuning mask 更完全兼容，也可以给所有 `model.named_parameters()` 生成 key，非目标参数全部为 `False`。

#### 13.9.8 `EAPForComponentRunner`

文件：`EAP_forComponent/runner.py`

```python
class EAPForComponentRunner:
    def load_model_and_tokenizer(self)
    def load_dataset(self)
    def build_components(self)
    def run_attribution(self)
    def allocate_lora_ranks(self)
    def build_control_mask(self)
    def save_outputs(self)
    def run(self)
```

### 13.10 CLI 设计

建议命令：

```bash
cd /ssd_users/chenhang/CSAT
/home/chenhang/.conda/envs/LLMSFT_BW/bin/python EAP_forComponent/run_eap_for_component.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --tokenizer_name_or_path mistralai/Mistral-7B-v0.1 \
  --cache_dir /home/chenhang/CSAT/.cache \
  --dataset_name ioi_mistral \
  --data_path EAP_forComponent/data/ioi_mistral.csv \
  --output_dir files/component_scores/ioi_mistral \
    --localization_mode current \
  --score_token_mode all_active \
    --attention_granularity projection_matrix \
  --score_normalization mean \
  --min_rank 0 \
  --max_rank 32 \
  --rank_multiple 1 \
  --batch_size 1 \
  --max_samples 128
```

建议参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_name_or_path` | `mistralai/Mistral-7B-v0.1` | 模型路径或 HF 名称。 |
| `--cache_dir` | `/home/chenhang/CSAT/.cache` | 与 finetuning_model.py 保持一致，避免重复下载。 |
| `--dataset_name` | `ioi_mistral` | `bool/gender/ioi_mistral`。 |
| `--data_path` | 自动推断 | pair CSV 路径。 |
| `--target_modules` | 七类 projection | 逗号分隔目标矩阵类型。 |
| `--attention_granularity` | `projection_matrix` | `projection_matrix` 或 `head`。两种都必须实现。 |
| `--localization_mode` | `current` | `current` 或 `future`。控制 attribution score 的计算公式。 |
| `--future_model_name_or_path` | None | future localization 中 `theta'` 的模型目录或 HF 名称。`localization_mode=future` 时必填。 |
| `--future_model_cache_dir` | None | `theta'` 的 cache dir，默认沿用 `--cache_dir`。 |
| `--future_step_k` | 1.0 | future localization 公式中的 `K`。 |
| `--future_delta_parameter_filter` | None | 可选参数名过滤器，用于限制参与 `Delta_theta` 的参数。 |
| `--future_hvp_strategy` | `hvp` | `hvp` 或 `finite_difference`，默认用 autograd HVP。 |
| `--score_token_mode` | `all_active` | `all_active` 或 `label_position`。 |
| `--score_normalization` | `mean` | rank 分配使用的 score 归一方式。 |
| `--rank_score_source` | `normalized_abs` | 用 raw abs 还是 normalized abs 分配 rank。 |
| `--head_to_matrix_aggregation` | `mean` | head 粒度 score 生成标准 PEFT rank pattern 时，如何聚合同一矩阵内的 head score。 |
| `--min_rank` | 0 | 最低 LoRA rank。 |
| `--max_rank` | 32 | 最高 LoRA rank。 |
| `--rank_budget` | None | 可选总 rank budget。 |
| `--rank_multiple` | 1 | rank 取整倍数。 |
| `--mask_fill_strategy` | `random` | 对照 mask 的矩阵内部 true 位置选择方式。 |
| `--mask_seed` | 0 | 对照 mask 随机种子。 |
| `--max_samples` | 128 | attribution 样本数。 |
| `--batch_size` | 1 | batch size。 |

### 13.11 与现有代码的新增、修改、删除

#### 13.11.1 新增

必须新增：

- `EAP_forComponent/` 文件夹。
- `EAP_forComponent/run_eap_for_component.py`：命令行入口。
- `EAP_forComponent/cli.py`：参数解析。
- `EAP_forComponent/schemas.py`：`ComponentTarget`、`ComponentScore`、config dataclass；字段必须支持矩阵级和 head 级 component。
- `EAP_forComponent/data.py`：加载 `bool/gender/ioi_mistral` pair CSV；可以复用或复制 EAP_forNeuron 数据逻辑。
- `EAP_forComponent/model_loader.py`：加载 HF Mistral，cache_dir 默认 `/home/chenhang/CSAT/.cache`。
- `EAP_forComponent/components.py`：枚举 q/k/v/o/gate/up/down 矩阵，并在 `attention_granularity="head"` 时生成 head slice component。
- `EAP_forComponent/hooks.py`：捕获 output activation、output gradient；为 `o_proj` head 粒度额外捕获 input activation。
- `EAP_forComponent/scorer.py`：计算矩阵级和 head 级 component attribution score，并支持 `all_active/label_position` 两种 token 模式。
- `EAP_forComponent/future_localization.py`：新增 future localization 方向导数 / HVP 计算逻辑，读取 `theta'`，构造 `Delta_theta` 和 `theta_hat^I`，输出与 current scorer 相同结构的 component scalar score。
- `EAP_forComponent/rank_allocator.py`：生成 LoRA rank pattern；head 粒度下要支持 head score 聚合回矩阵级 rank pattern。
- `EAP_forComponent/mask_builder.py`：生成 component-level 对照 mask；head 粒度下按 row/column slice 写入完整参数矩阵 mask。
- `EAP_forComponent/outputs.py`：保存 `component_scores.json/pt`、`rank_pattern.json`、`component_mask.pt`、`summary.json`。
- `EAP_forComponent/data/bool.csv`、`gender.csv`、`ioi_mistral.csv`。
- `EAP_forComponent/tests/`：tiny Linear scorer、rank allocator、mask builder、dataset smoke test。

#### 13.11.2 修改

建议后续修改：

- `src/exec/finetuning_model.py`：新增 LoRA rank pattern 相关参数。
- `src/model/finetuning.py`：当 `use_lora=True` 且传入 `lora_rank_pattern_path` 时，读取 `rank_pattern.json` 并传给 `LoraConfig(rank_pattern=...)`。
- `src/model/finetuning.py`：LoRA `target_modules` 从当前只含 `q_proj/v_proj` 扩展为可配置，默认支持 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。
- 如果要真正使用 head 级不同 rank，而不是聚合回矩阵级 rank，需要新增自定义 head-wise LoRA 模块；标准 PEFT `rank_pattern` 本身不能表达同一 Linear 内不同 head 的不同 rank。
- `enviroment.md`：如果 PEFT 版本确认支持 `rank_pattern/alpha_pattern`，补充说明；如果当前版本不支持，需要记录升级要求或 fallback 实现。
- `README.md` 或 `current_functionality_requirements.md`：补充 EAP_forComponent 的运行命令和输出文件说明。

可选修改：

- 如果 EAP_forNeuron 和 EAP_forComponent 共享大量 data/model loader 逻辑，可以新增公共包 `EAP_common/`，避免复制三份数据处理代码。但第一版为了降低改动范围，可以先各自独立。

#### 13.11.3 删除

本阶段不删除现有代码。尤其不要删除：

- `EAP_forNeuron/`：它仍可用于小 mask 或 tiny 实验。
- `src/unlearn/generate_mask.py`：gradient mask 生成功能仍是已有训练路径的一部分。
- `src/finetuning/base.py::mask_gradient(...)`：component mask 对照实验仍依赖该格式。
- `EAP-IG/` 原始参考实现。

### 13.12 验证标准

实现完成后至少需要以下验证：

1. Tiny Linear component scorer：手工计算 `sum((y_clean - y_corr) * grad_y_corr)`，与 `ComponentAttributionScorer` 的 `projection_matrix` 输出一致，并在 summary 中记录 `delta_activation_convention="clean_minus_corrupted"`。
2. Head slice scorer：对 tiny multi-head Linear，验证 q/k/v 的 row slice score 和 o_proj 的 input column slice contribution score 与手工计算一致。
3. Score token mode 测试：同一 batch 下分别验证 `all_active` 使用所有有效 token，`label_position` 只使用每个样本最后 label token。
4. Component registry 测试：对 tiny Mistral-like module，只枚举 q/k/v/o/gate/up/down，不包含 embedding、lm_head、norm、bias；`projection_matrix/head` 两种粒度的 component 数量符合预期。
5. GQA head mapping 测试：当 `num_attention_heads != num_key_value_heads` 时，q/o 使用 query head 数，k/v 使用 KV head 数，并在 summary 中正确记录。
6. Dataset smoke test：`bool/gender/ioi_mistral` 都能读取 clean/corrupted batch。
7. Rank allocator 测试：score 高的 component rank 不低于 score 低的 component；满足 `min_rank/max_rank/rank_multiple/rank_budget`；head 粒度下能按 `head_to_matrix_aggregation` 聚合为矩阵级 rank pattern。
8. Mask builder 测试：最高 score component 全 True，最低 score component 全 False，中间 component true ratio 按排序线性递减；head 粒度下只写对应 row/column slice。
9. 输出兼容测试：`component_mask.pt` 能被当前 finetuning mask 逻辑加载，key 和 parameter shape 对齐。
10. PEFT smoke test：矩阵级或聚合后的 `rank_pattern.json` 能被当前 PEFT `LoraConfig` 接收，并实际命中目标 module。
11. Mistral dry run：`--max_samples 1 --batch_size 1` 跑通模型加载、hook、score、rank pattern、mask 保存。
12. Localization mode smoke test：同一 tiny Linear 模型下，`--localization_mode current` 与 13.3 中的 `current_localization` 手工公式输出一致；默认参数不变时 current 模式可复现。
13. Future model 对齐测试：当 `theta'` 与 `theta` 参数名或 shape 不一致时，必须报错；一致时能正确计算 `Delta_theta`。
14. Future localization HVP 测试：对 tiny 二层模型，用 autograd HVP 计算的 `S(theta) dot Delta_theta` 与有限差分近似方向导数在容差内一致。
15. Future localization 端到端测试：`--localization_mode future --future_model_name_or_path <tiny_finetuned_model> --future_step_k 1.0 --max_samples 1` 能输出 `component_scores.json/rank_pattern.json/summary.json`，且 summary 记录 future 相关字段。

### 13.13 风险和注意事项

- raw score 会受到矩阵输出维度、token 数和参数规模影响；必须保存 raw 和 normalized score，并明确 rank 分配使用哪一个。
- `k_proj/v_proj` 在 GQA 模型中输出维度可能小于 `q_proj/o_proj`，如果使用 raw sum，它们天然更容易得低分；这可能是合理的总体影响，也可能只是维度偏置。
- head 粒度下，`q_proj/o_proj` 的 head 数和 `k_proj/v_proj` 的 KV head 数可能不同，不能假设 q/k/v/o 的 `head_idx` 一一对应。
- 标准 PEFT LoRA 不能直接给同一个 Linear 内的不同 head 设置不同 rank；head score 若用于 PEFT，必须先聚合回矩阵级 rank，或者另行实现自定义 head-wise LoRA。
- `all_active` 更接近 EAP-IG，但会把 prompt 中所有有效 token 的贡献都聚合进去；`label_position` 更贴近当前最后答案 token 评估语义，但可能漏掉前文 token 上的间接 causal contribution。
- `gate_proj/up_proj/down_proj` 的梯度已经包含 MLP 非线性和门控乘法的后续影响，不需要手写 MLP 公式。
- LoRA rank 为 0 的 module 是否能直接写入 PEFT `rank_pattern` 取决于 PEFT 版本；不支持时应从 LoRA target 中删除该 module。
- component mask 内部的 true/false 位置不是 EAP 计算出来的 neuron 重要性，只是 component-level score 的比例投影；正式汇报时必须把它标注为 control mask。
- 与 EAP_forNeuron 相比，EAP_forComponent 的速度瓶颈应该主要是两次 forward 和一次 backward，而不再是十亿级候选参数打分；如果仍然很慢，应优先检查 hook 是否保存了过多 batch activation 或是否把大 tensor 反复搬到 CPU。
- Future localization 需要二阶信息或 HVP，计算量和显存压力会明显高于 current localization；严禁显式构造完整 Hessian。
- Future localization 需要同时处理 `theta`、`theta'` 和 `theta_hat^I`。Mistral-7B 下不能默认把三份完整模型都常驻 GPU；应优先使用 CPU state dict、lazy delta、分层/分模块计算或临时插值。
- `K` 会放大 `Delta_theta`。如果 `K` 过大，`theta_hat^I` 可能远离模型原训练邻域，activation/gradient 可能不稳定；summary 必须记录 `K`，正式实验建议先扫描小范围 `K`。
- 如果 `theta'` 来自 LoRA finetuning，必须明确使用 merged full checkpoint 还是 adapter checkpoint。Future localization 的 `Delta_theta` 必须和基础模型参数同名同 shape，否则方向导数没有明确含义。

