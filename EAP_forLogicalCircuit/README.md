# EAP_forLogicalCircuit

`EAP_forLogicalCircuit` computes edge-level attribution and builds logical circuits from pair CSV datasets.

## Supported Data

When `--data_path` is omitted, the loader selects a packaged CSV for the requested dataset. Supported dataset names include:

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

## Optional Circuit Graph Export

Pass `--graph` or `--graph true` to additionally export an EAP-style graph for the forward `circuit` only. The export uses the top `--graph_node_topn` ranked nodes, then first keeps the highest-attribution incoming and outgoing edge for every selected nonterminal node. `input` is exempt from the incoming-edge requirement and `logits` is exempt from the outgoing-edge requirement. After those required connectivity edges, remaining edges are filled by attribution score up to `ceil(--graph_edge_budget_multiplier * graph_node_count)`, default `3.0`. Final exports also soft-cap `input` outgoing edges to `floor(--graph_input_edge_limit_ratio * graph_node_count)`, default `0.3`, while preserving any input edges required to keep selected nodes connected; remaining input edges are kept by attribution score.

When enabled, the output directory contains `graph.json`, `graph_edges.json`, `graph_summary.json`, and `graph.dot`. If the Graphviz `dot` executable is available, it also renders `graph.jpg`; otherwise `graph_summary.json` records that image rendering was skipped and `graph.dot` can be rendered later. The rendered graph uses a top-to-bottom DOT layout, keeps the original `input` and `logits` node names, uses uniform edge width, and applies q/k/v/sign edge colors. Final graph exports deduplicate by rendered source/destination node pair, so any two displayed nodes have at most one edge between them; for q/k/v alternatives, the highest-attribution edge is kept.

For `node_induced` circuit construction, graph edge attribution is computed only for the top-node-induced candidate edges used by the graph export, instead of rescoring every edge in the full dense residual graph.