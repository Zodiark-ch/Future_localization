"""
将 Graph 导出的 .pt 文件转为 JSON：
- graph.json: edges_in_graph 为 True 的边，形如 [["src", "dst"], ...]
- score.json: 包含 src_nodes、dst_nodes、edges_scores（二维浮点列表）

导出时对节点名做统一替换（与 .pt 内原始名对应，仅影响 JSON）：
- src: input -> embeds
- dst: logits -> resid_post
- dst: a0.h31<q> -> a0.h31.q（<k>/<v> 同理）
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def transform_src_node(name: str) -> str:
    """导出 JSON 时统一 src 节点命名。"""
    if name == "input":
        return "embeds"
    return name


def transform_dst_node(name: str) -> str:
    """导出 JSON 时统一 dst 节点命名：logits -> resid_post；a0.h31<q> -> a0.h31.q"""
    if name == "logits":
        return "resid_post"
    # 将 <q>/<k>/<v> 转为 .q / .k / .v
    return re.sub(r"<(q|k|v)>", r".\1", name)


def tensor_to_nested_list(t: torch.Tensor) -> list:
    """将 2D tensor 转为嵌套 list，便于 JSON 序列化。"""
    if t.dim() != 2:
        raise ValueError(f"edges_scores 期望 2D tensor，当前 shape={tuple(t.shape)}")
    return t.detach().cpu().float().tolist()


def build_edges_in_graph(
    src_nodes: list[str],
    dst_nodes: list[str],
    edges_in_graph: torch.Tensor,
) -> list[list[str]]:
    """找出 edges_in_graph 为 True 的边，返回 [[src, dst], ...]。"""
    mask = edges_in_graph.detach().cpu()
    if mask.dim() != 2:
        raise ValueError(f"edges_in_graph 期望 2D tensor，当前 shape={tuple(mask.shape)}")
    n_src, n_dst = mask.shape
    if len(src_nodes) != n_src or len(dst_nodes) != n_dst:
        raise ValueError(
            f"维度不匹配: src_nodes={len(src_nodes)}, dst_nodes={len(dst_nodes)}, "
            f"mask={tuple(mask.shape)}"
        )
    edges: list[list[str]] = []
    for i in range(n_src):
        for j in range(n_dst):
            if bool(mask[i, j].item()):
                edges.append([src_nodes[i], dst_nodes[j]])
    return edges


def pt_to_json(
    pt_path: str | Path,
    graph_json_path: str | Path | None = None,
    score_json_path: str | Path | None = None,
) -> None:
    pt_path = Path(pt_path)
    out_dir = pt_path.parent
    if graph_json_path is None:
        graph_json_path = out_dir / "graph.json"
    if score_json_path is None:
        score_json_path = out_dir / "score.json"

    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(pt_path, map_location="cpu")

    required = ("src_nodes", "dst_nodes", "edges_scores", "edges_in_graph")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f".pt 缺少键: {missing}，实际 keys={list(data.keys())}")

    src_nodes_raw = list(data["src_nodes"])
    dst_nodes_raw = list(data["dst_nodes"])
    src_nodes = [transform_src_node(n) for n in src_nodes_raw]
    dst_nodes = [transform_dst_node(n) for n in dst_nodes_raw]
    edges_scores = data["edges_scores"]
    edges_in_graph = data["edges_in_graph"]

    if not isinstance(edges_scores, torch.Tensor):
        edges_scores = torch.as_tensor(edges_scores)
    if not isinstance(edges_in_graph, torch.Tensor):
        edges_in_graph = torch.as_tensor(edges_in_graph, dtype=torch.bool)

    graph_edges = build_edges_in_graph(src_nodes, dst_nodes, edges_in_graph)

    score_payload = {
        "src_nodes": src_nodes,
        "dst_nodes": dst_nodes,
        "edges_scores": tensor_to_nested_list(edges_scores),
    }

    graph_json_path = Path(graph_json_path)
    score_json_path = Path(score_json_path)
    graph_json_path.parent.mkdir(parents=True, exist_ok=True)
    score_json_path.parent.mkdir(parents=True, exist_ok=True)

    with graph_json_path.open("w", encoding="utf-8") as f:
        json.dump(graph_edges, f, indent=4, ensure_ascii=False)

    with score_json_path.open("w", encoding="utf-8") as f:
        json.dump(score_payload, f, indent=2, ensure_ascii=False)

    print(f"已写入 {len(graph_edges)} 条 in-graph 边 -> {graph_json_path}")
    print(f"已写入 score（src/dst + 全矩阵）-> {score_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 ioi_graph.pt 等转为 graph.json 与 score.json")
    parser.add_argument(
        "pt_path",
        nargs="?",
        default="ioi_localization2w_graph_epoch1.pt",
        help="输入的 .pt 文件路径（默认: ioi_graph.pt）",
    )
    parser.add_argument(
        "--graph-out",
        default="ioi_localization2w_graph_epoch_1.json",
        help="ioi_graph_epoch0.json 输出路径（默认与 .pt 同目录下的 graph.json）",
    )
    parser.add_argument(
        "--score-out",
        default="ioi_localization2w_score_epoch_1.json",
        help="ioi_score_epoch0.json 输出路径（默认与 .pt 同目录下的 score.json）",
    )
    args = parser.parse_args()
    pt_to_json(args.pt_path, args.graph_out, args.score_out)


if __name__ == "__main__":
    main()
