"""
将单个 circuit 的 JSON 文件转换为只包含节点名的 list JSON。

每条 circuit 边形如 ["a19.h16.o", "a30.h2.q", "ADDER"]，取第 0、1 个元素为节点，
合并后去重（保持首次出现顺序），保存为 JSON 数组。
"""

import argparse
import json
import os

# --- 默认路径（可直接改下面两行；不传 -i / -o 时使用）---
_EDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CIRCUIT_JSON = os.path.join(_EDGE_ROOT, "data", "/home/lthpc/hangc/EAP-IG/ioi_graph_epoch0_edges_bool.json")
DEFAULT_OUTPUT_JSON = os.path.join(_EDGE_ROOT, "data", "masks", "ioi_graph_epoch0_edges_bool_nodes_only.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 circuit JSON 提取节点列表并保存为 JSON 数组"
    )
    parser.add_argument(
        "-i",
        "--circuit",
        type=str,
        default=DEFAULT_CIRCUIT_JSON,
        help=f"circuit JSON 文件路径（默认: {DEFAULT_CIRCUIT_JSON}）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help=f"输出 JSON 文件路径（默认: {DEFAULT_OUTPUT_JSON}）",
    )
    return parser.parse_args()


def circuit_to_node_list(data):
    """
    从 circuit 结构提取所有第 0、1 个元素，去重并保持顺序。
    """
    if not isinstance(data, list):
        raise ValueError("circuit JSON 顶层应为数组")

    seen = {}
    for item in data:
        if not isinstance(item, (list, tuple)):
            continue
        for idx in (0, 1):
            if idx < len(item):
                node = item[idx]
                if isinstance(node, str) and node not in seen:
                    seen[node] = None
    return list(seen.keys())


def main():
    args = parse_args()
    circuit_path = os.path.abspath(args.circuit)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(circuit_path):
        raise FileNotFoundError(f"找不到 circuit 文件: {circuit_path}")

    with open(circuit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = circuit_to_node_list(data)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"共提取 {len(nodes)} 个不重复节点，已写入: {output_path}")


if __name__ == "__main__":
    main()
