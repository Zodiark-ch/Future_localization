import json
import os
import argparse
import graphviz

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--in_path", "-i", type=str, default='/data/zodiark/CSAT/files/runs/mistral_gp/edges.json')
    parser.add_argument("--out_path", "-o", type=str, default=None)
    parser.add_argument("--constants", "-c", default=[], nargs="+")
    parser.add_argument("--no-sanitize", "-ns", action="store_true")
    
    args = parser.parse_args()
    
    if args.out_path is None:
        assert "edges.json" in args.in_path, "Please provide the out_path"
        args.out_path = args.in_path.replace("edges.json", "edges.pdf")
    args.constants = [c for c in args.constants if c.lower() != "[none]"]
    return args

def get_constant_color(color="aquamarine1"):
    return lambda x: color

def get_circuit_colors_v1(
    embeds_color="azure", # azure2
    mlp_color="cadetblue2", # cornflowerblue
    q_color="plum1", # orchid3
    k_color="lightpink", # chocolate1
    v_color="khaki1", # gold1
    o_color="darkslategray3", # aquamarine1
    resid_post_color="azure" # azure2
):
    def decide_color(node_name):
        if "embed" in node_name:
            return embeds_color
        elif node_name == "resid_post":
            return resid_post_color
        elif node_name.startswith("m"):
            return mlp_color
        elif node_name.endswith(".q"):
            return q_color
        elif node_name.endswith(".k"):
            return k_color
        elif node_name.endswith(".v"):
            return v_color
        else:
            return o_color
    return decide_color

def get_circuit_colors(
    tok_embeds_color="grey", # azure2
    pos_embeds_color="lightsteelblue", # azure2
    mlp_color="cadetblue2", # cornflowerblue
    q_color="plum1", # orchid3
    k_color="lightpink", # chocolate1
    v_color="khaki1", # gold1
    o_color="darkslategray3", # aquamarine1
    resid_post_color="azure" # azure2
):
    def decide_color(node_name):
        node_name = node_name.lower()
        if "embed" in node_name:
            if "pos" in node_name:
                return pos_embeds_color
            else:
                return tok_embeds_color
        elif node_name == "output":
            return resid_post_color
        elif node_name.startswith("m"):
            return mlp_color
        elif node_name.endswith(".q"):
            return q_color
        elif node_name.endswith(".k"):
            return k_color
        elif node_name.endswith(".v"):
            return v_color
        else:
            return o_color
    return decide_color

def sanitize_edges(edges):
    # First, add all q,k,v -> o edges
    new_edges_ = set()
    for edge in edges:
        if edge[0][0] == "a" and edge[0][-1] not in ["q", "k", "v"]:
            new_edges_.add(edge[0])
    for to in new_edges_:
        for suffix in [".q", ".k", ".v"]:
            from_ = to +  suffix
            edges.append((from_, to))
    while True:
        orig_len = len(edges)
        # Find all nodes that are destinations but not sources
        froms = set()
        tos = set()
        for edge in edges:
            froms.add(edge[0])
            if edge[1] != "resid_post":
                tos.add(edge[1])
        banned_tos = tos.difference(froms)
        edges = [e for e in edges if e[1] not in banned_tos]

        # Find qkv nodes that have no incoming edges, and remove the q -> o edge for them
        qkv_nodes = set()
        for edge in edges:
            if edge[1].endswith(".q"):
                qkv_nodes.add(edge[1])
            elif edge[1].endswith(".k"):
                qkv_nodes.add(edge[1])
            elif edge[1].endswith(".v"):
                qkv_nodes.add(edge[1])

        edges = [
            e for e in edges if not (
                (e[0].endswith(".q") and e[0] not in qkv_nodes) or
                (e[0].endswith(".k") and e[0] not in qkv_nodes) or
                (e[0].endswith(".v") and e[0] not in qkv_nodes)
            )
        ]
        if orig_len == len(edges):
            break

    return edges

def rename(name):
    if type(name) != str:
        return [rename(n) for n in name]
    if "embeds" in name:
        return "Embeddings"
    if name == "resid_post":
        return "Output"
    if name.startswith("m"):
        l = int(name[1:])
        return f"MLP {l}"
    if name.endswith(".q"):
        parts = name.split(".")
        l = int(parts[0][1:])
        h = int(parts[1][1:])
        return f"Head {l}.{h}.Q"
    if name.endswith(".k"):
        parts = name.split(".")
        l = int(parts[0][1:])
        h = int(parts[1][1:])
        return f"Head {l}.{h}.K"
    if name.endswith(".v"):
        parts = name.split(".")
        l = int(parts[0][1:])
        h = int(parts[1][1:])
        return f"Head {l}.{h}.V"
    parts = name.split(".")
    assert len(parts) == 2, f"Invalid node name {name}"
    l = int(parts[0][1:])
    h = int(parts[1][1:])
    return f"Head {l}.{h}.O"

def main():
    args = parse_args()
    
    # 首先检查文件是否存在和基本信息
    if not os.path.exists(args.in_path):
        raise FileNotFoundError(f"文件不存在: {args.in_path}")
    
    file_size = os.path.getsize(args.in_path)
    print(f"文件大小: {file_size} 字节")
    
    # 检查文件的前几个字节
    try:
        with open(args.in_path, 'rb') as f:
            first_bytes = f.read(50)
            print(f"文件前50个字节: {first_bytes}")
            print(f"文件前50个字节(hex): {first_bytes.hex()}")
    except Exception as e:
        print(f"读取文件字节时出错: {e}")
    
    # 尝试不同的编码方式读取文件
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    edges = None
    
    for encoding in encodings:
        try:
            with open(args.in_path, 'r', encoding=encoding) as f:
                content = f.read()
                print(f"使用 {encoding} 编码读取成功，文件内容长度: {len(content)} 字符")
                print(f"文件内容前200字符: {repr(content[:200])}")
                
                # 尝试解析JSON
                edges = json.loads(content)
                print(f"JSON解析成功，数据类型: {type(edges)}")
                if isinstance(edges, list):
                    print(f"列表长度: {len(edges)}")
                    if len(edges) > 0:
                        print(f"第一个元素: {edges[0]}")
                break
                
        except UnicodeDecodeError as e:
            print(f"使用 {encoding} 编码失败: {e}")
            continue
        except json.JSONDecodeError as e:
            print(f"使用 {encoding} 编码读取成功，但JSON解析失败: {e}")
            print(f"错误位置附近的内容: {content[max(0, e.pos-50):e.pos+50]}")
            continue
        except Exception as e:
            print(f"使用 {encoding} 编码时发生其他错误: {e}")
            continue
    
    if edges is None:
        print("\n所有编码方式都失败了。请检查:")
        print("1. 文件是否是有效的JSON格式")
        print("2. 文件是否损坏")
        print("3. 文件是否为空")
        raise ValueError("无法使用任何编码方式读取文件")
    if not args.no_sanitize:
        edges = sanitize_edges(edges)
    
    edges = [rename(e) for e in edges]

    out_path = args.out_path
    out_path_temp = args.out_path + ".pdf"
    
    coloring_fn = get_circuit_colors()
    constant_edge_color = 'gray66'

    constant_in = args.constants

    nodes = set(constant_in + [x for y in edges for x in y])
    
    kwargs = {
        "graph_attr": {
            "nodesep": "0.02",
            "ranksep": "0.02",   
            "ratio":"1:6",
        },
        "node_attr": {
            "shape": "box",
            "style": "rounded,filled",
        },
    }

    g = graphviz.Digraph(**kwargs)
    for node in nodes:
        g.node(node, color='black', fillcolor=coloring_fn(node))

    for edge in edges:
        g.edge(edge[0], edge[1], color=coloring_fn(edge[0]))
        
    for node in nodes:
        for cin in constant_in:
            if node not in constant_in:
                g.edge(cin, node, color=constant_edge_color)
        
    g.render(out_path)
    os.rename(out_path_temp, out_path)

    
if __name__ == '__main__':
    main()