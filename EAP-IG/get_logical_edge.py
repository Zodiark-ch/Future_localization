import os
import json
import argparse
import sys
import re
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/solver/"
    )
)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-edge1", "--edge1", type=str, default="/home/lthpc/hangc/EAP-IG/ioi_localization2w_graph_epoch_1.json")
    parser.add_argument("-edge2", "--edge2", type=str, default="/home/lthpc/hangc/EAP-IG/ioi_localization2w_graph_epoch_1_or.json")
    parser.add_argument("-w", "--with_embedding_nodes", action="store_true", default=True)
    parser.add_argument("-o", "--output", type=str, default="")
    
    args = parser.parse_args()
    
    if args.output == "":
        
        args.output = os.path.join("/home/lthpc/hangc/EAP-IG", "ioi_localization2w_graph_epoch_1_bool.json")
    else:
        raise ValueError("Output file must be specified when comparing two models") 
    
    return args

def add_o_suffix_to_nodes(edges):
    """
    对于"a"开头的节点，如果最后的字符是数字而不是'q', 'k', 'v'中的一种，加上后缀字符'.o'
    然后添加额外的边处理逻辑
    """
    modified_edges = []
    for edge in edges:
        from_node, to_node = edge
        
        # 处理from_node
        if from_node.startswith("a") and from_node[-1].isdigit() and from_node[-1] not in ['q', 'k', 'v', 'o']:
            from_node = from_node + ".o"
        
        # 处理to_node
        if to_node.startswith("a") and to_node[-1].isdigit() and to_node[-1] not in ['q', 'k', 'v', 'o']:
            to_node = to_node + ".o"
        
        modified_edges.append((from_node, to_node))
    
    # 额外的边处理逻辑
    additional_edges = []
    
    # 构建所有边的集合，用于快速查找
    all_edges_set = set(modified_edges)
    
    # 构建from_node的集合，用于检查某个节点是否作为from_node存在
    from_nodes_set = set(edge[0] for edge in modified_edges)
    
    for edge in modified_edges:
        from_node, to_node = edge
        
        # 检查条件1：to_node以"a"开头并且最后的字符是'q', 'k', 'v'中的一种
        if (to_node.startswith("a") and 
            to_node[-1] in ['q', 'k', 'v']):
            
            # 检查条件2：这个to_node不以from_node的方式存在于其他的边中
            if to_node not in from_nodes_set:
                
                # 检查条件3：这个to_node用'.o'替换'q', 'k', 'v'后(命名为o_node)，以from_node的形式存在于其他的边中
                o_node = to_node[:-1] + 'o'  # 将最后一个字符替换为'o'
                
                if o_node in from_nodes_set:
                    # 满足所有条件，添加新边 [to_node, o_node]
                    additional_edges.append((to_node, o_node))
    
    # 将额外的边添加到结果中
    modified_edges.extend(additional_edges)
    
    return modified_edges

def classify_h_numbers(edges):
    """
    对于"a"开头的节点，如果最后的字符是'k', 'v'中的一个，进行h数字分类处理
    例如：'a14.h6.q' -> 'a14.H1.q' (因为h6属于H1组)
    """
    modified_edges = []
    for edge in edges:
        from_node, to_node,type = edge
        
        # 处理from_node
        if from_node.startswith("a") and from_node[-1] in ['k', 'v']:
            from_node = process_h_classification(from_node)
        
        # 处理to_node
        if to_node.startswith("a") and to_node[-1] in ['k', 'v']:
            to_node = process_h_classification(to_node)
        
        modified_edges.append((from_node, to_node,type))
    
    return modified_edges

def process_h_classification(node_name):
    """
    处理单个节点的h数字分类
    例如：'a14.h6.k' -> 'a14.H1.k'
    """
    if not node_name.startswith("a") or node_name[-1] not in ['k', 'v']:
        return node_name
    
    # 使用正则表达式分割节点名称
    # 匹配模式：a数字.h数字.字母
    pattern = r'^([a-z]\d+)\.(h\d+)\.([qkv])$'
    match = re.match(pattern, node_name)
    
    if match:
        prefix, h_part, suffix = match.groups()
        
        # 提取h后面的数字
        h_number = int(h_part[1:])  # 去掉'h'，获取数字
        
        # 计算H组号：每4个数字为一组
        h_group = h_number // 4
        
        # 构建新的节点名称
        new_node_name = f"{prefix}.H{h_group}.{suffix}"
        return new_node_name
    
    return node_name

def find_OR_gate_edges(edges_ns, edges_dn):
    """
    找出logical gate的sub_edges，满足以下条件：
    1. 都在edges_dn中存在，而在edges_ns中不存在
    2. to_node不会以".o"结尾
    3. 对于任何一个to_node，应该能在sub_edges中的另一个元素中找到相同的to_node
    """
    # 将edges转换为set以便快速查找
    edges_ns_set = set(edges_ns)
    edges_dn_set = set(edges_dn)
    
    # 条件1：找出在edges_dn中存在，而在edges_ns中不存在的边
    candidate_edges = edges_dn_set - edges_ns_set
    
    # 条件2：过滤掉to_node以".o"结尾的边
    filtered_edges = []
    for edge in candidate_edges:
        from_node, to_node = edge
        if not to_node.endswith(".o"):
            filtered_edges.append(edge)
    
    # 条件3：确保每个to_node在sub_edges中至少出现两次
    to_node_count = {}
    for edge in filtered_edges:
        to_node = edge[1]
        to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
    # 检查edges_ns_set中的边，如果某个to_node只以to_node身份出现过一次，也将其计数加1
    ns_to_node_count = {}
    for edge in edges_ns_set:
        to_node = edge[1]
        ns_to_node_count[to_node] = ns_to_node_count.get(to_node, 0) + 1
    
    # 将只出现一次的to_node的计数加到to_node_count中
    for to_node, count in ns_to_node_count.items():
        if count == 1:
            to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
    # 只保留to_node出现次数>=2的边
    sub_edges = []
    for edge in filtered_edges:
        to_node = edge[1]
        if to_node_count[to_node] >= 2:
            sub_edges.append(edge)
    
    return sub_edges

def find_AND_gate_edges(edges_ns, edges_dn):
    """
    找出logical gate的sub_edges，满足以下条件：
    1. 都在edges_ns中存在，而在edges_dn中不存在
    2. to_node不会以".o"结尾
    3. 对于任何一个to_node，应该能在sub_edges中的另一个元素中找到相同的to_node
    """
    # 将edges转换为set以便快速查找
    edges_ns_set = set(edges_ns)
    edges_dn_set = set(edges_dn)
    
    # 条件1：找出在edges_dn中存在，而在edges_ns中不存在的边
    candidate_edges = edges_ns_set-edges_dn_set 
    
    # 条件2：过滤掉to_node以".o"结尾的边
    filtered_edges = []
    for edge in candidate_edges:
        from_node, to_node = edge
        if not to_node.endswith(".o"):
            filtered_edges.append(edge)
    
    # 条件3：确保每个to_node在sub_edges中至少出现两次
    to_node_count = {}
    for edge in filtered_edges:
        to_node = edge[1]
        to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
    # 检查edges_ns_set中的边，如果某个to_node只以to_node身份出现过一次，也将其计数加1
    dn_to_node_count = {}
    for edge in edges_dn_set:
        to_node = edge[1]
        dn_to_node_count[to_node] = dn_to_node_count.get(to_node, 0) + 1
    
    # 将只出现一次的to_node的计数加到to_node_count中
    for to_node, count in dn_to_node_count.items():
        if count == 1:
            to_node_count[to_node] = to_node_count.get(to_node, 0) + 1
    
    # 只保留to_node出现次数>=2的边
    sub_edges = []
    for edge in filtered_edges:
        to_node = edge[1]
        if to_node_count[to_node] >= 2:
            sub_edges.append(edge)
    
    return sub_edges

def create_bool_edges(edges_ns, edges_dn, or_edges, and_edges):
    """
    创建bool_edges，包含所有边并添加标签：
    - 如果边在sub_edges中存在，label为"OR"
    - 否则label为"AND"
    - 每个元素从[from_node, to_node]扩展为[from_node, to_node, label]
    """
    # 合并edges_ns和edges_dn，去重
    all_edges = list(set(edges_ns + edges_dn))
    
    # 将sub_edges转换为set以便快速查找
    or_edges_set = set(or_edges)
    and_edges_set = set(and_edges)
    
    # 创建bool_edges，添加标签
    bool_edges = []
    for edge in all_edges:
        from_node, to_node = edge
        
        # 确定标签
        if edge in or_edges_set:
            label = "OR"
        elif edge in and_edges_set:
            label = "AND"
        else:
            label = "ADDER"
        
        # 扩展为三元组
        bool_edges.append([from_node, to_node, label])
    
    return bool_edges

def validate_paths_to_resid_post(bool_edges):
    """
    验证bool_edges中的所有from_node都能递归地找到路径到达"resid_post"
    如果发现节点不在邻接表中，会动态添加边并重新验证
    """
    # 创建bool_edges的副本，以便动态修改
    current_bool_edges = bool_edges.copy()
    

    # 构建邻接表，便于查找
    adjacency_list = {}
    for edge in current_bool_edges:
        from_node, to_node, label = edge
        if from_node not in adjacency_list:
            adjacency_list[from_node] = []
        adjacency_list[from_node].append(to_node)
    
    # 获取所有唯一的from_node
    all_from_nodes = set(edge[0] for edge in current_bool_edges)
    
    # 验证每个from_node
    failed_nodes = []
    successful_nodes = []
    added_edges = []
    
    for from_node in all_from_nodes:
        success, failed_node = can_reach_resid_post(from_node, adjacency_list, set(),current_bool_edges)
        if success:
            successful_nodes.append(from_node)
        else:
            failed_nodes.append(failed_node)
            # 如果失败的节点不在邻接表中，添加一条到resid_post的边
                
    
    # 返回验证结果
    result = {
        "total_from_nodes": len(all_from_nodes),
        "successful_nodes": len(successful_nodes),
        "failed_nodes": len(failed_nodes),
        "failed_node_list": failed_nodes,
        "is_valid": len(failed_nodes) == 0,
        "updated_bool_edges": current_bool_edges
    }
    
    return current_bool_edges

def can_reach_resid_post(node, adjacency_list, visited,current_bool_edges):
    """
    递归检查节点是否能到达resid_post
    使用深度优先搜索避免循环
    返回元组 (success, failed_node)，其中failed_node是导致失败的节点
    """
    # 如果已经访问过这个节点，说明有循环，返回False和当前节点
    if node in visited:
        return False, node
    visited.add(node)
    
    # 如果当前节点就是resid_post，返回True和None
    if node == "resid_post":
        return True, None
    
    # 如果节点不在邻接表中，说明没有出边，返回False和当前节点
    if node not in adjacency_list:
        new_edge = [node, "resid_post", "ADDER"]
        current_bool_edges.append(new_edge)
        return False, node
    
    # 标记当前节点为已访问
    
    # 递归检查所有邻居节点
    for neighbor in adjacency_list[node]:
        success, node = can_reach_resid_post(neighbor, adjacency_list, visited.copy(),current_bool_edges)
        if success:
            return True, None
    
    # 如果所有邻居都无法到达resid_post，返回False和当前节点
    return False, node

def save_bool_edges(bool_edges, output_path):
    """
    将bool_edges保存到指定的输出路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存为JSON格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bool_edges, f, indent=2, ensure_ascii=False)
    
    print(f"成功保存 {len(bool_edges)} 条边到 {output_path}")

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

def main():
    args = parse_args()
    edges_ns = json.load(open(args.edge1))
    edges_dn = json.load(open(args.edge2))
    print("原始边数量:", len(edges_ns), len(edges_dn))
    
    # if strict, uncomment the following
    
    # # 原有的sanitize_edges处理
    # edges_ns = sanitize_edges(edges_ns)
    # edges_dn = sanitize_edges(edges_dn)
    # print("sanitize后:", len(edges_ns), len(edges_dn))
    
    #应用第一个功能：为数字结尾的a节点添加.o后缀
    edges_ns = add_o_suffix_to_nodes(edges_ns)
    edges_dn = add_o_suffix_to_nodes(edges_dn)
    print("添加.o后缀后:", len(edges_ns), len(edges_dn))
    
    # 应用第二个功能：对k,v结尾的节点进行h数字分类
    # edges_ns = classify_h_numbers(edges_ns)
    # edges_dn = classify_h_numbers(edges_dn)
    # print("h数字分类后:", len(edges_ns), len(edges_dn))
    
    # 删除重复的边
    edges_ns = list(set(edges_ns))
    edges_dn = list(set(edges_dn))
    print("去重后:", len(edges_ns), len(edges_dn))
    
    # 找出logical gate的sub_edges
    OR_edges = find_OR_gate_edges(edges_ns, edges_dn)
    AND_edges = find_AND_gate_edges(edges_ns, edges_dn)
    print("OR gate边数量:", len(OR_edges))
    print("AND gate边数量:", len(AND_edges))
    
    # 创建bool_edges，合并所有边并添加标签
    bool_edges = create_bool_edges(edges_ns, edges_dn, OR_edges,AND_edges)
    print("bool_edges数量:", len(bool_edges))
    
    # 验证所有from_node都能到达resid_post
    validation_result = validate_paths_to_resid_post(bool_edges)
    print("路径验证结果:", len(validation_result))
    validation_result=classify_h_numbers(validation_result)
    validation_result=list(set(tuple(edge) for edge in validation_result))
    print("h数字分类后:", len(validation_result))
    
    # 保存bool_edges到输出文件
    save_bool_edges(validation_result, args.output)
    print(f"bool_edges已保存到: {args.output}")
    
    
    
    
if __name__ == '__main__':
    main()