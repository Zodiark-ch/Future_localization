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

class Circuit:
    def __init__(self, json_file_path, leaf_nodes=None):
        """
        初始化Circuit类
        Args:
            json_file_path: JSON文件路径，包含circuit数据
            leaf_nodes: 叶子节点列表，用于拓扑排序构建circuit
        """
        self.circuit = {}
        self.NOR = False  # 默认NOR参数
        self.load_circuit(json_file_path, leaf_nodes)
    
    def load_circuit(self, json_file_path, leaf_nodes=None):
        """
        从JSON文件加载circuit数据并按照拓扑排序重组为字典结构
        Args:
            json_file_path: JSON文件路径
            leaf_nodes: 叶子节点列表，用于拓扑排序构建circuit
        """
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # 初始化known_node_list，从叶子节点开始
            known_node_list = set(leaf_nodes) if leaf_nodes else set()
            
            # 创建所有to_node的映射，用于检查依赖关系
            to_node_dependencies = {}
            for item in data:
                if len(item) == 3:
                    from_node, to_node, node_type = item
                    if to_node not in to_node_dependencies:
                        to_node_dependencies[to_node] = []
                    to_node_dependencies[to_node].append((from_node, node_type))
            
            # 拓扑排序构建circuit
            while to_node_dependencies:
                # 找到所有依赖都已满足的to_node
                ready_to_process = []
                for to_node, dependencies in to_node_dependencies.items():
                    all_dependencies_met = True
                    for from_node, _ in dependencies:
                        if from_node not in known_node_list:
                            all_dependencies_met = False
                            break
                    
                    if all_dependencies_met:
                        ready_to_process.append(to_node)
                
                # 如果没有找到可以处理的节点，说明存在循环依赖
                if not ready_to_process:
                    print(f"警告：存在循环依赖，无法完成拓扑排序")
                    # 将剩余的节点直接加入circuit
                    # for to_node, dependencies in to_node_dependencies.items():
                    #     if to_node not in self.circuit:
                    #         self.circuit[to_node] = {}
                    #     for from_node, node_type in dependencies:
                    #         if node_type not in self.circuit[to_node]:
                    #             self.circuit[to_node][node_type] = []
                    #         self.circuit[to_node][node_type].append(from_node)
                    break
                
                # 处理所有可以处理的to_node
                for to_node in ready_to_process:
                    dependencies = to_node_dependencies[to_node]
                    
                    # 构建circuit结构
                    if to_node not in self.circuit:
                        self.circuit[to_node] = {}
                    
                    for from_node, node_type in dependencies:
                        if node_type not in self.circuit[to_node]:
                            self.circuit[to_node][node_type] = []
                        self.circuit[to_node][node_type].append(from_node)
                    
                    # 将to_node加入known_node_list
                    known_node_list.add(to_node)
                    
                    # 从待处理列表中删除
                    del to_node_dependencies[to_node]
                    
        except FileNotFoundError:
            print(f"错误：找不到文件 {json_file_path}")
        except json.JSONDecodeError:
            print(f"错误：JSON文件格式不正确 {json_file_path}")
        except Exception as e:
            print(f"加载circuit数据时发生错误：{e}")
    
    def bool_validation(self, VALUE_retain, VALUE_forget, input_retain=None, input_forget=None, NOR=None):
        """
        布尔验证circuit中的节点赋值
        Args:
            VALUE_retain: 包含[node, bool_value]的列表，用于非NOR模式
            VALUE_forget: 包含[node, bool_value]的列表，用于NOR模式
            input_retain: 包含叶子节点[node, bool_value]的列表，用于非NOR模式
            input_forget: 包含叶子节点[node, bool_value]的列表，用于NOR模式
            NOR: 是否使用NOR模式（如果为None则使用实例的NOR属性）
        Returns:
            bool: 如果验证通过返回True，否则返回False
            list: 如果验证通过，返回[node, numeric_value]的列表，否则返回空列表
        """
        # 如果NOR参数为None，使用实例的NOR属性
        if NOR is None:
            NOR = self.NOR
        
        # 创建实时value的list，包含当前circuit的所有node的取值
        real_time_values = {}
        
        # 收集circuit中的所有节点
        all_nodes = set()
        for to_node, type_groups in self.circuit.items():
            all_nodes.add(to_node)
            for from_nodes in type_groups.values():
                all_nodes.update(from_nodes)
        
        # 初始化实时value的list
        for node in all_nodes:
            node_value = None
            
            if NOR:
                # 如果是NOR模式，参考VALUE_forget
                for n, value in VALUE_forget:
                    if n == node:
                        node_value = value
                        break
            else:
                # 如果不是NOR模式，参考VALUE_retain
                for n, value in VALUE_retain:
                    if n == node:
                        node_value = value
                        break
            
            real_time_values[node] = node_value
        
        # 更新input_retain和input_forget中的叶子节点值
        if NOR:
            # NOR模式：使用input_forget更新叶子节点值
            for node, value in input_forget:
                if node in real_time_values:
                    real_time_values[node] = value
        else:
            # 非NOR模式：使用input_retain更新叶子节点值
            if input_retain is not None:
                for node, value in input_retain:
                    if node in real_time_values:
                        real_time_values[node] = value
        
        
        # 对self.circuit中的每个key进行bool计算
        for to_node, type_groups in self.circuit.items():
            for node_type, from_nodes in type_groups.items():
                # 获取from_node的值
                from_node_values = []
                for from_node in from_nodes:
                    from_node_value = real_time_values.get(from_node)
                    if from_node_value is None:
                        # 如果from_node的值为None，无法进行计算
                        return False, []
                    from_node_values.append(from_node_value)
                
                # 根据NOR模式和节点类型进行bool计算
                if NOR:
                    # NOR模式：所有的from_node先取非
                    inverted_from_values = [not val for val in from_node_values]
                    
                    if node_type in ["OR", "ADDER"]:
                        # OR/ADDER: compute_result = from_node1 and from_node2 and from_node3 ...
                        compute_result = all(inverted_from_values)
                    elif node_type == "AND":
                        # AND: compute_result = from_node1 or from_node2 or from_node3 ...
                        compute_result = any(inverted_from_values)
                    else:
                        # 未知节点类型
                        return False, []
                else:
                    # 非NOR模式：from_node值直接使用
                    if node_type in ["AND", "ADDER"]:
                        # AND/ADDER: compute_result = from_node1 and from_node2 and from_node3 ...
                        compute_result = all(from_node_values)
                    elif node_type == "OR":
                        # OR: compute_result = from_node1 or from_node2 or from_node3 ...
                        compute_result = any(from_node_values)
                    else:
                        # 未知节点类型
                        return False, []
                
                # 检查to_node的值
                to_node_value = real_time_values.get(to_node)
                
                if to_node_value is None:
                    # 如果to_node的值为None，将compute_result赋值给to_node
                    if NOR:
                        real_time_values[to_node] = not compute_result
                    else:
                        real_time_values[to_node] = compute_result
                else:
                    # 如果to_node不为None，判断compute_result和to_node是否相等，因为to_node_value要取非，所以相等就是不相等
                    if NOR:
                        if compute_result == to_node_value:
                            return False, []
                    else:
                        if compute_result != to_node_value:
                            return False, []
        
        # 验证通过，返回每个节点和实时value的list中的value搭配起来的list
        
        return True, real_time_values

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-circuit1", "--circuit1", type=str, default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/runs/cyber_edges/edges_bool.json")
    parser.add_argument("-circuit2", "--circuit2", type=str, default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/runs/winogrande_edges/edges_bool.json")
    parser.add_argument("-circuit3", "--circuit3", type=str, default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/runs/sst2_edges/edges_bool.json")
    parser.add_argument("-circuit4", "--circuit4", type=str, default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/runs/bool_edges/edges_bool.json")
    parser.add_argument("-circuit5", "--circuit5", type=str, default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/runs/ioi_edges/edges_bool.json")
    parser.add_argument("-circuit6", "--circuit6", type=str, default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/runs/gp_edges/edges_bool.json")
    parser.add_argument("-circuit7", "--circuit7", type=str, default="/home/lthpc/hangc/CSAT/Edge-Pruning/data/runs/induction_edges/edges_bool.json")
    parser.add_argument("-w", "--with_embedding_nodes", action="store_true", default=True)
    
    args = parser.parse_args()
    
    return args




def initialize_values(args, leaf_nodes_by_circuit):
    """
    根据args中的circuit文件初始化VALUE_retain、VALUE_forget、input_retain和input_forget
    Args:
        args: 包含circuit文件路径的参数对象
        leaf_nodes_by_circuit: 每个circuit的叶子节点字典
    Returns:
        tuple: (VALUE_retain, VALUE_forget, input_retain, input_forget) 四个包含[node, bool_value]的列表
    """
    VALUE_retain = []
    VALUE_forget = []
    input_retain = []
    input_forget = []
    
    # 获取所有circuit文件路径
    circuit_files = []
    for i in range(1, 7):  # 检查circuit1到circuit4
        circuit_attr = f"circuit{i}"
        if hasattr(args, circuit_attr):
            circuit_path = getattr(args, circuit_attr)
            if circuit_path and os.path.exists(circuit_path):
                circuit_files.append((i, circuit_path))
    
    if not circuit_files:
        print("错误：没有找到有效的circuit文件！")
        return VALUE_retain, VALUE_forget, input_retain, input_forget
    
    print(f"找到 {len(circuit_files)} 个circuit文件")
    
    # 处理每个circuit文件
    for circuit_num, circuit_path in circuit_files:
        try:
            # 直接加载JSON数据来获取所有节点，不依赖Circuit类
            with open(circuit_path, 'r') as f:
                data = json.load(f)
            
            # 收集该circuit中的所有节点
            circuit_nodes = set()
            for item in data:
                if len(item) == 3:
                    from_node, to_node, node_type = item
                    circuit_nodes.add(from_node)
                    circuit_nodes.add(to_node)
            
            circuit_nodes_list = list(circuit_nodes)
            print(f"Circuit{circuit_num} ({circuit_path}) 包含 {len(circuit_nodes_list)} 个节点")
            
            # 根据circuit编号分配节点
            if circuit_num == 1:
                # circuit1的所有节点添加到VALUE_forget，值为False
                for node in circuit_nodes_list:
                    VALUE_forget.append([node, False])
                print(f"  → 添加到VALUE_forget（取值为False）")
            else:
                # 其他circuit的所有节点添加到VALUE_retain，值为True
                for node in circuit_nodes_list:
                    VALUE_retain.append([node, True])
                print(f"  → 添加到VALUE_retain（取值为True）")
                
        except Exception as e:
            print(f"处理circuit{circuit_num}时发生错误：{e}")
            continue
    
    # 初始化input_forget和input_retain
    print("\n初始化input_forget和input_retain...")
    
    # input_forget的节点就是leaf_nodes_by_circuit[1]的节点，value默认为False
    if 1 in leaf_nodes_by_circuit:
        circuit1_leaf_nodes = leaf_nodes_by_circuit[1]
        for node in circuit1_leaf_nodes:
            input_forget.append([node, False])
        print(f"  input_forget: {len(input_forget)} 个节点（来自Circuit1的叶子节点，值默认为False）")
    else:
        print("  警告：没有找到Circuit1的叶子节点")
    
    # input_retain的节点就是leaf_nodes_by_circuit[2:]的所有节点的集合，value默认为True
    all_other_leaf_nodes = set()
    for circuit_num in range(2, 7):  # circuit2到circuit4
        if circuit_num in leaf_nodes_by_circuit:
            all_other_leaf_nodes.update(leaf_nodes_by_circuit[circuit_num])
    
    for node in all_other_leaf_nodes:
        input_retain.append([node, True])
    print(f"  input_retain: {len(input_retain)} 个节点（来自Circuit2-4的叶子节点，值默认为True）")
    
    # 对所有四个列表进行去重处理
    #print("\n进行去重处理...")
    
    # 去重VALUE_retain
    seen_nodes = set()
    unique_VALUE_retain = []
    for node, value in VALUE_retain:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_VALUE_retain.append([node, value])
    VALUE_retain = unique_VALUE_retain
    #print(f"  VALUE_retain去重后: {len(VALUE_retain)} 个节点")
    
    # 去重VALUE_forget
    seen_nodes = set()
    unique_VALUE_forget = []
    for node, value in VALUE_forget:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_VALUE_forget.append([node, value])
    VALUE_forget = unique_VALUE_forget
    #print(f"  VALUE_forget去重后: {len(VALUE_forget)} 个节点")
    
    # 去重input_retain
    seen_nodes = set()
    unique_input_retain = []
    for node, value in input_retain:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_input_retain.append([node, value])
    input_retain = unique_input_retain
    #print(f"  input_retain去重后: {len(input_retain)} 个节点")
    
    # 去重input_forget
    seen_nodes = set()
    unique_input_forget = []
    for node, value in input_forget:
        if node not in seen_nodes:
            seen_nodes.add(node)
            unique_input_forget.append([node, value])
    input_forget = unique_input_forget
    #print(f"  input_forget去重后: {len(input_forget)} 个节点")
    
    return VALUE_retain, VALUE_forget, input_retain, input_forget



def modify(VALUE_retain, VALUE_forget, input_retain, input_forget, circuits):
    """
    修改VALUE_retain, VALUE_forget, input_retain, input_forget的值
    Args:
        VALUE_retain: 保留节点列表 [[node, bool_value], ...]
        VALUE_forget: 遗忘节点列表 [[node, bool_value], ...]
        input_retain: 输入保留节点列表 [[node, bool_value], ...]
        input_forget: 输入遗忘节点列表 [[node, bool_value], ...]
        circuits: 所有circuit的字典 {circuit_num: circuit}
    Returns:
        tuple: (VALUE_retain, VALUE_forget, input_retain, input_forget) 修改后的四个列表
    """
    import copy
    
    # 创建副本以避免修改原始数据
    VALUE_retain = copy.deepcopy(VALUE_retain)
    VALUE_forget = copy.deepcopy(VALUE_forget)
    input_retain = copy.deepcopy(input_retain)
    input_forget = copy.deepcopy(input_forget)
    
    print("\n开始执行modify函数...")
    
    # 1. 将VALUE_forget和VALUE_retain的所有node的value都从bool值变成None
    print("1. 将所有VALUE_forget和VALUE_retain的值设为None...")
    for i in range(len(VALUE_forget)):
        VALUE_forget[i][1] = None
    for i in range(len(VALUE_retain)):
        VALUE_retain[i][1] = None
    
    # 2. 查找那些只在circuit1中出现，在其他circuit中不存在的node
    print("2. 查找只在circuit1中出现的节点...")
    circuit1_nodes = set()
    other_circuit_nodes = set()
    
    # 获取circuit1的所有节点
    if 1 in circuits:
        circuit1 = circuits[1]
        for to_node, type_groups in circuit1.circuit.items():
            circuit1_nodes.add(to_node)
            for from_nodes in type_groups.values():
                circuit1_nodes.update(from_nodes)
    
    # 获取其他circuit的所有节点
    for circuit_num, circuit in circuits.items():
        if circuit_num != 1:
            for to_node, type_groups in circuit.circuit.items():
                other_circuit_nodes.add(to_node)
                for from_nodes in type_groups.values():
                    other_circuit_nodes.update(from_nodes)
    
    # # 只在circuit1中出现的节点
    # only_in_circuit1 = circuit1_nodes - other_circuit_nodes
    # for node in only_in_circuit1:
    #     # 在VALUE_forget中设置值为False
    #     for i, (n, _) in enumerate(VALUE_forget):
    #         if n == node:
    #             VALUE_forget[i][1] = False
    #             break
    # print(f"  找到 {len(only_in_circuit1)} 个只在circuit1中出现的节点，已设置为False")
    
    # # 3. 查找那些只在其他circuit中出现，并且在circuit1中不存在的node
    # print("3. 查找只在其他circuit中出现的节点...")
    # only_in_other_circuits = other_circuit_nodes - circuit1_nodes
    # for node in only_in_other_circuits:
    #     # 在VALUE_retain中设置值为True
    #     for i, (n, _) in enumerate(VALUE_retain):
    #         if n == node:
    #             VALUE_retain[i][1] = True
    #             break
    # print(f"  找到 {len(only_in_other_circuits)} 个只在其他circuit中出现的节点，已设置为True")
    
    # 4. 从circuit1中的resid_post开始，递归处理OR和ADDER类型的from_node
    print("4. 处理circuit1中的OR和ADDER类型节点...")
    if 1 in circuits:
        circuit1 = circuits[1]
        processed_nodes = set()
        
        def process_circuit1_node(node):
            """递归处理circuit1中的节点"""
            if node in processed_nodes:
                return
            
            processed_nodes.add(node)
            
            if node in circuit1.circuit:
                for node_type, from_nodes in circuit1.circuit[node].items():
                    if node_type in ["OR", "ADDER"]:
                        for from_node in from_nodes:
                            # 检查from_node在VALUE_forget中的值
                            from_node_value = None
                            for i, (n, v) in enumerate(VALUE_forget):
                                if n == from_node:
                                    from_node_value = v
                                    break
                            
                            if from_node_value is None:
                                # 设置为False
                                for i, (n, v) in enumerate(VALUE_forget):
                                    if n == from_node:
                                        VALUE_forget[i][1] = False
                                        break
                                #print(f"    节点 {from_node} 设置为False")
                            elif from_node_value != False:
                                # 报错
                                raise ValueError(f"节点 {from_node} 的值不是False，而是 {from_node_value}")
                            
                            # 递归处理这个from_node
                            process_circuit1_node(from_node)
        
        # 从resid_post开始处理
        resid_post_nodes = []
        for to_node in circuit1.circuit.keys():
            if "resid_post" in to_node:
                for i, (n, v) in enumerate(VALUE_forget):
                                    if n == 'resid_post':
                                        VALUE_forget[i][1] = False
                                        break
                resid_post_nodes.append(to_node)
        
        for resid_post in resid_post_nodes:
            print(f"  从 {resid_post} 开始处理...")
            process_circuit1_node(resid_post)
    
    # 5. 从其他circuit中的resid_post开始，递归处理AND和ADDER类型的from_node
    print("5. 处理其他circuit中的AND和ADDER类型节点...")
    for circuit_num, circuit in circuits.items():
        if circuit_num != 1:
            processed_nodes = set()
            
            def process_other_circuit_node(node):
                """递归处理其他circuit中的节点"""
                if node in processed_nodes:
                    return
                
                processed_nodes.add(node)
                
                if node in circuit.circuit:
                    for node_type, from_nodes in circuit.circuit[node].items():
                        if node_type in ["AND", "ADDER"]:
                            for from_node in from_nodes:
                                # 检查from_node在VALUE_retain中的值
                                from_node_value = None
                                for i, (n, v) in enumerate(VALUE_retain):
                                    if n == from_node:
                                        from_node_value = v
                                        break
                                
                                if from_node_value is None:
                                    # 设置为True
                                    for i, (n, v) in enumerate(VALUE_retain):
                                        if n == from_node:
                                            VALUE_retain[i][1] = True
                                            break
                                    #print(f"    节点 {from_node} 设置为True")
                                elif from_node_value != True:
                                    # 报错
                                    raise ValueError(f"节点 {from_node} 的值不是True，而是 {from_node_value}")
                                
                                # 递归处理这个from_node
                                process_other_circuit_node(from_node)
            
            # 从resid_post开始处理
            resid_post_nodes = []
            for to_node in circuit.circuit.keys():
                if "resid_post" in to_node:
                    for i, (n, v) in enumerate(VALUE_retain):
                                        if n == "resid_post":
                                            VALUE_retain[i][1] = True
                                            break
                    resid_post_nodes.append(to_node)
            
            for resid_post in resid_post_nodes:
                print(f"  从Circuit{circuit_num}的 {resid_post} 开始处理...")
                process_other_circuit_node(resid_post)
    
    # # 6. 处理circuit1中的AND类型节点
    # print("6. 处理circuit1中的AND类型节点...")
    # if 1 in circuits:
    #     circuit1 = circuits[1]
    #     for to_node, type_groups in circuit1.circuit.items():
    #         for node_type, from_nodes in type_groups.items():
    #             if node_type == "AND":
    #                 # 检查是否存在任意一个from_node的value为False
    #                 has_false_from_node = False
    #                 for from_node in from_nodes:
    #                     for i, (n, v) in enumerate(VALUE_forget):
    #                         if n == from_node and v == False:
    #                             has_false_from_node = True
    #                             break
    #                     if has_false_from_node:
    #                         break
                    
    #                 if has_false_from_node:
    #                     # 将其他from_node在VALUE_forget中的值设置为False
    #                     for from_node in from_nodes:
    #                         # 检查这个from_node是否在input_forget中
    #                         in_input_forget = False
    #                         for n, v in input_forget:
    #                             if n == from_node:
    #                                 in_input_forget = True
    #                                 break
                            
    #                         if in_input_forget:
    #                             # 在VALUE_forget中设置值为False
    #                             for i, (n, v) in enumerate(VALUE_forget):
    #                                 if n == from_node and VALUE_forget[i][1] is None:
    #                                     VALUE_forget[i][1] = True
    #                                     print(f"    节点 {from_node} 在VALUE_forget中设置为False")
    #                                     break
    
    # # 7. 处理其他circuit中的OR类型节点
    # print("7. 处理其他circuit中的OR类型节点...")
    # seen_node_list = []
    # for circuit_num, circuit in circuits.items():
    #     if circuit_num != 1:
    #         for to_node, type_groups in circuit.circuit.items():
    #             for node_type, from_nodes in type_groups.items():
    #                 if node_type == "OR":
    #                     # 检查是否存在任意一个from_node的value为True
    #                     has_true_from_node = False
    #                     for from_node in from_nodes:
    #                         # 如果from_node在seen_node_list中，跳过
    #                         if from_node in seen_node_list:
    #                             continue
    #                         for i, (n, v) in enumerate(VALUE_retain):
    #                             if n == from_node and v == True:
    #                                 has_true_from_node = True
    #                                 break
    #                         if has_true_from_node:
    #                             break
                        
    #                     if has_true_from_node:
    #                         # 将其他from_node在VALUE_retain中的值设置为True
    #                         for from_node in from_nodes:
    #                             if from_node in seen_node_list:
    #                                 continue
    #                             # 检查这个from_node是否在input_retain中
    #                             in_input_retain = False
    #                             for n, v in input_retain:
    #                                 if n == from_node:
    #                                     in_input_retain = True
    #                                     break
                                
    #                             if in_input_retain:
    #                                                                         # 在VALUE_retain中设置值为True
    #                                     for i, (n, v) in enumerate(VALUE_retain):
    #                                         if n == from_node and VALUE_retain[i][1] is None:
    #                                             VALUE_retain[i][1] = False
    #                                             #print(f"    节点 {from_node} 在VALUE_retain中设置为True")
    #                                         elif n == from_node and VALUE_retain[i][1] ==False:
    #                                             seen_node_list.append(from_node)
    #                                             break
    
    # 8. 更新input_retain的值
    print("8. 更新input_retain的值...")
    # 将input_retain中所有节点的value设为None
    for i in range(len(input_retain)):
        input_retain[i][1] = None
    
    # 遍历VALUE_retain，更新input_retain中的值
    for node, value in VALUE_retain:
        if value is not None:
            for i, (n, v) in enumerate(input_retain):
                if n == node:
                    input_retain[i][1] = value
                    print(f"    节点 {node} 在input_retain中更新为 {value}")
                    break
    
    # 9. 更新input_forget的值
    print("9. 更新input_forget的值...")
    # 将input_forget中所有节点的value设为None
    for i in range(len(input_forget)):
        input_forget[i][1] = None
    
    # 遍历VALUE_forget，更新input_forget中的值
    for node, value in VALUE_forget:
        if value is not None:
            for i, (n, v) in enumerate(input_forget):
                if n == node:
                    input_forget[i][1] = value
                    print(f"    节点 {node} 在input_forget中更新为 {value}")
                    break
    
    # 10. 统计value=None的节点数量
    print("10. 统计value=None的节点数量...")
    none_count_retain = sum(1 for _, v in VALUE_retain if v is None)
    none_count_forget = sum(1 for _, v in VALUE_forget if v is None)
    none_count_input_retain = sum(1 for _, v in input_retain if v is None)
    none_count_input_forget = sum(1 for _, v in input_forget if v is None)
    
    print(f"  VALUE_retain中value=None的节点数量: {none_count_retain}")
    print(f"  VALUE_forget中value=None的节点数量: {none_count_forget}")
    print(f"  input_retain中value=None的节点数量: {none_count_input_retain}")
    print(f"  input_forget中value=None的节点数量: {none_count_input_forget}")
    
    # 11. 统计input_retain和input_forget中为None的节点，构成uncertain_node_list
    print("11. 统计不确定节点列表...")
    uncertain_node_list_retain = []
    uncertain_node_list_forget = []
    
    for node, value in input_retain:
        if value is None:
            uncertain_node_list_retain.append(node)
    
    for node, value in input_forget:
        if value is None:
            uncertain_node_list_forget.append(node)
    
    print(f"  uncertain_node_list_retain: {len(uncertain_node_list_retain)} 个节点")
    if uncertain_node_list_retain:
        print(f"    节点列表: {uncertain_node_list_retain}")
    
    print(f"  uncertain_node_list_forget: {len(uncertain_node_list_forget)} 个节点")
    if uncertain_node_list_forget:
        print(f"    节点列表: {uncertain_node_list_forget}")
    
    print("modify函数执行完成")
    return VALUE_retain, VALUE_forget, input_retain, input_forget, uncertain_node_list_retain, uncertain_node_list_forget


def investigate_leaf_nodes(circuit_files):
    """
    调查每个circuit的叶子节点
    叶子节点的定义：该node仅出现在from_node中，不出现在to_node中
    Args:
        circuit_files: 包含(circuit_num, circuit_path)元组的列表
    Returns:
        dict: {circuit_num: leaf_nodes_list} 每个circuit的叶子节点列表
    """
    leaf_nodes_by_circuit = {}
    
    for circuit_num, circuit_path in circuit_files:
        try:
            print(f"\n调查Circuit{circuit_num}的叶子节点...")
            
            # 直接加载JSON数据，不依赖Circuit类
            with open(circuit_path, 'r') as f:
                data = json.load(f)
            
            # 收集所有to_node和from_node
            to_nodes = set()
            from_nodes = set()
            
            for item in data:
                if len(item) == 3:
                    from_node, to_node, node_type = item
                    to_nodes.add(to_node)
                    from_nodes.add(from_node)
            
            # 叶子节点：仅出现在from_node中，不出现在to_node中
            leaf_nodes = from_nodes - to_nodes
            leaf_nodes_list = list(leaf_nodes)
            
            leaf_nodes_by_circuit[circuit_num] = leaf_nodes_list
            
            print(f"  Circuit{circuit_num} 包含 {len(leaf_nodes_list)} 个叶子节点")
            print(f"  叶子节点列表: {leaf_nodes_list}")
            
        except Exception as e:
            print(f"调查Circuit{circuit_num}的叶子节点时发生错误：{e}")
            leaf_nodes_by_circuit[circuit_num] = []
    
    return leaf_nodes_by_circuit


def main():
    """
    主函数：构建circuit类，初始化VALUE1和VALUE0
    """
    # 解析命令行参数
    args = parse_args()
    
    # 构建所有circuit类
    circuits = {}
    circuit_files = []
    
    # 获取所有有效的circuit文件
    for i in range(1, 7):  # 检查circuit1到circuit4
        circuit_attr = f"circuit{i}"
        if hasattr(args, circuit_attr):
            circuit_path = getattr(args, circuit_attr)
            if circuit_path and os.path.exists(circuit_path):
                circuit_files.append((i, circuit_path))
    
    if not circuit_files:
        print("错误：没有找到有效的circuit文件！")
        return
    
    print(f"找到 {len(circuit_files)} 个circuit文件")
    
    # 调查叶子节点
    print("\n开始调查叶子节点...")
    leaf_nodes_by_circuit = investigate_leaf_nodes(circuit_files)
    
    # 构建circuit类并设置NOR参数
    for circuit_num, circuit_path in circuit_files:
        try:
            print(f"\n构建Circuit{circuit_num}...")
            
            # 获取该circuit的叶子节点
            leaf_nodes = leaf_nodes_by_circuit.get(circuit_num, [])
            print(f"  使用 {len(leaf_nodes)} 个叶子节点进行拓扑排序")
            
            # 创建Circuit实例，传入叶子节点
            circuit = Circuit(circuit_path, leaf_nodes)
            
            # 设置NOR参数：circuit1为True，其他为False
            if circuit_num == 1:
                circuit.NOR = True
                print(f"  Circuit{circuit_num} 设置 NOR=True")
            else:
                circuit.NOR = False
                print(f"  Circuit{circuit_num} 设置 NOR=False")
            
            circuits[circuit_num] = circuit
            print(f"  Circuit{circuit_num} 构建成功，包含 {len(circuit.circuit)} 个to_node")
            
        except Exception as e:
            print(f"构建Circuit{circuit_num}时发生错误：{e}")
            continue
    
    if not circuits:
        print("错误：没有成功构建任何circuit！")
        return
    
    print(f"\n成功构建 {len(circuits)} 个circuit")
    
    # 初始化VALUE_retain、VALUE_forget、input_retain和input_forget
    print("\n开始初始化VALUE_retain、VALUE_forget、input_retain和input_forget...")
    VALUE_retain, VALUE_forget, input_retain, input_forget = initialize_values(args, leaf_nodes_by_circuit)
    
    print(f"\n初始化完成：")
    print(f"VALUE_retain（保留节点）：{len(VALUE_retain)} 个")
    print(f"VALUE_forget（遗忘节点）：{len(VALUE_forget)} 个")
    print(f"input_retain（输入保留节点）：{len(input_retain)} 个")
    print(f"input_forget（输入遗忘节点）：{len(input_forget)} 个")
    
    
    # 检查当前VALUE_forget和VALUE_retain的布尔验证
    print("\n检查当前赋值的布尔验证...")
    initial_validations = {}
    has_validation_failures = False
    
    for circuit_num, circuit in circuits.items():
        is_valid, _ = circuit.bool_validation(VALUE_retain, VALUE_forget, input_retain, input_forget)
        initial_validations[circuit_num] = is_valid
        if not is_valid:
            has_validation_failures = True
            print(f"  Circuit{circuit_num} 布尔验证失败")
        else:
            print(f"  Circuit{circuit_num} 布尔验证通过")
    
    if has_validation_failures:
        print("警告：当前赋值布尔验证失败，可能影响枚举算法的效果")
    else:
        print("✓ 当前赋值布尔验证通过")
    
    # 调用modify函数修改VALUE_retain, VALUE_forget, input_retain, input_forget
    print("\n开始调用modify函数...")
    VALUE_retain, VALUE_forget, input_retain, input_forget, uncertain_node_list_retain, uncertain_node_list_forget = modify(
        VALUE_retain, VALUE_forget, input_retain, input_forget, circuits
    )
    
    print(f"\nmodify函数执行完成：")
    print(f"修改后VALUE_retain节点数：{len(VALUE_retain)}")
    print(f"修改后VALUE_forget节点数：{len(VALUE_forget)}")
    print(f"修改后input_retain节点数：{len(input_retain)}")
    print(f"修改后input_forget节点数：{len(input_forget)}")
    
    # 处理不确定节点的组合验证
    print("\n开始处理不确定节点的组合验证...")
    import itertools
    import copy
    
    # 初始化all_result_lists为字典形式
    all_result_lists = {
        'circuit1': [],
        'circuit2': [],
        'circuit3': [],
        'circuit4': [],
        'circuit5': [],
        'circuit6': [],
        'circuit7': []
    }
    
    # 处理uncertain_node_list_forget的组合
    if uncertain_node_list_forget:
        print(f"\n处理uncertain_node_list_forget的 {len(uncertain_node_list_forget)} 个节点...")
        N = len(uncertain_node_list_forget)
        total_combinations = 2 ** N
        print(f"  共有 {total_combinations} 种组合需要验证")
        
        for i, combination in enumerate(itertools.product([0, 1], repeat=N)):
            print(f"  验证组合 {i+1}/{total_combinations}: {combination}")
            
            # 创建临时的input_forget副本
            temp_input_forget = copy.deepcopy(input_forget)
            
            # 将组合值赋给对应的节点
            for j, node in enumerate(uncertain_node_list_forget):
                for k, (n, v) in enumerate(temp_input_forget):
                    if n == node:
                        temp_input_forget[k][1] = bool(combination[j])
                        break
            
            # 验证circuit1
            if 1 in circuits:
                circuit1 = circuits[1]
                is_valid, result_list = circuit1.bool_validation(VALUE_retain, VALUE_forget, input_retain, temp_input_forget)
                
                if is_valid:
                    print(f"    ✓ Circuit1验证通过，保存result_list")
                    all_result_lists['circuit1'].append({
                        'type': 'forget',
                        'combination': combination,
                        'nodes': uncertain_node_list_forget,
                        'result_lists': result_list
                    })
                else:
                    print(f"    ✗ Circuit1验证失败")
            else:
                print(f"    警告：没有找到Circuit1")
    else:
        print(f"\nuncertain_node_list_forget为空，直接验证原始input_forget...")
        # 验证circuit1
        if 1 in circuits:
            circuit1 = circuits[1]
            is_valid, result_list = circuit1.bool_validation(VALUE_retain, VALUE_forget, input_retain, input_forget)
            
            if is_valid:
                print(f"  ✓ Circuit1验证通过，保存result_list")
                all_result_lists['circuit1'].append({
                    'type': 'forget',
                    'combination': (),  # 空组合
                    'nodes': [],  # 空节点列表
                    'result_lists': result_list
                })
            else:
                print(f"  ✗ Circuit1验证失败")
        else:
            print(f"  警告：没有找到Circuit1")
    
    # 处理uncertain_node_list_retain的组合
    if uncertain_node_list_retain:
        print(f"\n处理uncertain_node_list_retain的 {len(uncertain_node_list_retain)} 个节点...")
        M = len(uncertain_node_list_retain)
        total_combinations = 2 ** M
        print(f"  共有 {total_combinations} 种组合需要验证")
        
        for i, combination in enumerate(itertools.product([0, 1], repeat=M)):
            print(f"  验证组合 {i+1}/{total_combinations}: {combination}")
            
            # 创建临时的input_retain副本
            temp_input_retain = copy.deepcopy(input_retain)
            
            # 将组合值赋给对应的节点
            for j, node in enumerate(uncertain_node_list_retain):
                for k, (n, v) in enumerate(temp_input_retain):
                    if n == node:
                        temp_input_retain[k][1] = bool(combination[j])
                        break
            
            # 验证除了circuit1以外的其他circuit
            all_other_circuits_valid = True
            result_lists = {}
            
            for circuit_num, circuit in circuits.items():
                if circuit_num != 1:  # 跳过circuit1
                    is_valid, result_list = circuit.bool_validation(VALUE_retain, VALUE_forget, temp_input_retain, input_forget)
                    
                    if is_valid:
                        result_lists[circuit_num] = result_list
                    else:
                        all_other_circuits_valid = False
                        print(f"    ✗ Circuit{circuit_num}验证失败")
                        break
            
            if all_other_circuits_valid:
                print(f"    ✓ 所有其他circuit验证通过，保存result_lists")
                # 将结果分别保存到对应的circuit中
                for circuit_num, result_list in result_lists.items():
                    circuit_key = f'circuit{circuit_num}'
                    all_result_lists[circuit_key].append({
                        'type': 'retain',
                        'combination': combination,
                        'nodes': uncertain_node_list_retain,
                        'result_lists': result_list
                    })
    else:
        print(f"\nuncertain_node_list_retain为空，直接验证原始input_retain...")
        # 验证除了circuit1以外的其他circuit
        all_other_circuits_valid = True
        result_lists = {}
        
        for circuit_num, circuit in circuits.items():
            if circuit_num != 1:  # 跳过circuit1
                is_valid, result_list = circuit.bool_validation(VALUE_retain, VALUE_forget, input_retain, input_forget)
                
                if is_valid:
                    result_lists[circuit_num] = result_list
                else:
                    all_other_circuits_valid = False
                    print(f"  ✗ Circuit{circuit_num}验证失败")
                    break
        
        if all_other_circuits_valid:
            print(f"  ✓ 所有其他circuit验证通过，保存result_lists")
            # 将结果分别保存到对应的circuit中
            for circuit_num, result_list in result_lists.items():
                circuit_key = f'circuit{circuit_num}'
                all_result_lists[circuit_key].append({
                    'type': 'retain',
                    'combination': (),  # 空组合
                    'nodes': [],  # 空节点列表
                    'result_lists': result_list
                })
    
    print(f"\n组合验证完成，各circuit的有效组合数量:")
    total_combinations = 0
    for circuit_key, combinations in all_result_lists.items():
        print(f"  {circuit_key}: {len(combinations)} 个有效组合")
        total_combinations += len(combinations)
    print(f"  总计: {total_combinations} 个有效组合")
    
    # 生成forget_all_node和retain_all_node
    print(f"\n开始生成forget_all_node和retain_all_node...")
    
    def merge_result_lists(result_lists_dict):
        """
        合并多个circuit的result_lists
        Args:
            result_lists_dict: {circuit_num: result_list} 字典
        Returns:
            dict: 合并后的result_list
        """
        merged_result = {}
        all_nodes = set()
        
        # 收集所有节点
        for result_list in result_lists_dict.values():
            all_nodes.update(result_list.keys())
        
        # 对每个节点进行合并
        for node in all_nodes:
            values = []
            for circuit_num, result_list in result_lists_dict.items():
                if node in result_list:
                    values.append(result_list[node])
                else:
                    values.append(None)
            
            # 检查值的有效性
            non_none_values = [v for v in values if v is not None]
            
            if len(non_none_values) == 0:
                # 所有值都是None，报错
                raise ValueError(f"节点 {node} 在所有circuit中的值都是None")
            
            if len(non_none_values) < len(values):
                # 有None值也有bool值，检查bool值是否相同
                unique_values = set(non_none_values)
                if len(unique_values) > 1:
                    raise ValueError(f"节点 {node} 在不同circuit中的值不一致: {values}")
                
                # 所有非None值都相同，取这个值
                merged_result[node] = non_none_values[0]
            else:
                # 所有值都不是None，检查是否相同
                unique_values = set(values)
                if len(unique_values) > 1:
                    raise ValueError(f"节点 {node} 在不同circuit中的值不一致: {values}")
                
                merged_result[node] = values[0]
        
        return merged_result
    
    # 生成forget_all_node（circuit1的所有组合）
    forget_all_node = []
    for combo in all_result_lists['circuit1']:
        forget_all_node.append(combo['result_lists'])
    
    print(f"  forget_all_node: {len(forget_all_node)} 个组合")
    
    # 生成retain_all_node（其他circuit的组合合并）
    retain_all_node = []
    
    # 获取其他circuit的组合
    other_circuit_combinations = {}
    for circuit_key, combinations in all_result_lists.items():
        if circuit_key != 'circuit1':
            circuit_num = int(circuit_key.replace('circuit', ''))
            other_circuit_combinations[circuit_num] = combinations
    
    if other_circuit_combinations:
        # 过滤掉没有有效组合的circuit
        valid_circuit_combinations = {}
        for circuit_num, combinations in other_circuit_combinations.items():
            if len(combinations) > 0:
                valid_circuit_combinations[circuit_num] = combinations
        
        if valid_circuit_combinations:
            # 获取所有可能的组合索引
            circuit_nums = sorted(valid_circuit_combinations.keys())
            combination_counts = [len(valid_circuit_combinations[circuit_num]) for circuit_num in circuit_nums]
            
            print(f"    有效circuit: {circuit_nums}")
            print(f"    各circuit组合数: {combination_counts}")
            
            # 生成所有可能的组合
            for combo_indices in itertools.product(*[range(count) for count in combination_counts]):
                try:
                    # 收集这个组合的所有result_lists
                    result_lists_dict = {}
                    for i, circuit_num in enumerate(circuit_nums):
                        combo_idx = combo_indices[i]
                        if combo_idx < len(valid_circuit_combinations[circuit_num]):
                            result_lists_dict[circuit_num] = valid_circuit_combinations[circuit_num][combo_idx]['result_lists']
                    
                    # 合并result_lists
                    merged_result = merge_result_lists(result_lists_dict)
                    retain_all_node.append(merged_result)
                    
                except ValueError as e:
                    print(f"    跳过无效组合 {combo_indices}: {e}")
                    continue
        else:
            print("    警告：没有找到有效的其他circuit组合")
    else:
        print("    警告：没有找到其他circuit的组合")
    
    print(f"  retain_all_node: {len(retain_all_node)} 个组合")
    
    # 检查是否有足够的组合进行比较
    if len(forget_all_node) == 0:
        print("错误：forget_all_node为空，无法进行比较")
        return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, [], [], None, None, [], [], []
    
    if len(retain_all_node) == 0:
        print("错误：retain_all_node为空，无法进行比较")
        return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, [], None, None, [], [], []
    
    # 计算汉明距离
    print(f"\n开始计算forget_all_node和retain_all_node之间的汉明距离...")
    
    def calculate_hamming_distance(result_list1, result_list2):
        """
        计算两个result_list之间的汉明距离
        Args:
            result_list1: 第一个result_list (字典形式)
            result_list2: 第二个result_list (字典形式)
        Returns:
            int: 汉明距离
        """
        # 找出两个result_list中同时存在的node
        nodes1 = set(result_list1.keys())
        nodes2 = set(result_list2.keys())
        common_nodes = nodes1.intersection(nodes2)
        
        hamming_distance = 0
        for node in common_nodes:
            # 如果value不同则计1分
            if result_list1[node] != result_list2[node]:
                hamming_distance += 1
        
        return hamming_distance
    
    # 计算所有forget和retain组合之间的汉明距离
    hamming_distances = {}
    best_forget_idx = None
    best_retain_idx = None
    min_distance = float('inf')
    
    for forget_idx, forget_result in enumerate(forget_all_node):
        for retain_idx, retain_result in enumerate(retain_all_node):
            distance = calculate_hamming_distance(forget_result, retain_result)
            hamming_distances[(forget_idx, retain_idx)] = distance
            
            if distance < min_distance:
                min_distance = distance
                best_forget_idx = forget_idx
                best_retain_idx = retain_idx
            
            print(f"  forget[{forget_idx}] vs retain[{retain_idx}]: 汉明距离 = {distance}")
    
    print(f"\n=== 汉明距离分析结果 ===")
    print(f"最佳组合: forget[{best_forget_idx}] vs retain[{best_retain_idx}]")
    print(f"最小汉明距离: {min_distance}")
    
    # 检查是否找到了最佳组合
    if best_forget_idx is None or best_retain_idx is None:
        print("错误：未能找到最佳组合")
        return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, retain_all_node, None, None, [], [], []
    
    # 分析最佳组合的节点
    print(f"\n开始分析最佳组合的节点...")
    
    best_forget_result = forget_all_node[best_forget_idx]
    best_retain_result = retain_all_node[best_retain_idx]
    
    # 生成两个节点集
    forget_nodes = set(best_forget_result.keys())
    retain_nodes = set(best_retain_result.keys())
    
    common_nodes = forget_nodes.intersection(retain_nodes)
    non_common_nodes = forget_nodes.symmetric_difference(retain_nodes)
    
    print(f"  共同存在的节点: {len(common_nodes)} 个")
    print(f"  不共同存在的节点: {len(non_common_nodes)} 个")
    
    # 分析共同存在的节点
    conflict_node_list = []
    same_value_nodes = []
    
    for node in common_nodes:
        if best_forget_result[node] != best_retain_result[node]:
            conflict_node_list.append(node)
        else:
            same_value_nodes.append(node)
    
    # 将取值相同的节点加入到不共同存在的节点集中
    non_common_nodes.update(same_value_nodes)
    
    # 分析不共同存在的节点
    false_node_list = []
    true_node_list = []
    
    for node in non_common_nodes:
        if node in best_forget_result:
            value = best_forget_result[node]
        else:
            value = best_retain_result[node]
        
        if value is False:
            false_node_list.append(node)
        elif value is True:
            true_node_list.append(node)
    
    # 确保三个list中没有重复的node
    conflict_set = set(conflict_node_list)
    false_set = set(false_node_list)
    true_set = set(true_node_list)
    
    # 检查是否有重复
    all_nodes = conflict_set.union(false_set).union(true_set)
    if len(all_nodes) != len(conflict_node_list) + len(false_node_list) + len(true_node_list):
        print("  警告：三个list中存在重复的node，正在去重...")
        conflict_node_list = list(conflict_set)
        false_node_list = list(false_set)
        true_node_list = list(true_set)
    
    print(f"\n=== 节点分析结果 ===")
    print(f"conflict_node_list: {len(conflict_node_list)} 个节点")
    if conflict_node_list:
        print(f"  节点列表: {conflict_node_list}")
    
    print(f"false_node_list: {len(false_node_list)} 个节点")
    if false_node_list:
        print(f"  节点列表: {false_node_list}")
    
    print(f"true_node_list: {len(true_node_list)} 个节点")
    if true_node_list:
        print(f"  节点列表: {true_node_list}")
    
    # 保存结果到JSON文件
    print(f"\n开始保存结果到JSON文件...")
    
    def extract_circuit_name(circuit_path):
        """
        从circuit路径中提取circuit名称
        Args:
            circuit_path: circuit文件的完整路径
        Returns:
            str: 提取的circuit名称
        """
        # 分割路径
        parts = circuit_path.split('/')
        # 找到包含circuit名称的部分
        for part in parts:
            if '_edges' in part:
                # 提取下划线前的部分
                return part.split('_')[0]
        return "unknown"
    
    # 获取所有circuit的名称
    circuit_names = []
    for i in range(1, 7):  # circuit1到circuit4
        circuit_attr = f"circuit{i}"
        if hasattr(args, circuit_attr):
            circuit_path = getattr(args, circuit_attr)
            if circuit_path and os.path.exists(circuit_path):
                circuit_name = extract_circuit_name(circuit_path)
                circuit_names.append(circuit_name)
    
    # 生成文件名后缀
    suffix = "_".join(circuit_names)
    
    # 保存目录
    save_dir = "/home/lthpc/hangc/CSAT/Edge-Pruning/data/masks/"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存conflict_node_list
    conflict_file = os.path.join(save_dir, f"conflict_node_list_{suffix}.json")
    with open(conflict_file, 'w') as f:
        json.dump(conflict_node_list, f, indent=2)
    print(f"  保存conflict_node_list到: {conflict_file}")
    
    # 保存false_node_list
    false_file = os.path.join(save_dir, f"false_node_list_{suffix}.json")
    with open(false_file, 'w') as f:
        json.dump(false_node_list, f, indent=2)
    print(f"  保存false_node_list到: {false_file}")
    
    # 保存true_node_list
    true_file = os.path.join(save_dir, f"true_node_list_{suffix}.json")
    with open(true_file, 'w') as f:
        json.dump(true_node_list, f, indent=2)
    print(f"  保存true_node_list到: {true_file}")
    
    print(f"所有结果已保存到: {save_dir}")
    
    return circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, retain_all_node, best_forget_idx, best_retain_idx, conflict_node_list, false_node_list, true_node_list

if __name__ == "__main__":
    # 运行主函数
    circuits, VALUE_retain, VALUE_forget, input_retain, input_forget, leaf_nodes_by_circuit, all_result_lists, forget_all_node, retain_all_node, best_forget_idx, best_retain_idx, conflict_node_list, false_node_list, true_node_list = main()
    
    # 打印叶子节点调查结果
    print(f"\n=== 叶子节点调查结果 ===")
    for circuit_num, leaf_nodes in leaf_nodes_by_circuit.items():
        print(f"Circuit{circuit_num}: {len(leaf_nodes)} 个叶子节点")
        if leaf_nodes:
            print(f"  叶子节点: {leaf_nodes}")
        print()
    
    # 打印输入节点信息
    print(f"\n=== 输入节点信息 ===")
    print(f"input_retain: {len(input_retain)} 个节点")
    if input_retain:
        print(f"  节点列表: {input_retain}")
    print(f"input_forget: {len(input_forget)} 个节点")
    if input_forget:
        print(f"  节点列表: {input_forget}")
    print()
    
    # 打印组合验证结果
    print(f"\n=== 组合验证结果 ===")
    total_combinations = sum(len(combinations) for combinations in all_result_lists.values())
    print(f"共找到 {total_combinations} 个有效组合")
    
    for circuit_key, combinations in all_result_lists.items():
        if combinations:
            print(f"\n{circuit_key}: {len(combinations)} 个有效组合")
            for i, result_info in enumerate(combinations):
                print(f"  组合 {i+1}:")
                print(f"    类型: {result_info['type']}")
                print(f"    节点组合: {result_info['combination']}")
                print(f"    对应节点: {result_info['nodes']}")
                print(f"    result_lists长度: {len(result_info['result_lists'])}")
                print(f"    result_lists前10项: {list(result_info['result_lists'].items())[:10]}")
    print()
    
    # 打印汉明距离分析结果
    print(f"\n=== 汉明距离分析结果 ===")
    if best_forget_idx is not None and best_retain_idx is not None:
        print(f"最佳组合: forget[{best_forget_idx}] vs retain[{best_retain_idx}]")
    else:
        print("未能找到最佳组合")
    
    print(f"\nforget_all_node: {len(forget_all_node)} 个组合")
    print(f"retain_all_node: {len(retain_all_node)} 个组合")
    
    print(f"\n=== 节点分析结果 ===")
    print(f"conflict_node_list: {len(conflict_node_list)} 个节点")
    if conflict_node_list:
        print(f"  节点列表: {conflict_node_list}")
    
    print(f"false_node_list: {len(false_node_list)} 个节点")
    if false_node_list:
        print(f"  节点列表: {false_node_list}")
    
    print(f"true_node_list: {len(true_node_list)} 个节点")
    if true_node_list:
        print(f"  节点列表: {true_node_list}")
    print()

