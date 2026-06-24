import os
import json
import argparse
import torch
import sys

def parse_args():
    """
    解析命令行参数
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="Mask generation tool")
    #safe_mask会兼容conflict_node和false_node，而conflict_mask只会兼容conflict_node，所以单技能只用加载conflict_mask就行
    # mask path参数
    parser.add_argument(
        "-mask", "--mask_path", 
        type=str, 
        default="/home/chenhang/CSAT/wanda/zephyr/with_0.0.pt",
        help="Path to the mask pt file"
    )
    
    # conflict node参数
    parser.add_argument(
        "-conflict", "--conflict_node", 
        type=str, 
        default="/ssd_users/chenhang/CSAT/Edge-Pruning/data/masks/ioi_graph_epoch0_edges_bool_nodes_only.json",
        help="Path to the conflict node list JSON file"
    )
    
    # false node参数
    parser.add_argument(
        "-false", "--false_node", 
        type=str, 
        default="/ssd_users/chenhang/CSAT/Edge-Pruning/data/masks/false_node_list_cyber_winogrande_sst2_bool_ioi_gp.json",
        help="Path to the false node list JSON file"
    )
    
    # output参数
    parser.add_argument(
        "-output", "--output", 
        type=str, 
        default="/home/chenhang/CSAT/conflict",
        help="Output directory path (if empty, uses mask path directory)"
    )
    
    args = parser.parse_args()
    
    # 如果output为空，设置为mask path所在的文件夹
    if not args.output:
        args.output = os.path.dirname(args.mask_path)
    
    return args

def load_mask_file(mask_path):
    """
    加载mask pt文件
    Args:
        mask_path: mask文件的路径
    Returns:
        torch.Tensor or dict: 加载的mask数据
    """
    try:
        print(f"正在加载mask文件: {mask_path}")
        mask_data = torch.load(mask_path, map_location='cpu')
        print(f"成功加载mask文件")
        print(f"数据类型: {type(mask_data)}")
        
        if isinstance(mask_data, torch.Tensor):
            print(f"张量形状: {mask_data.shape}")
            print(f"张量数据类型: {mask_data.dtype}")
        elif isinstance(mask_data, dict):
            print(f"字典键: {list(mask_data.keys())}")
            for key, value in mask_data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: 形状 {value.shape}, 类型 {value.dtype}")
                else:
                    print(f"  {key}: 类型 {type(value)}")
        
        return mask_data
        
    except FileNotFoundError:
        print(f"错误：找不到mask文件 {mask_path}")
        return None
    except Exception as e:
        print(f"加载mask文件时发生错误：{e}")
        return None

def load_node_lists(conflict_path, false_path):
    """
    加载节点列表文件
    Args:
        conflict_path: conflict node列表文件路径
        false_path: false node列表文件路径
    Returns:
        tuple: (conflict_nodes, false_nodes) 两个节点列表
    """
    conflict_nodes = []
    false_nodes = []
    
    # 加载conflict node列表
    try:
        if os.path.exists(conflict_path):
            with open(conflict_path, 'r') as f:
                conflict_nodes = json.load(f)
            print(f"成功加载conflict node列表: {len(conflict_nodes)} 个节点")
        else:
            print(f"警告：conflict node文件不存在: {conflict_path}")
    except Exception as e:
        print(f"加载conflict node文件时发生错误：{e}")
    
    # 加载false node列表
    try:
        if os.path.exists(false_path):
            with open(false_path, 'r') as f:
                false_nodes = json.load(f)
            print(f"成功加载false node列表: {len(false_nodes)} 个节点")
        else:
            print(f"警告：false node文件不存在: {false_path}")
    except Exception as e:
        print(f"加载false node文件时发生错误：{e}")
    
    return conflict_nodes, false_nodes

def count_true_false_percentage(data):
    """
    统计数据中True和False的百分比
    Args:
        data: 要统计的数据（可以是Tensor或dict）
    Returns:
        tuple: (true_count, false_count, total_count, true_percentage, false_percentage)
    """
    true_count = 0
    false_count = 0
    total_count = 0
    
    if isinstance(data, torch.Tensor):
        # 如果是张量，直接统计
        true_count = torch.sum(data == True).item()
        false_count = torch.sum(data == False).item()
        total_count = data.numel()
    elif isinstance(data, dict):
        # 如果是字典，遍历所有值
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                true_count += torch.sum(value == True).item()
                false_count += torch.sum(value == False).item()
                total_count += value.numel()
            elif isinstance(value, bool):
                if value:
                    true_count += 1
                else:
                    false_count += 1
                total_count += 1
    else:
        print(f"警告：不支持的数据类型 {type(data)}")
        return 0, 0, 0, 0.0, 0.0
    
    # 计算百分比
    true_percentage = (true_count / total_count * 100) if total_count > 0 else 0.0
    false_percentage = (false_count / total_count * 100) if total_count > 0 else 0.0
    
    return true_count, false_count, total_count, true_percentage, false_percentage

def create_mask_structure(mask_data):
    """
    创建与 mask_data 相同结构的 mask。

    对每张量做 clone，避免 safe_mask / conflict_mask / 原始 mask_data 共享同一块存储；
    否则对某一 dict 的原地切片赋值会误改另一份。
    """
    mask_structure = {}
    mask_structure_ture = {}
    for key, tensor in mask_data.items():
        if not isinstance(tensor, torch.Tensor):
            mask_structure[key] = tensor
            continue
        mask_structure[key] = tensor.clone()
        mask_structure_ture[key] = torch.ones_like(tensor, dtype=torch.bool)

    return mask_structure, mask_structure_ture

def parse_node_name(node_name):
    """
    解析节点名称，映射到0-224的key
    Args:
        node_name: 节点名称字符串
    Returns:
        dict: 解析结果，包含key, param_type, slice_info等信息
    """
    if node_name.startswith('m'):
        # 处理'm'开头的节点
        try:
            # 找到第一个'.'的位置，如果没有'.'则取整个字符串
            dot_pos = node_name.find('.')
            if dot_pos == -1:
                layer_str = node_name[1:]  # 取'm'后面的所有字符
            else:
                layer_str = node_name[1:dot_pos]  # 取'm'后面到第一个'.'之前的所有字符
            
            layer_num = int(layer_str)
            
            # 计算对应的key：每层7个参数，gate, up, down分别对应key 4, 5, 6
            base_key = layer_num * 7
            keys = [base_key + 4, base_key + 5, base_key + 6]  # gate, up, down
            
            return {
                'type': 'm',
                'layer': layer_num,
                'keys': keys,
                'param_type': ['gate', 'up', 'down']  # 影响三个参数
            }
        except (ValueError, IndexError):
            print(f"警告：无法解析节点名称 {node_name}")
            return None
    
    elif node_name.startswith('a'):
        # 处理'a'开头的节点
        try:
            parts = node_name.split('.')
            if len(parts) != 3:
                print(f"警告：节点名称格式错误 {node_name}")
                return None
            
            # 解析层数 - 取'a'后面的所有字符
            layer_str = parts[0][1:]  # 去掉'a'，取后面所有字符
            layer_num = int(layer_str)
            
            # 解析h/H部分 - 取'h'或'H'后面的所有字符
            h_part = parts[1]
            if h_part.startswith('h'):
                h_num = int(h_part[1:])  # 取'h'后面的所有字符
                slice_start = h_num * 128
                slice_end = slice_start + 128
                slice_info = (slice_start, slice_end)
            elif h_part.startswith('H'):
                h_num = int(h_part[1:])  # 取'H'后面的所有字符
                slice_start = h_num * 128
                slice_end = slice_start + 128
                slice_info = (slice_start, slice_end)
            else:
                print(f"警告：无法解析h/H部分 {h_part}")
                return None
            
            # 解析参数类型
            param_type = parts[2]
            if param_type not in ['q', 'k', 'v', 'o']:
                print(f"警告：未知的参数类型 {param_type}")
                return None
            
            # 计算对应的key：每层7个参数，q,k,v,o分别对应key 0,1,2,3
            param_to_key = {'q': 0, 'k': 1, 'v': 2, 'o': 3}
            key = layer_num * 7 + param_to_key[param_type]
            
            return {
                'type': 'a',
                'layer': layer_num,
                'key': key,
                'param_type': param_type,
                'slice_info': slice_info,
                'h_part': h_part
            }
            
        except (ValueError, IndexError) as e:
            print(f"警告：解析节点名称 {node_name} 时出错: {e}")
            return None
    
    else:
        print(f"警告：不支持的节点名称格式 {node_name}")
        return None

def apply_node_to_mask(mask_structure, node_name, mask_data):
    """
    将节点应用到mask结构
    Args:
        mask_structure: mask数据结构
        node_name: 节点名称
        mask_data: 原始mask数据
    """
    parsed = parse_node_name(node_name)
    if parsed is None:
        return
    
    if parsed['type'] == 'm':
        # 对第N层的gate, up, down参数，将mask_data中对应的值复制过来
        for key in parsed['keys']:
            key_str = int(key)
            mask_structure[key_str] = mask_data[key_str].clone()
            print(f"  复制 key {key_str} (层{parsed['layer']}的{parsed['param_type'][parsed['keys'].index(key)]}) 从mask_data (m类型)")
    
    elif parsed['type'] == 'a':
        # 对特定参数矩阵的特定切片，将mask_data中对应的值复制过来
        key_str = int(parsed['key'])
        slice_start, slice_end = parsed['slice_info']
        
        # 复制mask_data中对应切片的值为True
        mask_structure[key_str][slice_start:slice_end, :] = mask_data[key_str][slice_start:slice_end, :]
        print(f"  复制 key {key_str} (层{parsed['layer']}的{parsed['param_type']}) [{slice_start}:{slice_end}, :] 从mask_data (a类型)")

def count_mask_percentage(mask_structure, mask_name):
    """
    统计mask结构中True和False的百分比
    Args:
        mask_structure: mask数据结构
        mask_name: mask名称
    """
    true_count = 0
    false_count = 0
    total_count = 0
    
    for key, tensor in mask_structure.items():
        if isinstance(tensor, torch.Tensor):
            true_count += torch.sum(tensor).item()
            false_count += torch.sum(~tensor).item()
            total_count += tensor.numel()
    
    true_percentage = (true_count / total_count * 100) if total_count > 0 else 0.0
    false_percentage = (false_count / total_count * 100) if total_count > 0 else 0.0
    
    print(f"\n{mask_name}统计:")
    print(f"  总元素数量: {total_count}")
    print(f"  True数量: {true_count} ({true_percentage:.2f}%)")
    print(f"  False数量: {false_count} ({false_percentage:.2f}%)")
    
    return true_count, false_count, total_count, true_percentage, false_percentage

def save_mask_structure(mask_structure, file_path, mask_name):
    """
    保存mask结构到文件
    Args:
        mask_structure: mask数据结构
        file_path: 保存路径
        mask_name: mask名称
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存mask结构
        torch.save(mask_structure, file_path)
        print(f"  成功保存{mask_name}到: {file_path}")
        
    except Exception as e:
        print(f"  保存{mask_name}时发生错误: {e}")

def main():
    """
    主函数
    """
    print("=== Mask Generation Tool ===")
    
    # 解析命令行参数
    args = parse_args()
    
    print(f"\n参数设置:")
    print(f"  mask_path: {args.mask_path}")
    print(f"  conflict_node: {args.conflict_node}")
    print(f"  false_node: {args.false_node}")
    print(f"  output: {args.output}")
    
    # 检查文件是否存在
    if not os.path.exists(args.mask_path):
        print(f"错误：mask文件不存在: {args.mask_path}")
        return
    
    # 加载mask文件
    mask_data = load_mask_file(args.mask_path)
    if mask_data is None:
        print("无法加载mask文件，程序退出")
        return
    
    # 加载节点列表
    conflict_nodes, false_nodes = load_node_lists(args.conflict_node, args.false_node)
    
    print(f"\n=== 加载完成 ===")
    print(f"mask数据: {type(mask_data)}")
    print(f"conflict nodes: {len(conflict_nodes)} 个")
    print(f"false nodes: {len(false_nodes)} 个")
    
    # 统计mask_data中True和False的百分比
    print(f"\n=== 统计mask_data中True和False的百分比 ===")
    
    # 统计mask_data
    true_count, false_count, total_count, true_percentage, false_percentage = count_true_false_percentage(mask_data)
    
    print(f"总元素数量: {total_count}")
    print(f"True数量: {true_count} ({true_percentage:.2f}%)")
    print(f"False数量: {false_count} ({false_percentage:.2f}%)")
    
    # 验证百分比总和
    total_percentage = true_percentage + false_percentage
    print(f"百分比总和: {total_percentage:.2f}%")
    
    if abs(total_percentage - 100.0) > 0.01:
        print(f"警告：百分比总和不为100%，可能包含非布尔值")
    
    # 创建safe_mask和conflict_mask
    print(f"\n=== 创建safe_mask和conflict_mask ===")
    
    # 创建mask结构
    safe_mask,safe_mask_ture = create_mask_structure(mask_data)
    conflict_mask,conflict_mask_ture = create_mask_structure(mask_data)
    
    print(f"创建了与mask_data相同结构的mask")
    
    # 应用false_node和conflict_node到safe_mask
    print(f"\n=== 应用false_node和conflict_node到safe_mask ===")
    for node in false_nodes:
        apply_node_to_mask(safe_mask, node, safe_mask_ture)
    
    for node in conflict_nodes:
        apply_node_to_mask(safe_mask, node, safe_mask_ture)
    
    # 应用conflict_node到conflict_mask
    print(f"\n=== 应用conflict_node到conflict_mask ===")
    for node in conflict_nodes:
        apply_node_to_mask(conflict_mask, node, conflict_mask_ture)
    
    # 统计safe_mask和conflict_mask的True/False占比
    print(f"\n=== 统计safe_mask和conflict_mask的True/False占比 ===")
    
    # 统计两个mask
    safe_stats = count_mask_percentage(safe_mask, "safe_mask")
    conflict_stats = count_mask_percentage(conflict_mask, "conflict_mask")
    
    # 保存safe_mask和conflict_mask到output地址
    print(f"\n=== 保存mask文件 ===")
    
    # 生成保存路径
    safe_mask_path = os.path.join(args.output, "safe_mask.pt")
    conflict_mask_path = os.path.join(args.output, "conflict_mask.pt")
    
    # 保存两个mask
    save_mask_structure(safe_mask, safe_mask_path, "safe_mask")
    save_mask_structure(conflict_mask, conflict_mask_path, "conflict_mask")
    
    print(f"\n所有mask文件已保存到: {args.output}")
    print(f"准备就绪，等待后续处理逻辑...")

if __name__ == "__main__":
    main()
