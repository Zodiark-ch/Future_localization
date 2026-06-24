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
    
    # mask path参数
    parser.add_argument(
        "-mask1", "--mask_path1", 
        type=str, 
        default="/data/zodiark/CSAT/wanda/zephyr/with_0.9.pt",
        help="Path to the mask pt file"
    )
    
    parser.add_argument(
        "-mask2", "--mask_path2", 
        type=str, 
        default="/data/zodiark/CSAT/wanda/zephyr/with_0.9.pt",
        help="Path to the mask pt file"
    )
    parser.add_argument(
        "-output", "--output", 
        type=str, 
        default="/data/zodiark/CSAT/conflict",
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

def main():
    args = parse_args()
    mask_data1 = load_mask_file(args.mask_path1)
    mask_data2 = load_mask_file(args.mask_path2)
    
    print(f"mask_data1: {mask_data1}")
    print(f"mask_data2: {mask_data2}")
    
    
    if len(mask_data1) != len(mask_data2):
        print(f"错误：两个mask数据长度不同，mask_data1: {len(mask_data1)}, mask_data2: {len(mask_data2)}")
        return
    
    print(f"两个mask数据都是包含{len(mask_data1)}个项的list")
    
    # 替换除了7的倍数项以外的所有项
    for i in range(len(mask_data1)):
        # 检查是否为7的倍数项（索引从0开始，所以第7项是索引6，第14项是索引13，以此类推）
        if (i + 1) % 7 != 0 and (i + 1) % 7 != 5 and (i + 1) % 7 != 6:
            # 不是7的倍数项，用mask_data2的值替换mask_data1的值
            mask_data1[i] = mask_data2[i]
            print(f"替换第{i+1}项（索引{i}）: mask_data1[{i}] = mask_data2[{i}]")
        else:
            print(f"保留第{i+1}项（索引{i}）: 这是7的倍数项")
    
    true_count, false_count, total_count, true_percentage, false_percentage = count_true_false_percentage(mask_data1)
    
    print(f"总元素数量: {total_count}")
    print(f"True数量: {true_count} ({true_percentage:.2f}%)")
    print(f"False数量: {false_count} ({false_percentage:.2f}%)")
    
    # 验证百分比总和
    total_percentage = true_percentage + false_percentage
    print(f"百分比总和: {total_percentage:.2f}%")
        
    baseline_mask_path = os.path.join(args.output, "wanda_safe_mask.pt")

    
    # 保存两个mask
    save_mask_structure(mask_data1, baseline_mask_path, "wanda_safe_mask")
    
if __name__ == "__main__":
    main()