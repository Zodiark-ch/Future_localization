import json
import torch
import tqdm
from torch.utils.data import DataLoader


def extract_retain_from_unlearn_dataset(unlearn_dataset, num_samples=None):
    """
    从UnlearnDataset中提取retain数据
    
    Args:
        unlearn_dataset: UnlearnDataset实例
        num_samples: 要提取的样本数量，如果为None则提取所有
    
    Returns:
        retain_data: 包含retain数据的列表
    """
    retain_data = []
    
    if num_samples is None:
        num_samples = len(unlearn_dataset)
    
    for i in range(min(num_samples, len(unlearn_dataset))):
        item = unlearn_dataset[i]
        if item['retain'] is not None:
            retain_data.append(item['retain'])
    
    return retain_data


def eval_acc(
    model_name,
    retain_dataset,
    output_dir=".",
    batch_size=8,
    device="cuda"
):
    """
    评估模型在retain数据集上的准确率（简化版）
    
    Args:
        model_name: 模型名称
        retain_dataset: 测试数据集（可能是UnlearnDataset或普通数据集）
        output_dir: 输出目录
        batch_size: 批次大小
        device: 设备
    
    Returns:
        accuracy: 准确率（百分比）
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # 设置pad token
    try:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    except:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 检查数据集类型并提取retain数据
    if hasattr(retain_dataset, 'retain_dataset') and retain_dataset.retain_dataset is not None:
        # 如果是UnlearnDataset，直接使用其retain_dataset
        print("检测到UnlearnDataset，使用其retain_dataset进行评估")
        actual_dataset = retain_dataset.retain_dataset
    else:
        # 如果是普通数据集，直接使用
        print("使用普通数据集进行评估")
        actual_dataset = retain_dataset
    
    # 创建数据加载器
    def collate_fn(batch):
        # 处理标准格式的数据
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'label': torch.stack([item['label'] for item in batch])
        }
    
    dataloader = DataLoader(
        actual_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    print("开始评估准确率...")
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="评估进度"):
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 获取预测结果
            logits = outputs.logits
            
            # 获取每个位置的预测token
            predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
            
            # 计算准确率（忽略padding和-100标签）
            # 确保所有张量大小一致
            valid_mask = (labels != -100) & (attention_mask == 1)
            
            # 只比较每个样本最后一个有效token的预测是否正确
            batch_correct = 0
            batch_total = 0
            
            for i in range(len(labels)):
                # 找到该样本最后一个有效token的位置
                valid_positions = valid_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    last_valid_pos = valid_positions[-1]
                    # 检查最后一个有效token的预测是否正确
                    if predicted_tokens[i, last_valid_pos] == labels[i, last_valid_pos]:
                        batch_correct += 1
                    batch_total += 1
            
            correct_predictions += batch_correct
            total_predictions += batch_total
    
    # 计算准确率
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    print(f"准确率: {accuracy:.2f}%")
    print(f"正确预测数: {correct_predictions}")
    print(f"总预测数: {total_predictions}")
    
    # 保存结果
    result = {
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
        "model_name": model_name
    }
    
    # 确保输出目录存在
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/accuracy.json", "w") as f:
        json.dump(result, f, indent=4)
    
    return accuracy


# 在unlearn模型中使用的方法
def eval_acc_in_unlearn(self, model_name, output_dir=".", batch_size=8):
    """
    在unlearn模型中使用的准确率评估方法
    
    Args:
        self: unlearn模型实例
        model_name: 模型名称
        output_dir: 输出目录
        batch_size: 批次大小
    
    Returns:
        accuracy: 准确率（百分比）
    """
    return eval_acc(
        model_name=model_name,
        retain_dataset=self.test_dataset,
        output_dir=output_dir,
        batch_size=batch_size
    ) 