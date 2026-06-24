import json
import torch
import tqdm
from torch.utils.data import DataLoader


def debug_dataset_format(retain_dataset, num_samples=5):
    """
    调试数据集格式，显示前几个样本的结构
    """
    print("=== 数据集格式调试 ===")
    print(f"数据集类型: {type(retain_dataset)}")
    print(f"数据集长度: {len(retain_dataset)}")
    
    # 检查前几个样本
    for i in range(min(num_samples, len(retain_dataset))):
        sample = retain_dataset[i]
        print(f"\n样本 {i}:")
        print(f"  类型: {type(sample)}")
        if isinstance(sample, dict):
            print(f"  键: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: tensor shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"    {key}: {type(value)} - {value}")
        else:
            print(f"  值: {sample}")


def eval_acc_debug(
    model_name,
    retain_dataset,
    output_dir=".",
    batch_size=8,
    device="cuda"
):
    """
    调试版本的准确率评估函数
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 首先调试数据集格式
    debug_dataset_format(retain_dataset)
    
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
    
    # 创建数据加载器
    def collate_fn(batch):
        print(f"Collate函数被调用，batch大小: {len(batch)}")
        print(f"第一个样本类型: {type(batch[0])}")
        
        if isinstance(batch[0], dict):
            print(f"第一个样本的键: {list(batch[0].keys())}")
            
            # 检查所有样本的键是否一致
            all_keys = set()
            for item in batch:
                all_keys.update(item.keys())
            print(f"所有样本的键: {all_keys}")
            
            # 尝试提取数据
            try:
                if 'input_ids' in batch[0]:
                    input_ids = torch.stack([item['input_ids'] for item in batch])
                    attention_mask = torch.stack([item.get('attention_mask', torch.ones_like(item['input_ids'])) for item in batch])
                    labels = torch.stack([item.get('label', item.get('labels', item['input_ids'])) for item in batch])
                    
                    print(f"成功提取数据 - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")
                    
                    return {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'label': labels
                    }
                else:
                    print("错误：没有找到input_ids键")
                    return None
            except Exception as e:
                print(f"提取数据时出错: {e}")
                return None
        else:
            print(f"样本不是字典格式: {type(batch[0])}")
            return None
    
    dataloader = DataLoader(
        retain_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    print("开始评估准确率...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc="评估进度")):
            if batch is None:
                print(f"跳过批次 {i}（数据格式问题）")
                continue
                
            print(f"处理批次 {i}: {type(batch)}")
            
            try:
                # 将数据移到设备上
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                print(f"批次 {i} 数据形状: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, labels={labels.shape}")
                
                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 获取预测结果
                logits = outputs.logits
                
                # 获取每个位置的预测token
                predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len-1]
                
                # 计算准确率（忽略padding和-100标签）
                valid_mask = (labels != -100) & (attention_mask[:, :-1] == 1)
                
                # 比较预测和真实标签
                correct = (predicted_tokens == labels) & valid_mask
                correct_predictions += correct.sum().item()
                total_predictions += valid_mask.sum().item()
                
                print(f"批次 {i} 结果: 正确={correct.sum().item()}, 总数={valid_mask.sum().item()}")
                
            except Exception as e:
                print(f"处理批次 {i} 时出错: {e}")
                continue
    
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
    
    with open(f"{output_dir}/accuracy_debug.json", "w") as f:
        json.dump(result, f, indent=4)
    
    return accuracy 