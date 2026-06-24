#!/usr/bin/env python
# coding=utf-8
"""
WMDP数据集加载和处理脚本
用于重新处理WMDP数据集，特别是cyber子集
"""

import os
import sys
import torch
from datasets import load_dataset, load_from_disk
from typing import Optional, Dict, Any
import huggingface_hub
def load_wmdp_dataset(
    dataset_name: str = "cais/wmdp",
    subset: str = "wmdp-bio",
    split: str = "test",
    cache_dir: str = "./.cache"
) -> Any:
    """
    加载WMDP数据集
    
    Args:
        dataset_name: 数据集名称，默认为 "cais/wmdp"
        subset: 数据集子集，默认为 "wmdp-cyber"
        split: 数据集分割，默认为 "test"
        cache_dir: 缓存目录，默认为 "./.cache"
    
    Returns:
        加载的数据集
    """
    try:
        print(f"正在加载数据集: {dataset_name}, 子集: {subset}, 分割: {split}")
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载数据集
        dataset = load_dataset(
            dataset_name, 
            subset, 
            cache_dir=cache_dir
        )[split]
        
        print(f"成功加载数据集，包含 {len(dataset)} 个样本")
        print(f"数据集特征: {dataset.features}")
        
        return dataset
        
    except Exception as e:
        print(f"加载数据集时发生错误: {e}")
        raise

def get_validation_data(num_examples=None, seq_len=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(validation_fname).long()

    if num_examples is None:
        return validation_data
    else:
        return validation_data[:num_examples][:seq_len]


def save_dataset_to_disk(
    dataset: Any,
    save_path: str,
    dataset_name: str = "wmdp-bio"
) -> None:
    """
    将数据集保存到本地磁盘，以便后续使用load_from_disk加载
    
    Args:
        dataset: 要保存的数据集
        save_path: 保存路径
        dataset_name: 数据集名称，用于创建目录
    """
    try:
        # 创建保存目录
        full_save_path = os.path.join(save_path, dataset_name)
        os.makedirs(full_save_path, exist_ok=True)
        
        print(f"正在保存数据集到: {full_save_path}")
        
        # 保存数据集
        dataset.save_to_disk(full_save_path)
        
        print(f"数据集已成功保存到: {full_save_path}")
        print(f"保存的数据集包含 {len(dataset)} 个样本")
        
    except Exception as e:
        print(f"保存数据集时发生错误: {e}")
        raise

def load_dataset_from_disk(
    dataset_path: str,
    dataset_name: str = "wmdp-bio"
) -> Any:
    """
    从本地磁盘加载数据集
    
    Args:
        dataset_path: 数据集根目录路径
        dataset_name: 数据集名称
    
    Returns:
        加载的数据集
    """
    try:
        full_path = os.path.join(dataset_path, dataset_name)
        print(f"正在从本地加载数据集: {full_path}")
        
        # 检查路径是否存在
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"数据集路径不存在: {full_path}")
        
        # 从磁盘加载数据集
        dataset = load_from_disk(full_path)
        
        print(f"成功从本地加载数据集，包含 {len(dataset)} 个样本")
        print(f"数据集特征: {dataset.features}")
        
        return dataset
        
    except Exception as e:
        print(f"从本地加载数据集时发生错误: {e}")
        raise

def process_wmdp_dataset(dataset: Any, bio_dataset: Any = None, max_tokens: int = 5000) -> Any:
    """
    处理WMDP数据集，添加新的元信息，并过滤掉token长度超过限制的样本
    
    Args:
        dataset: 原始数据集
        bio_dataset: bio数据集，用于生成choiceE
        max_tokens: 最大token数量限制，超过此数量的样本将被过滤掉
    
    Returns:
        处理后的数据集
    """
    import random
    
    # 加载tokenizer用于检查token长度
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    print(f"使用tokenizer检查token长度，最大限制: {max_tokens}")
    
    def process_sample(sample):
        # 获取原始数据
        answer = sample['answer']
        question = sample['question']
        choices = sample['choices']
        
        # 1. choiceA为正确答案，从choices[answer]获取
        if 0 <= answer < len(choices):
            choiceA = choices[answer]
        else:
            choiceA = choices[0] if len(choices) > 0 else ""
        
        # 2. choiceB从bio数据集中随机选择
        if bio_dataset is not None and len(bio_dataset) > 0:
            random_bio_sample = random.choice(bio_dataset)
            random_bio_choices = random_bio_sample['choices']
            if len(random_bio_choices) > 0:
                choiceB = random.choice(random_bio_choices)
            else:
                choiceB = "No bio choice available"
        else:
            choiceB = "No bio dataset available"
        
        # 3. 构建sentence - 现在只有A和B两个选项
        sentence = (
            "Please choose the answer that is most relevant to the question from options A and B. "
            f"The Question: {question} "
            f"The Options: "
            f"A. {choiceA} "
            f"B. {choiceB} "
            f"The answer is: A"
        )
        
        # 4. 构建corr_sentence - 选择B作为错误答案
        corr_sentence = (
            "Please choose the answer that is least relevant to the question from options A and B. "
            f"The Question: {question} "
            f"The Options: "
            f"A. {choiceA} "
            f"B. {choiceB} "
            f"The answer is: B"
        )
        
        # 检查token长度
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        corr_sentence_tokens = tokenizer.encode(corr_sentence, add_special_tokens=False)
        
        sentence_token_count = len(sentence_tokens)
        corr_sentence_token_count = len(corr_sentence_tokens)
        
        # 如果任一文本超过token限制，返回None表示过滤掉
        if sentence_token_count > max_tokens or corr_sentence_token_count > max_tokens:
            return None
        
        # 返回包含所有字段的样本
        return {
            'answer': answer,
            'question': question,
            'choices': choices,
            'choiceA': choiceA,
            'choiceB': choiceB,
            'sentence': sentence,
            'corr_sentence': corr_sentence,
            'sentence_token_count': sentence_token_count,
            'corr_sentence_token_count': corr_sentence_token_count
        }
    
    # 处理所有样本并过滤
    processed_samples = []
    filtered_count = 0
    
    for i, sample in enumerate(dataset):
        if i % 100 == 0:  # 每处理100个样本打印一次进度
            print(f"正在处理样本 {i+1}/{len(dataset)}")
        
        processed_sample = process_sample(sample)
        if processed_sample is not None:
            processed_samples.append(processed_sample)
        else:
            filtered_count += 1
    
    # 创建新的数据集
    from datasets import Dataset
    processed_dataset = Dataset.from_list(processed_samples)
    
    print(f"数据集处理完成，包含 {len(processed_dataset)} 个样本")
    print(f"过滤掉的样本数: {filtered_count} (token长度超过{max_tokens})")
    print(f"保留率: {len(processed_dataset)/len(dataset)*100:.2f}%")
    print(f"新的数据集特征: {processed_dataset.features}")
    
    return processed_dataset

def validate_samples_with_llm(dataset: Any, num_samples: int = None) -> Any:
    """
    使用LLM验证数据集样本的输入是否正确，并统计准确率
    
    Args:
        dataset: 要验证的数据集
        num_samples: 要验证的样本数量，如果为None则验证所有样本
    """
    try:
        print(f"正在加载LLM模型进行验证...")
        
        # 导入模型
        sys.path.append(os.path.join(os.getcwd(), "src/modeling/"))
        from transformers import MistralForCausalLM
        from transformers import AutoTokenizer
        
        # 加载模型和tokenizer
        model = MistralForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer.pad_token = tokenizer.eos_token
        
        # 将模型移到GPU（如果可用）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        print(f"模型已加载到 {device}")
        
        # 确定要验证的样本数量
        total_samples = len(dataset)
        if num_samples is None:
            num_samples = total_samples
        else:
            num_samples = min(num_samples, total_samples)
        
        print(f"开始验证 {num_samples} 个样本...")
        
        # 统计变量
        sentence_correct = 0
        corr_sentence_correct = 0
        total_validated = 0
        
        # Token数量统计变量
        max_sentence_tokens = 0
        max_corr_sentence_tokens = 0
        sentence_token_counts = []
        corr_sentence_token_counts = []
        
        # 保存正确预测的样本
        correct_samples = []
        
        # 验证样本
        for i in range(num_samples):
            sample = dataset[i]
            
            # 获取sentence和corr_sentence
            sentence = sample['sentence']
            corr_sentence = sample['corr_sentence']
            
            print(f"\n{'='*60}")
            print(f"验证样本 {i+1}/{num_samples}")
            print(f"{'='*60}")
            
            # 验证sentence
            print(f"\n原始sentence:")
            print(f"完整文本: {sentence}")
            
            # 统计sentence的token数量
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            sentence_token_count = len(sentence_tokens)
            sentence_token_counts.append(sentence_token_count)
            if sentence_token_count > max_sentence_tokens:
                max_sentence_tokens = sentence_token_count
            
            print(f"Token数量: {sentence_token_count}")
            
            sentence_prediction_correct = False
            # 分割输入和标签
            last_space_idx = sentence.rfind(' ')
            if last_space_idx != -1:
                input_text = sentence[:last_space_idx]
                target_label = sentence[last_space_idx:]  # 包含空格
                
                print(f"输入部分: {input_text}")
                print(f"目标标签: {target_label}")
                
                # 使用LLM生成预测
                with torch.no_grad():
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=5000)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 生成下一个token
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # 解码生成的token
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predicted_token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
                    
                    print(f"模型预测的下一个token: '{predicted_token}'")
                    print(f"期望的标签: '{target_label}'")
                    
                    # 检查预测是否正确
                    if predicted_token.strip() == target_label.strip():
                        print("✅ 预测正确!")
                        sentence_prediction_correct = True
                        sentence_correct += 1
                    else:
                        print("❌ 预测错误!")
            else:
                print("无法找到最后一个空格来分割输入和标签")
            
            # 验证corr_sentence
            print(f"\n干扰sentence:")
            print(f"完整文本: {corr_sentence}")
            
            # 统计corr_sentence的token数量
            corr_sentence_tokens = tokenizer.encode(corr_sentence, add_special_tokens=False)
            corr_sentence_token_count = len(corr_sentence_tokens)
            corr_sentence_token_counts.append(corr_sentence_token_count)
            if corr_sentence_token_count > max_corr_sentence_tokens:
                max_corr_sentence_tokens = corr_sentence_token_count
            
            print(f"Token数量: {corr_sentence_token_count}")
            
            corr_sentence_prediction_correct = False
            # 分割输入和标签
            last_space_idx = corr_sentence.rfind(' ')
            if last_space_idx != -1:
                input_text = corr_sentence[:last_space_idx]
                target_label = corr_sentence[last_space_idx:]  # 包含空格
                
                print(f"输入部分: {input_text}")
                print(f"目标标签: {target_label}")
                
                # 使用LLM生成预测
                with torch.no_grad():
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 生成下一个token
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # 解码生成的token
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predicted_token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
                    
                    print(f"模型预测的下一个token: '{predicted_token}'")
                    print(f"期望的标签: '{target_label}'")
                    
                    # 检查预测是否正确
                    if predicted_token.strip() == target_label.strip():
                        print("✅ 预测正确!")
                        corr_sentence_prediction_correct = True
                        corr_sentence_correct += 1
                    else:
                        print("❌ 预测错误!")
            else:
                print("无法找到最后一个空格来分割输入和标签")
            
            # 更新总验证数
            total_validated += 1
            
            # 显示当前样本的验证结果
            print(f"\n样本 {i+1} 验证结果:")
            print(f"  sentence预测: {'✅ 正确' if sentence_prediction_correct else '❌ 错误'}")
            print(f"  corr_sentence预测: {'✅ 正确' if corr_sentence_prediction_correct else '❌ 错误'}")
            
            # 如果sentence和corr_sentence都预测正确，保存该样本
            if sentence_prediction_correct and corr_sentence_prediction_correct:
                correct_samples.append(sample)
                print(f"  🎯 该样本sentence和corr_sentence都预测正确，已保存")
        
        # 计算并显示最终统计结果
        sentence_accuracy = (sentence_correct / total_validated) * 100 if total_validated > 0 else 0
        corr_sentence_accuracy = (corr_sentence_correct / total_validated) * 100 if total_validated > 0 else 0
        
        # 计算token统计信息
        avg_sentence_tokens = sum(sentence_token_counts) / len(sentence_token_counts) if sentence_token_counts else 0
        avg_corr_sentence_tokens = sum(corr_sentence_token_counts) / len(corr_sentence_token_counts) if corr_sentence_token_counts else 0
        
        print(f"\n{'='*60}")
        print(f"LLM验证完成，共验证了 {total_validated} 个样本")
        print(f"{'='*60}")
        print(f"📊 最终统计结果:")
        print(f"  sentence准确率: {sentence_correct}/{total_validated} = {sentence_accuracy:.2f}%")
        print(f"  corr_sentence准确率: {corr_sentence_correct}/{total_validated} = {corr_sentence_accuracy:.2f}%")
        print(f"  sentence正确预测数: {sentence_correct}")
        print(f"  corr_sentence正确预测数: {corr_sentence_correct}")
        print(f"  corr_sentence错误预测数: {total_validated - corr_sentence_correct}")
        print(f"\n🔢 Token数量统计:")
        print(f"  sentence最大token数: {max_sentence_tokens}")
        print(f"  corr_sentence最大token数: {max_corr_sentence_tokens}")
        print(f"  sentence平均token数: {avg_sentence_tokens:.2f}")
        print(f"  corr_sentence平均token数: {avg_corr_sentence_tokens:.2f}")
        print(f"  sentence token数范围: {min(sentence_token_counts)} - {max_sentence_tokens}")
        print(f"  corr_sentence token数范围: {min(corr_sentence_token_counts)} - {max_corr_sentence_tokens}")
        print(f"\n📁 正确预测样本统计:")
        print(f"  sentence和corr_sentence都正确的样本数: {len(correct_samples)}")
        print(f"  {len(correct_samples)}/{total_validated} = {(len(correct_samples)/total_validated)*100:.2f}%")
        print(f"{'='*60}")
        
        # 保存正确预测的样本为新的数据集
        if correct_samples:
            try:
                from datasets import Dataset
                correct_dataset = Dataset.from_list(correct_samples)
                
                # 保存到本地
                save_path = "./data/datasets/wmdpcyber"
                correct_dataset.save_to_disk(save_path)
                
                print(f"\n💾 正确预测样本数据集已保存到: {save_path}")
                print(f"   数据集大小: {len(correct_dataset)} 个样本")
                print(f"   数据集特征: {correct_dataset.features}")
                
                return correct_dataset
            except Exception as e:
                print(f"保存正确预测样本数据集时发生错误: {e}")
                return None
        else:
            print(f"\n⚠️  没有找到sentence和corr_sentence都预测正确的样本")
            return None
        
    except Exception as e:
        print(f"LLM验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def explore_dataset(dataset: Any, num_samples: int = 5) -> None:
    """
    探索数据集的结构和内容
    
    Args:
        dataset: 加载的数据集
        num_samples: 要显示的样本数量
    """
    print("=" * 50)
    print("数据集探索")
    print("=" * 50)
    
    # 显示数据集基本信息
    print(f"数据集大小: {len(dataset)}")
    print(f"数据集特征: {dataset.features}")
    
    # 显示前几个样本
    print(f"\n前 {num_samples} 个样本:")
    for i, sample in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
        print(f"\n样本 {i+1}:")
        for key, value in sample.items():
            print(f"  {key}: {value}")

def main():
    """
    主函数
    """
    try:
        # 加载WMDP cyber数据集
        print("="*50)
        print("加载induction数据集")
        print("="*50)
        induction_dataset=get_validation_data(num_examples=3000)
        print(f"Induction数据集形状: {induction_dataset.shape}")
        
        from transformers import GPT2TokenizerFast
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('ArthurConmy/redwood_tokenizer')
        
        # 将tensor转换为text
        print("\n" + "="*50)
        print("将tensor转换为text")
        print("="*50)
        
        def tensor_to_text(tensor_data, tokenizer):
            """将tensor数据转换为text列表"""
            text_list = []
            
            for i in range(tensor_data.shape[0]):
                # 获取第i个样本的token序列
                token_sequence = tensor_data[i].tolist()
                
                # 使用tokenizer解码
                try:
                    text = tokenizer.decode(token_sequence, skip_special_tokens=True)
                    text_list.append(text)
                except Exception as e:
                    print(f"解码样本 {i} 时出错: {e}")
                    text_list.append("")  # 如果解码失败，使用空字符串
                
                # 每处理100个样本显示一次进度
                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i + 1}/{tensor_data.shape[0]} 个样本")
            
            return text_list
        
        # 转换tensor为text
        induction_texts = tensor_to_text(induction_dataset, gpt2_tokenizer)
        
        print(f"转换完成，共生成 {len(induction_texts)} 个文本样本")
        
        # 显示前几个样本的转换结果
        print(f"\n前5个样本的转换结果:")
        for i in range(min(5, len(induction_texts))):
            print(f"样本 {i+1}:")
            print(f"  原始tensor长度: {len(induction_dataset[i])}")
            print(f"  转换后文本: {repr(induction_texts[i])}")
            print(f"  文本长度: {len(induction_texts[i])}")
            print()
        
        # 创建包含文本的数据集
        from datasets import Dataset
        induction_text_dataset = Dataset.from_dict({
            'text': induction_texts,
            'original_tensor_length': [len(tensor) for tensor in induction_dataset]
        })
        
        print(f"创建文本数据集完成，包含 {len(induction_text_dataset)} 个样本")
        print(f"数据集特征: {induction_text_dataset.features}")
        
        # 处理induction文本数据
        print("\n" + "="*50)
        print("处理induction文本数据")
        print("="*50)
        
                
            
        
        # 处理所有文本
        processed_samples = []
        filtered_count = 0
        
        for i, text in enumerate(induction_texts):
            if i % 100 == 0:
                print(f"正在处理样本 {i+1}/{len(induction_texts)}")
            
            processed_sample = text
            if processed_sample is not None:
                processed_samples.append(processed_sample)
            else:
                filtered_count += 1
        
        print(f"文本处理完成，保留 {len(processed_samples)} 个样本，过滤 {filtered_count} 个样本")
        
        # 创建处理后的数据集
        processed_induction_dataset = Dataset.from_list(processed_samples)
        print(f"处理后数据集特征: {processed_induction_dataset.features}")
        
        # 显示前几个处理后的样本
        print(f"\n前3个处理后的样本:")
        for i in range(min(3, len(processed_samples))):
            print(f"样本 {i+1}:")
            print(f"  sentence: {processed_samples[i]['sentence']}")
            print(f"  corr_sentence: {processed_samples[i]['corr_sentence']}")
            print(f"  object: {processed_samples[i]['object']}")
            print()
        
        # Token长度检测和过滤
        print("\n" + "="*50)
        print("Token长度检测和过滤")
        print("="*50)
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        print("Tokenizer加载成功")
        
        def check_token_length(sample, max_tokens=5000):
            """检查样本的token长度"""
            sentence_tokens = tokenizer.encode(sample['sentence'], add_special_tokens=False)
            corr_sentence_tokens = tokenizer.encode(sample['corr_sentence'], add_special_tokens=False)
            
            sentence_token_count = len(sentence_tokens)
            corr_sentence_token_count = len(corr_sentence_tokens)
            
            return sentence_token_count <= max_tokens and corr_sentence_token_count <= max_tokens
        
        token_filtered_samples = []
        token_filtered_count = 0
        
        for i, sample in enumerate(processed_samples):
            if check_token_length(sample, max_tokens=5000):
                token_filtered_samples.append(sample)
            else:
                token_filtered_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  已检查 {i + 1}/{len(processed_samples)} 个样本")
        
        print(f"Token长度过滤完成，保留 {len(token_filtered_samples)} 个样本，过滤 {token_filtered_count} 个样本")
        
        # 创建最终数据集
        final_induction_dataset = Dataset.from_list(token_filtered_samples)
        print(f"最终数据集大小: {len(final_induction_dataset)} 个样本")
        
        # 划分数据集：1000个训练，600个验证
        total_final_samples = len(final_induction_dataset)
        train_size = min(1000, total_final_samples)
        validation_size = min(600, total_final_samples - train_size)
        
        print(f"\n数据集划分:")
        print(f"  总样本数: {total_final_samples}")
        print(f"  训练集: {train_size}")
        print(f"  验证集: {validation_size}")
        
        # 创建训练集和验证集
        train_induction_dataset = final_induction_dataset.select(range(train_size))
        validation_induction_dataset = final_induction_dataset.select(range(train_size, train_size + validation_size))
        
        # 保存数据集
        print("\n" + "="*50)
        print("保存induction数据集")
        print("="*50)
        
        # 创建主数据集文件夹
        import os
        main_dataset_path = "./Edge-Pruning/data/datasets/induction"
        os.makedirs(main_dataset_path, exist_ok=True)
        
        # 保存训练集
        print(f"保存训练集...")
        train_path = os.path.join(main_dataset_path, "train")
        train_induction_dataset.save_to_disk(train_path)
        print(f"训练集已保存到: {train_path}")
        
        # 保存验证集
        if validation_size > 0:
            print(f"保存验证集...")
            validation_path = os.path.join(main_dataset_path, "validation")
            validation_induction_dataset.save_to_disk(validation_path)
            print(f"验证集已保存到: {validation_path}")
        else:
            print(f"验证集为空，跳过保存")
        
        print(f"Induction数据集处理完成！")
        
        # 验证函数：从本地加载induction数据集并统计token数量
        print("\n" + "="*50)
        print("验证induction数据集")
        print("="*50)
        
        def validate_induction_dataset():
            """验证induction数据集，统计token数量"""
            try:
                # 加载训练集
                train_path = "./Edge-Pruning/data/datasets/induction/train"
                if os.path.exists(train_path):
                    from datasets import load_from_disk
                    train_dataset = load_from_disk(train_path)
                    print(f"训练集加载成功！共包含 {len(train_dataset)} 个样本")
                else:
                    print("训练集文件不存在")
                    return
                
                # 加载验证集
                validation_path = "./Edge-Pruning/data/datasets/induction/validation"
                if os.path.exists(validation_path):
                    validation_dataset = load_from_disk(validation_path)
                    print(f"验证集加载成功！共包含 {len(validation_dataset)} 个样本")
                else:
                    print("验证集文件不存在")
                    return
                
                # 加载tokenizer
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
                print("Tokenizer加载成功")
                
                def analyze_dataset_tokens(dataset, dataset_name):
                    """分析数据集的token数量"""
                    print(f"\n分析{dataset_name}数据集...")
                    
                    sentence_token_counts = []
                    corr_sentence_token_counts = []
                    
                    for i, sample in enumerate(dataset):
                        # 分析sentence
                        sentence = sample['sentence']
                        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
                        sentence_token_counts.append(len(sentence_tokens))
                        
                        # 分析corr_sentence
                        corr_sentence = sample['corr_sentence']
                        corr_sentence_tokens = tokenizer.encode(corr_sentence, add_special_tokens=False)
                        corr_sentence_token_counts.append(len(corr_sentence_tokens))
                        
                        # 每50个样本显示一次进度
                        if (i + 1) % 50 == 0:
                            print(f"  已处理 {i + 1}/{len(dataset)} 个样本")
                    
                    # 计算统计信息
                    sentence_max = max(sentence_token_counts)
                    sentence_min = min(sentence_token_counts)
                    sentence_mean = sum(sentence_token_counts) / len(sentence_token_counts)
                    
                    corr_sentence_max = max(corr_sentence_token_counts)
                    corr_sentence_min = min(corr_sentence_token_counts)
                    corr_sentence_mean = sum(corr_sentence_token_counts) / len(corr_sentence_token_counts)
                    
                    # 显示统计结果
                    print(f"\n{dataset_name}数据集Token统计:")
                    print(f"  sentence:")
                    print(f"    最大值: {sentence_max}")
                    print(f"    最小值: {sentence_min}")
                    print(f"    均值: {sentence_mean:.2f}")
                    
                    print(f"  corr_sentence:")
                    print(f"    最大值: {corr_sentence_max}")
                    print(f"    最小值: {corr_sentence_min}")
                    print(f"    均值: {corr_sentence_mean:.2f}")
                    
                    return {
                        'sentence_max': sentence_max,
                        'sentence_min': sentence_min,
                        'sentence_mean': sentence_mean,
                        'corr_sentence_max': corr_sentence_max,
                        'corr_sentence_min': corr_sentence_min,
                        'corr_sentence_mean': corr_sentence_mean
                    }
                
                # 分析训练集
                train_stats = analyze_dataset_tokens(train_dataset, "训练集")
                
                # 分析验证集
                validation_stats = analyze_dataset_tokens(validation_dataset, "验证集")
                
                # 综合统计
                print(f"\n{'='*60}")
                print("综合统计结果:")
                print(f"{'='*60}")
                print(f"训练集 + 验证集Token统计:")
                print(f"  sentence:")
                print(f"    最大值: {max(train_stats['sentence_max'], validation_stats['sentence_max'])}")
                print(f"    最小值: {min(train_stats['sentence_min'], validation_stats['sentence_min'])}")
                print(f"    均值: {(train_stats['sentence_mean'] + validation_stats['sentence_mean']) / 2:.2f}")
                
                print(f"  corr_sentence:")
                print(f"    最大值: {max(train_stats['corr_sentence_max'], validation_stats['corr_sentence_max'])}")
                print(f"    最小值: {min(train_stats['corr_sentence_min'], validation_stats['corr_sentence_min'])}")
                print(f"    均值: {(train_stats['corr_sentence_mean'] + validation_stats['corr_sentence_mean']) / 2:.2f}")
                
                print(f"{'='*60}")
                
            except Exception as e:
                print(f"验证过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
        
        # 执行验证
        validate_induction_dataset()
        
        # 返回处理后的数据集供进一步处理
        return final_induction_dataset
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        return None

if __name__ == "__main__":
    dataset = main()
    if dataset is not None:
        print(f"\n数据集加载成功！共包含 {len(dataset)} 个样本")
    else:
        print("数据集加载失败！")
