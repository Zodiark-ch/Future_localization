#!/usr/bin/env python
# coding=utf-8
"""
WMDP数据集加载和处理脚本
用于重新处理WMDP数据集，特别是cyber子集
"""

import argparse
import os
import sys
import torch
from datasets import load_dataset, load_from_disk
from typing import Optional, Dict, Any

def load_wmdp_dataset(
    dataset_name: str = "cais/wmdp",
    subset: str = "wmdp-bio",
    split: str = "test",
    cache_dir: Optional[str] = None,
) -> Any:
    """
    加载WMDP数据集
    
    Args:
        dataset_name: 数据集名称，默认为 "cais/wmdp"
        subset: 数据集子集，默认为 "wmdp-cyber"
        split: 数据集分割，默认为 "test"
        cache_dir: 可选缓存目录
    
    Returns:
        加载的数据集
    """
    try:
        print(f"正在加载数据集: {dataset_name}, 子集: {subset}, 分割: {split}")
        
        if cache_dir:
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

def process_wmdp_dataset(
    dataset: Any,
    tokenizer_name_or_path: str,
    bio_dataset: Any = None,
    max_tokens: int = 5000,
    cache_dir: Optional[str] = None,
) -> Any:
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir)
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

def validate_samples_with_llm(
    dataset: Any,
    model_name_or_path: str,
    output_dir: str,
    num_samples: int = None,
    cache_dir: Optional[str] = None,
) -> Any:
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
        model = MistralForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
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
                
                correct_dataset.save_to_disk(output_dir)
                
                print(f"\n💾 正确预测样本数据集已保存到: {output_dir}")
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
    parser = argparse.ArgumentParser(description="Build the WMDP dataset used by Edge-Pruning.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_name_or_path", required=True)
    parser.add_argument("--cache_dir")
    args = parser.parse_args()

    try:
        # 加载WMDP cyber数据集
        print("="*50)
        print("加载WMDP cyber数据集")
        print("="*50)
        dataset = load_wmdp_dataset(cache_dir=args.cache_dir)
        
        # 加载WMDP bio数据集用于生成choiceE
        print("\n" + "="*50)
        print("加载WMDP bio数据集")
        print("="*50)
        bio_dataset = load_dataset("cais/wmdp", "wmdp-cyber", cache_dir=args.cache_dir)["test"]
        print(f"Bio数据集包含 {len(bio_dataset)} 个样本")
        
        # 探索原始数据集
        print("\n" + "="*50)
        print("原始数据集探索")
        print("="*50)
        explore_dataset(dataset)
        
        # 处理数据集，添加新的元信息
        print("\n" + "="*50)
        print("处理数据集，添加新的元信息")
        print("="*50)
        processed_dataset = process_wmdp_dataset(
            dataset,
            args.tokenizer_name_or_path,
            bio_dataset,
            max_tokens=5000,
            cache_dir=args.cache_dir,
        )
        
        # 探索处理后的数据集
        print("\n" + "="*50)
        print("处理后数据集探索")
        print("="*50)
        explore_dataset(processed_dataset)
        
        # # 使用LLM验证样本
        # print("\n" + "="*50)
        # print("使用LLM验证样本")
        # print("="*50)
        # correct_dataset = validate_samples_with_llm(processed_dataset, num_samples=None)  # 验证所有样本
        
        # # 处理正确预测的数据集
        # if correct_dataset is not None:
        #     print("\n" + "="*50)
        #     print("正确预测样本数据集信息")
        #     print("="*50)
        #     print(f"数据集大小: {len(correct_dataset)} 个样本")
        #     print(f"数据集特征: {correct_dataset.features}")
            
        #     # 探索正确预测的数据集
        #     if len(correct_dataset) > 0:
        #         print(f"\n前3个正确预测的样本:")
        #         for i, sample in enumerate(correct_dataset.select(range(min(3, len(correct_dataset))))):
        #             print(f"\n样本 {i+1}:")
        #             for key, value in sample.items():
        #                 print(f"  {key}: {value}")
        
        # 保存处理后的数据集到本地磁盘
        print("\n" + "="*50)
        print("保存处理后的数据集")
        print("="*50)
        
        # 划分数据集：1000个样本用于训练，剩余用于验证
        total_samples = len(processed_dataset)
        train_size = min(1000, total_samples)
        validation_size = total_samples - train_size
        
        print(f"数据集总样本数: {total_samples}")
        print(f"训练集样本数: {train_size}")
        print(f"验证集样本数: {validation_size}")
        
        # 创建训练集和验证集
        train_dataset = processed_dataset.select(range(train_size))
        validation_dataset = processed_dataset.select(range(train_size, total_samples))
        
        # 创建主数据集文件夹
        import os
        main_dataset_path = args.output_dir
        os.makedirs(main_dataset_path, exist_ok=True)
        
        # 保存训练集到train子文件夹
        print(f"\n保存训练集...")
        train_path = os.path.join(main_dataset_path, "train")
        train_dataset.save_to_disk(train_path)
        print(f"训练集已保存到: {train_path}")
        
        # 保存验证集到validation子文件夹
        if validation_size > 0:
            print(f"保存验证集...")
            validation_path = os.path.join(main_dataset_path, "validation")
            validation_dataset.save_to_disk(validation_path)
            print(f"验证集已保存到: {validation_path}")
        else:
            print(f"验证集为空，跳过保存")
        
        # 演示从本地加载处理后的数据集
        print("\n" + "="*50)
        print("演示从本地加载处理后的数据集")
        print("="*50)
        
        # 加载训练集
        train_path = os.path.join(main_dataset_path, "train")
        if os.path.exists(train_path):
            from datasets import load_from_disk
            local_train_dataset = load_from_disk(train_path)
            print(f"训练集加载成功！共包含 {len(local_train_dataset)} 个样本")
        else:
            print("训练集文件不存在")
            local_train_dataset = None
        
        # 加载验证集
        validation_path = os.path.join(main_dataset_path, "validation")
        if os.path.exists(validation_path):
            local_validation_dataset = load_from_disk(validation_path)
            print(f"验证集加载成功！共包含 {len(local_validation_dataset)} 个样本")
        else:
            print("验证集文件不存在")
            local_validation_dataset = None
        
        # 测试token长度
        print("\n" + "="*50)
        print("测试数据集token长度")
        print("="*50)
        
        # 加载tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            cache_dir=args.cache_dir,
        )
        print("Tokenizer加载成功")
        
        def analyze_token_lengths(dataset, dataset_name):
            """分析数据集的token长度"""
            if dataset is None:
                print(f"{dataset_name}数据集不存在，跳过分析")
                return
            
            print(f"\n分析{dataset_name}数据集...")
            
            # 统计变量
            sentence_token_lengths = []
            corr_sentence_token_lengths = []
            max_sentence_tokens = 0
            max_corr_sentence_tokens = 0
            max_sentence_sample = None
            max_corr_sentence_sample = None
            over_limit_count = 0
            
            # 分析每个样本
            for i, sample in enumerate(dataset):
                # 分析sentence
                sentence = sample['sentence']
                sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
                sentence_token_count = len(sentence_tokens)
                sentence_token_lengths.append(sentence_token_count)
                
                if sentence_token_count > max_sentence_tokens:
                    max_sentence_tokens = sentence_token_count
                    max_sentence_sample = sample
                
                # 分析corr_sentence
                corr_sentence = sample['corr_sentence']
                corr_sentence_tokens = tokenizer.encode(corr_sentence, add_special_tokens=False)
                corr_sentence_token_count = len(corr_sentence_tokens)
                corr_sentence_token_lengths.append(corr_sentence_token_count)
                
                if corr_sentence_token_count > max_corr_sentence_tokens:
                    max_corr_sentence_tokens = corr_sentence_token_count
                    max_corr_sentence_sample = sample
                
                # 检查是否有超过1000的样本
                if sentence_token_count > 70 or corr_sentence_token_count > 70:
                    over_limit_count += 1
                    print(f"  警告: 样本 {i} 超过1000 tokens - sentence: {sentence_token_count}, corr_sentence: {corr_sentence_token_count}")
                
                # 每100个样本显示一次进度
                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i + 1}/{len(dataset)} 个样本")
            
            # 计算统计信息
            avg_sentence_tokens = sum(sentence_token_lengths) / len(sentence_token_lengths)
            avg_corr_sentence_tokens = sum(corr_sentence_token_lengths) / len(corr_sentence_token_lengths)
            min_sentence_tokens = min(sentence_token_lengths)
            min_corr_sentence_tokens = min(corr_sentence_token_lengths)
            
            # 显示统计结果
            print(f"\n{dataset_name}数据集Token长度统计:")
            print(f"  sentence:")
            print(f"    最大token数: {max_sentence_tokens}")
            print(f"    最小token数: {min_sentence_tokens}")
            print(f"    平均token数: {avg_sentence_tokens:.2f}")
            print(f"    token数范围: {min_sentence_tokens} - {max_sentence_tokens}")
            
            print(f"  corr_sentence:")
            print(f"    最大token数: {max_corr_sentence_tokens}")
            print(f"    最小token数: {min_corr_sentence_tokens}")
            print(f"    平均token数: {avg_corr_sentence_tokens:.2f}")
            print(f"    token数范围: {min_corr_sentence_tokens} - {max_corr_sentence_tokens}")
            
            if over_limit_count > 0:
                print(f"  ⚠️  警告: 发现 {over_limit_count} 个样本超过1000 tokens限制")
            else:
                print(f"  ✅ 所有样本都在1000 tokens限制内")
            
            # 显示最长样本的详细信息
            if max_sentence_sample:
                print(f"\n最长sentence样本 (token数: {max_sentence_tokens}):")
                print(f"  sentence: {max_sentence_sample['sentence']}")
                print(f"  corr_sentence: {max_sentence_sample['corr_sentence']}")
            
            if max_corr_sentence_sample and max_corr_sentence_sample != max_sentence_sample:
                print(f"\n最长corr_sentence样本 (token数: {max_corr_sentence_tokens}):")
                print(f"  sentence: {max_corr_sentence_sample['sentence']}")
                print(f"  corr_sentence: {max_corr_sentence_sample['corr_sentence']}")
            
            return {
                'max_sentence_tokens': max_sentence_tokens,
                'max_corr_sentence_tokens': max_corr_sentence_tokens,
                'avg_sentence_tokens': avg_sentence_tokens,
                'avg_corr_sentence_tokens': avg_corr_sentence_tokens
            }
        
        # 分析训练集
        train_stats = analyze_token_lengths(local_train_dataset, "训练集")
        
        # 分析验证集
        validation_stats = analyze_token_lengths(local_validation_dataset, "验证集")
        
        # 综合统计
        if train_stats and validation_stats:
            print(f"\n{'='*60}")
            print("综合统计结果:")
            print(f"{'='*60}")
            print(f"训练集 + 验证集最大token数:")
            print(f"  sentence: {max(train_stats['max_sentence_tokens'], validation_stats['max_sentence_tokens'])}")
            print(f"  corr_sentence: {max(train_stats['max_corr_sentence_tokens'], validation_stats['max_corr_sentence_tokens'])}")
            print(f"训练集 + 验证集平均token数:")
            print(f"  sentence: {(train_stats['avg_sentence_tokens'] + validation_stats['avg_sentence_tokens']) / 2:.2f}")
            print(f"  corr_sentence: {(train_stats['avg_corr_sentence_tokens'] + validation_stats['avg_corr_sentence_tokens']) / 2:.2f}")
        
        # 返回处理后的数据集供进一步处理
        return processed_dataset
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        return None

if __name__ == "__main__":
    dataset = main()
    if dataset is not None:
        print(f"\n数据集加载成功！共包含 {len(dataset)} 个样本")
    else:
        print("数据集加载失败！")
