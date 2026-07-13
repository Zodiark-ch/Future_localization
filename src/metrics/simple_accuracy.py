import json
import torch
import tqdm
from torch.utils.data import DataLoader


def extract_retain_from_unlearn_dataset(unlearn_dataset, num_samples=None):

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

    from transformers import AutoModelForCausalLM, AutoTokenizer


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


    try:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    except:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    if hasattr(retain_dataset, 'retain_dataset') and retain_dataset.retain_dataset is not None:

        print("检测到UnlearnDataset，使用其retain_dataset进行评估")
        actual_dataset = retain_dataset.retain_dataset
    else:

        print("使用普通数据集进行评估")
        actual_dataset = retain_dataset


    def collate_fn(batch):

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

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)


            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )


            logits = outputs.logits


            predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]



            valid_mask = (labels != -100) & (attention_mask == 1)


            batch_correct = 0
            batch_total = 0

            for i in range(len(labels)):

                valid_positions = valid_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    last_valid_pos = valid_positions[-1]

                    if predicted_tokens[i, last_valid_pos] == labels[i, last_valid_pos]:
                        batch_correct += 1
                    batch_total += 1

            correct_predictions += batch_correct
            total_predictions += batch_total


    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

    print(f"准确率: {accuracy:.2f}%")
    print(f"正确预测数: {correct_predictions}")
    print(f"总预测数: {total_predictions}")


    result = {
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
        "model_name": model_name
    }


    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/accuracy.json", "w") as f:
        json.dump(result, f, indent=4)

    return accuracy



def eval_acc_in_unlearn(self, model_name, output_dir=".", batch_size=8):

    return eval_acc(
        model_name=model_name,
        retain_dataset=self.test_dataset,
        output_dir=output_dir,
        batch_size=batch_size
    )