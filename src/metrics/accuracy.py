import json
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


def eval_acc(
    model_name,
    retain_dataset,
    output_dir=".",
    batch_size=8,
    device="cuda"
):


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=None,
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


    dataloader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'attention_mask': torch.stack([item['attention_mask'] for item in x]),
            'label': torch.stack([item['label'] for item in x])
        }
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






            predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len-1]


            valid_mask = (labels != -100) & (attention_mask[:, :-1] == 1)


            correct = (predicted_tokens == labels) & valid_mask
            correct_predictions += correct.sum().item()
            total_predictions += valid_mask.sum().item()


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

    with open(f"{output_dir}/accuracy.json", "w") as f:
        json.dump(result, f, indent=4)

    return accuracy


if __name__ == "__main__":
    import argparse

    from src.dataset import get_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset_names = {"forget": "SafePku", "retain": "IOI"}
    dataset_seed = 8888
    forget_ratio = 0.1
    self_retain = False

    unlearn_dataset, test_dataset, unlearn_collator, test_collator = get_dataset(
        dataset_names, tokenizer, dataset_seed, forget_ratio, self_retain
    )


    accuracy = eval_acc(
        model_name="gpt2",
        retain_dataset=test_dataset,
        output_dir=args.output_dir,
        batch_size=4
    )

    print(f"最终准确率: {accuracy:.2f}%")