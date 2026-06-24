import os
import json
import random
import argparse
from tqdm import tqdm

from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--names", "-n", default="data/helper_files/names.json")
    parser.add_argument("--templates", "-t", default="data/helper_files/templates-gp.json")
    parser.add_argument("--train", "-tr", default=150, type=int)
    parser.add_argument("--validation", "-va", default=150, type=int)
    parser.add_argument("--test", "-tt", default=3000, type=int)
    parser.add_argument("--out-path", "-o", default="data/datasets/gp")
    parser.add_argument("--enforce-single-token", "-e", action="store_true")
    parser.add_argument("--split-by-template", "-b", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()
    
    if args.seed >= 0:
        random.seed(args.seed)
    
    return args

def is_single_token(tokenizer, token):
    return len(tokenizer.encode(token)) == 1

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    names = json.load(open(args.names))
    girls = names["girls"]
    boys = names["boys"]

    if args.enforce_single_token:
        girls = [girl for girl in girls if is_single_token(tokenizer, girl)]
        boys = [boy for boy in boys if is_single_token(tokenizer, boy)]

    # Equalize the numbers of boys and girls
    l = min(len(boys), len(girls))
    boys, girls = boys[:l], girls[:l]
    print("{} names for each".format(l))
    
    templates = json.load(open(args.templates))

    data = []

    sample = templates[0]
    if "{name1}" in sample:
        # This is the two name template
        for template in tqdm(templates):
            for boy in boys:
                for girl in girls:
                    data.append({
                        "prefix": template.format(name1=boy, name2=girl),
                        "pronoun": "he",
                        "template": template,
                        "name1": boy,
                        "name2": girl
                    })
                    data.append({
                        "prefix": template.format(name1=girl, name2=boy),
                        "pronoun": "she",
                        "template": template,
                        "name1": girl,
                        "name2": boy
                    })
    else:
        # This is the one name template
        for template in tqdm(templates):
            for boy in boys:
                data.append({
                    "prefix": template.format(name=boy),
                    "pronoun": "he",
                    "template": template,
                    "name": boy
                })
            for girl in girls:
                data.append({
                    "prefix": template.format(name=girl),
                    "pronoun": "she",
                    "template": template,
                    "name": girl
                })
    
    print("Total number of datapoints:", len(data))

    # If there are >= 30 templates, reserve a few for validation and testing
    # Otherwise, split randomly

    if len(templates) >= 30 and args.split_by_template:
        random.shuffle(templates)
        validation = []
        test = []
        train = []
        cur = 0
        while len(validation) < args.val_ratio * len(data):
            validation.extend([e for e in data if e["template"] == templates[cur]])
            cur += 1
        while len(test) < args.test_ratio * len(data):
            test.extend([e for e in data if e["template"] == templates[cur]])
            cur += 1
        train = [e for e in data if e["template"] in templates[cur:]]
    else:
        random.shuffle(data)
        validation = data[:args.validation]
        test = data[args.validation:args.validation + args.test]
        train = data[args.validation + args.test:args.validation + args.test + args.train]
    
    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)
    print("Number of datapoints in train:", len(train))
    print("Number of datapoints in validation:", len(validation))
    print("Number of datapoints in test:", len(test))

    data = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(validation),
        "test": Dataset.from_list(test)
    })
    
    processed = {}
    
    for split in data:
        processed[split] = []
        for ex in tqdm(data[split]):
            old_len = len(tokenizer.tokenize(ex["prefix"]))
            if ex["pronoun"] == "she":
                new_pronoun = "he"
                split_ = "boys"
            else:
                new_pronoun = "she"
                split_ = "girls"
            while True:
                new_name = random.choice(names[split_])
                new_prefix = ex["template"].format(name=new_name)
                if old_len == len(tokenizer.tokenize(new_prefix)):
                    break
            
            ex["corr_prefix"] = new_prefix
            ex["corr_name"] = new_name
            ex["corr_pronoun"] = new_pronoun
            processed[split].append(ex)
            
    processed = DatasetDict({
        k: Dataset.from_list(v) for k, v in processed.items()
    })

    dataset.save_to_disk(args.out_path)

if __name__ == '__main__':
    main()