import os
import json
import random
from tqdm import tqdm
import sys
import argparse

import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--out-dir", "-o", default="data/datasets/ioi")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--start-size", "-n", type=int, default=300000) # Trim the dataset early to save time
    parser.add_argument("--num_test_templates", "-nt", type=int, default=4) # Number of held-out templates
    parser.add_argument("--train", "-t", type=int, default=200)
    parser.add_argument("--validation", "-v", type=int, default=200)
    parser.add_argument("--test", "-e", type=int, default=10000)
    parser.add_argument("--tokenizer", "-tk", default="gpt2")
    parser.add_argument("--names", "-nm", default="data/helper_files/names.json")
    
    args = parser.parse_args()
    
    args.names = json.load(open(args.names))
    if "girls" in args.names:
        args.names = args.names["boys"] + args.names["girls"]
    
    return args

out_dir = "data/ioi" if len(sys.argv) == 1 else sys.argv[1]

baba_templates = [
    "Then, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
    "Then, {B} and {A} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument, and afterwards {B} said to {A}",
    "After {B} and {A} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
    "While {B} and {A} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
    "While {B} and {A} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "After the lunch, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Afterwards, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument. Afterwards {B} said to {A}",
    "The {PLACE} {B} and {A} went to had a {OBJECT}. {B} gave it to {A}",
    "Friends {B} and {A} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
]

abba_templates = [
    "Then, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
    "Then, {A} and {B} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
    "Then, {A} and {B} had a long argument, and afterwards {B} said to {A}",
    "After {A} and {B} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
    "While {A} and {B} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
    "While {A} and {B} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "After the lunch, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Afterwards, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} had a long argument. Afterwards {B} said to {A}",
    "The {PLACE} {A} and {B} went to had a {OBJECT}. {B} gave it to {A}",
    "Friends {A} and {B} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
]

def try_fit_template(string, template):
    pieces_s, pieces_t = string.strip(), template.strip()
    
    mapping = {}
    
    for s, t in zip(pieces_s.split(), pieces_t.split()):
        if s == t:
            continue
        if s[-1] == t[-1] and s[-1] in [',', '.']:
            s, t = s[:-1], t[:-1]
        if t not in ['{A}', '{B}', '{PLACE}', '{OBJECT}']:
            return None
        elif t[1:-1].lower() in mapping:
            if mapping[t[1:-1].lower()] != s:
                return None
        else:
            mapping[t[1:-1].lower()] = s
    
    if 'place' not in mapping:
        mapping['place'] = None
    if 'object' not in mapping:
        mapping['object'] = None
    
    return mapping

def find_template(string):
    for template in baba_templates:
        mapping = try_fit_template(string, template)
        if mapping is not None:
            mapping.update({
                'template': template,
                'order': 'baba'
            })
            return mapping
    
    for template in abba_templates:
        mapping = try_fit_template(string, template)
        if mapping is not None:
            mapping.update({
                'template': template,
                'order': 'abba'
            })
            return mapping
    return None

def add_template(example, **kwargs):
    example.update(find_template(example['ioi_sentences']))
    return example

def train_val_split_templates(data, val_templates):
    val_indices = [i for i in tqdm(range(len(data))) if data[i]['template'] in val_templates]
    train_indices = [i for i in tqdm(range(len(data))) if i not in val_indices]
    
    return data.select(train_indices), data.select(val_indices)

def main():
    args = parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Part 1 : Prepare the splits
    
    n_baba = args.num_test_templates // 2
    n_abba = args.num_test_templates - n_baba
    
    val_templates = random.sample(baba_templates, n_baba) + random.sample(abba_templates, n_abba)
    
    data = load_dataset('fahamu/ioi')['train']
    if args.start_size < len(data):
        data = data.select(range(args.start_size))
    data = data.map(add_template, num_proc=32)
    
    train, test = train_val_split_templates(data, val_templates)
    
    assert len(train) >= args.train + args.validation, f"Train size {len(train)} is too small"
    assert len(test) >= args.test, f"Test size {len(test)} is too small"
    
    new_train = train.select(range(args.train))
    val = train.select(range(args.train, args.train+args.validation))
    test = test.select(range(args.test))
    
    data = DatasetDict({
        "train": new_train,
        "validation": val,
        "test": test
    })
    
    # Part 2: Prepare the corrupted (ABC) dataset
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    processed = {}
    
    for split in data:
        processed[split] = []
        for ex in tqdm(data[split]):
            sentence = ex['ioi_sentences']
            n_sent_tokens = len(tokenizer.tokenize(sentence))
            sentence = sentence.split()
            
            # Sample four random names
            new_a, new_b, name1, name2 = ex['a'], ex['b'], ex['a'], ex['b']
            while True:
                new_a, new_b, name1, name2 = random.sample(args.names, 4)
                
                if any(name in [ex['a'], ex['b']] for name in [new_a, new_b, name1, name2]):
                    continue
                
                # The number of tokens before and after should be the same            
                new_sentence = sentence.copy()
                first_a = new_sentence.index(ex['a'])
                first_b = new_sentence.index(ex['b'])
                second_a = new_sentence.index(ex['a'], new_sentence.index(ex['a']) + 1)
                second_b = new_sentence.index(ex['b'], new_sentence.index(ex['b']) + 1)
                new_sentence[first_a] = new_a
                new_sentence[first_b] = new_b
                new_sentence[second_a] = name1
                new_sentence[second_b] = name2
                new_sentence = " ".join(new_sentence)
                
                if len(tokenizer.tokenize(new_sentence)) == n_sent_tokens:
                    break
            
            ex['corr_a'] = new_a
            ex['corr_b'] = new_b
            ex['corr_c'] = name1          
            ex['corr_d'] = name2
            ex['corr_ioi_sentences'] = new_sentence
            processed[split].append(ex)
            
    processed = DatasetDict({
        k: Dataset.from_list(v) for k, v in processed.items()
    })
    
    # Part 3: Save the datasets
    
    processed.save_to_disk(out_dir)

if __name__ == '__main__':
    main()