import os
import json
import argparse
import random
from tqdm import tqdm

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

templates = [
    "The {noun} lasted from {start} to {end_century}",
    "The {noun} stretched from {start} to {end_century}",
    "The {noun} spanned the years {start} to {end_century}",
    "The {noun} unfolded from {start} to {end_century}",
    "The {noun} took place between {start} and {end_century}",
    "The {noun} persisted from {start} to {end_century}",
]

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--start-century", "-s", type=int, default=10)
    parser.add_argument("--end-century", "-e", type=int, default=21)
    parser.add_argument("--nouns", "-n", default="data/helper_files/nouns.txt")
    parser.add_argument("--out-path", "-o", default="data/datasets/gt/")
    parser.add_argument("--tokenizer", "-t", default="gpt2")
    parser.add_argument("--use-only-original-template", "-u", action="store_true")
    parser.add_argument("--train", "-tr", default=150)
    parser.add_argument("--validation", "-va", default=150)
    parser.add_argument("--test", "-tt", default=5000)
    parser.add_argument("--seed", "-sd", type=int, default=42)
    parser.add_argument("--randomize-noun", "-rn", action="store_true")
    
    args = parser.parse_args()
    
    if args.seed >= 0:
        random.seed(args.seed)
    
    return args

def get_potential_years(tokenizer, century):
    candidates = ["{:02d}{:02d}".format(century, i) for i in range(100)]
    years = []
    for candidate in candidates:
        tokens = tokenizer(candidate, return_tensors="pt")['input_ids'][0]
        eng_tokens = tokenizer.convert_ids_to_tokens(tokens)
        if len(eng_tokens) == 2 and len(eng_tokens[0]) == 2 and len(eng_tokens[1]) == 2:
            years.append(candidate)   
    return years

def get_examples_for_century(template, century, noun, potential_years):
    century = str(century)
    last_two_digits_possible = [yr[-2:] for yr in potential_years]
    examples = []
    for i in range(1, len(last_two_digits_possible)-1):     # At least one year before and after
        examples.append({
            "template": template,
            "century": century,
            "noun": noun,
            "digits": last_two_digits_possible[i],
            "prefix": template.format(noun=noun, start=century+last_two_digits_possible[i-1], end_century=century),
            "good": last_two_digits_possible[i+1:].copy(),
            "bad": last_two_digits_possible[:i].copy()
        })
    return examples
        

def main():
    args = parse_args()
    
    if args.use_only_original_template:
        chosen_templates = templates[:1]
    else:
        chosen_templates = templates
    
    nouns = [line.strip() for line in open(args.nouns)]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    centuries = list(range(args.start_century, args.end_century+1))
    potential_years = {
        century: get_potential_years(tokenizer, century)
        for century in centuries
    }
    year_digits = [f"{i}{j}" for i in range(10) for j in range(10)]
    # Remove 00 and 99
    year_digits = year_digits[1:-1]
    
    # Before any filtration, get all data points
    all_examples = []
    for i, template in enumerate(chosen_templates):
        print("{}/{}".format(i+1, len(chosen_templates)))
        for century in tqdm(centuries):
            for noun in nouns:
                all_examples.extend(get_examples_for_century(template, century, noun, potential_years[century]))

    total = len(all_examples)
    print("Before division: {} examples".format(total))
    
    # Make sure that a different set of years appears in the validation and test splits w.r.t the train split
    candidates = ["{:02d}".format(i) for i in range(100)]
    random.shuffle(candidates)
    
    print("Splitting examples by year...")
    examples_split = {candidates[i] : [e for e in all_examples if e['digits'] == candidates[i]] for i in tqdm(range(len(candidates)))}
    
    i = 0
    train = []
    validation = []
    test = []
    
    while len(validation) < args.validation:
        validation.extend(examples_split[candidates[i]])
        i += 1
    while len(test) < args.test:
        test.extend(examples_split[candidates[i]])
        i += 1
    while len(train) < args.train:
        train.extend(examples_split[candidates[i]])
        i += 1
    
    train = train[:args.train]
    validation = validation[:args.validation]
    test = test[:args.test]
    
    print("Shuffling...")
    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)
    
    print("After division: {} train, {} validation, {} test".format(len(train), len(validation), len(test)))
    
    data = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(validation),
        "test": Dataset.from_list(test)
    })    
    
    processed = {}
    centuries = [str(i) for i in centuries]
    
    for split in data:
        processed[split] = []
        for ex in tqdm(data[split]):
            while True:
                if args.randomize_noun:
                    noun = random.choice(nouns)
                else:
                    noun = ex["noun"]
                # Randomize the century and digits
                new_century = random.choice(centuries)
                new_digits = random.choice(year_digits) 
                new_prefix = ex["template"].format(noun=noun, start=new_century+new_digits, end_century=new_century)
                
                old_prefix_len = len(tokenizer.tokenize(ex["prefix"]))
                new_prefix_len = len(tokenizer.tokenize(new_prefix))
                
                new_digits_int = int(new_digits)
                any_ok = False
                for candidate in range(new_digits_int+1, 100):
                    candidate_str = new_century + f"{candidate:02d}"
                    tokenized = tokenizer.tokenize(candidate_str)
                    if len(tokenized) == 2 and len(tokenized[0]) == 2 and len(tokenized[1]) == 2:
                        any_ok = True
                        break

                if old_prefix_len == new_prefix_len and any_ok:
                    break
            
            ex["corr_noun"] = noun
            ex["corr_century"] = new_century
            ex["corr_digits"] = new_digits
            ex["corr_prefix"] = new_prefix
            processed[split].append(ex)
    
    processed = DatasetDict({
        k: Dataset.from_list(v) for k, v in processed.items()
    })
    
    print(processed)
    
    processed.save_to_disk(args.out_path)

if __name__ == '__main__':
    main()