import os
import json
import random
import argparse
from tqdm import tqdm

from datasets import Dataset, DatasetDict

class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def info(text):
    print(f"{bcolors.OKBLUE}{text}{bcolors.ENDC}")

def good(text):
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")

def bad(text):
    print(f"{bcolors.FAIL}{text}{bcolors.ENDC}")
    
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--out-path", "-o", default="data/datasets/boolean_expressions")
    parser.add_argument("--num-examples", "-n", type=int, default=10000)
    parser.add_argument("--val-ratio", "-v", type=float, default=0.1)
    parser.add_argument("--test-ratio", "-t", type=float, default=0.4)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--max-expr-length", "-l", type=int, default=6)
    parser.add_argument("--max-expr-depth", "-d", type=int, default=3)
    parser.add_argument("--max-consecutive-not", "-c", type=int, default=2)
    parser.add_argument("--max-tries", "-m", type=int, default=None)
    
    args = parser.parse_args()
    if args.max_tries is None:
        args.max_tries = args.num_examples * 5
    
    random.seed(args.seed)
    
    return args

def apply_or(expr_list, value_list, depth_list, idx):
    assert len(expr_list) > idx+1, "OR index out of bounds"
    lhs = expr_list[idx]
    if lhs not in ["False", "True"]:
        lhs = f"( {lhs} )"
    rhs = expr_list[idx+1]
    if rhs not in ["False", "True"]:
        rhs = f"( {rhs} )"
    expr = f"{lhs} or {rhs}"
    value = value_list[idx] or value_list[idx+1]
    expr_list = expr_list[:idx] + [expr] + expr_list[idx+2:]
    value_list = value_list[:idx] + [value] + value_list[idx+2:]
    depth_list = depth_list[:idx] + [max(depth_list[idx], depth_list[idx+1])+1] + depth_list[idx+2:]
    return expr_list, value_list, depth_list

def apply_and(expr_list, value_list, depth_list, idx):
    assert len(expr_list) > idx+1, "AND index out of bounds"
    lhs = expr_list[idx]
    if lhs not in ["False", "True"]:
        lhs = f"( {lhs} )"
    rhs = expr_list[idx+1]
    if rhs not in ["False", "True"]:
        rhs = f"( {rhs} )"
    expr = f"{lhs} and {rhs}"
    value = value_list[idx] and value_list[idx+1]
    expr_list = expr_list[:idx] + [expr] + expr_list[idx+2:]
    value_list = value_list[:idx] + [value] + value_list[idx+2:]
    depth_list = depth_list[:idx] + [max(depth_list[idx], depth_list[idx+1])+1] + depth_list[idx+2:]
    return expr_list, value_list, depth_list

def apply_not(expr_list, value_list, depth_list, idx):
    assert len(expr_list) > idx, "NOT index out of bounds"
    expr = expr_list[idx]
    if expr not in ["False", "True"]:
        expr = f"( {expr} )"
    expr = f"not {expr}"
    expr_list = expr_list[:idx] + [expr] + expr_list[idx+1:]
    value_list = value_list[:idx] + [not value_list[idx]] + value_list[idx+1:]
    depth_list = depth_list[:idx] + [depth_list[idx]+1] + depth_list[idx+1:]
    return expr_list, value_list, depth_list

def apply_random_operation(expr_list, value_list, depth_list):
    rand_op = random.choice(["or", "and", "not"])
    if rand_op == "or":
        idx = random.randint(0, len(expr_list)-2)
        return apply_or(expr_list, value_list, depth_list, idx)
    elif rand_op == "and":
        idx = random.randint(0, len(expr_list)-2)
        return apply_and(expr_list, value_list, depth_list, idx)
    elif rand_op == "not":
        idx = random.randint(0, len(expr_list)-1)
        return apply_not(expr_list, value_list, depth_list, idx)
    else:
        raise ValueError(f"Invalid operation {rand_op}")

def sample_initial_list(l=None, max_l=None):
    if l == None:
        assert max_l != None, "Either l or max_l must be specified"
        l = random.randint(1, max_l)
    value_list = [random.choice([False, True]) for _ in range(l)]
    expr_list = [str(value) for value in value_list]
    depth_list = [0 for _ in range(l)]
    return expr_list, value_list, depth_list

def sample_random_example(l=None, max_l=None, max_consecutive_nots=None, max_depth=None):
    count = 0
    while True:
        count += 1
        if count > 10000:
            # Assume that the parameters are invalid
            return None
        expr_list, value_list, depth_list = sample_initial_list(l, max_l)
        l_ = len(expr_list)
        while len(expr_list) > 1:
            expr_list, value_list, depth_list = apply_random_operation(expr_list, value_list, depth_list)
        check_for = (" not (" * max_consecutive_nots + " not").strip()
        if check_for not in expr_list[0] and depth_list[0] <= max_depth:
            break
    return {
        "input": f"{expr_list[0]} is",
        "target": str(value_list[0]),
        "length": l_,
        "depth": depth_list[0]
    }

def sample_n_examples(n, max_tries, l=None, max_l=None, max_depth=None, max_consecutive_nots=None):
    seen = set()
    data = []
    for _ in tqdm(range(max_tries)):
        example = sample_random_example(l, max_l, max_consecutive_nots, max_depth)
        if example is not None and example["input"] not in seen:
            data.append(example)
            seen.add(example["input"])
        if len(data) >= n:
            break
    return data

def construct_dataset(data, val_ratio, test_ratio):
    n = len(data)
    val_n = int(val_ratio * n)
    test_n = int(test_ratio * n)
    train_n = n - val_n - test_n
    random.shuffle(data)
    train_data = data[:train_n]
    val_data = data[train_n:train_n+val_n]
    test_data = data[train_n+val_n:]
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "val": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })

def main():
    args = parse_args()    
    
    info("[i] Sampling examples...")
    data = sample_n_examples(
        args.num_examples, 
        args.max_tries, 
        max_l=args.max_expr_length, 
        max_depth=args.max_expr_depth, 
        max_consecutive_nots=args.max_consecutive_not,
    )
    
    info(f"    [i] {len(data)} examples sampled")
    
    print()
    info(f"[i] Constructing dataset...")
    dataset = construct_dataset(data, args.val_ratio, args.test_ratio)
    
    info(f"[i] Train={len(dataset['train'])} Val={len(dataset['val'])} Test={len(dataset['test'])}")
    
    print()
    info("[i] Examples (2):")
    print(dataset["train"][0])
    print(dataset["test"][0])
    
    print()
    info(f"[i] Saving dataset to {args.out_path}...")
    
    dataset.save_to_disk(args.out_path)

if __name__ == "__main__":
    main()