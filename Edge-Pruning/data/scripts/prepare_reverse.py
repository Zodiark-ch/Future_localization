from datasets import Dataset
import random
import json

num_tokens = 3
seq_len = 3
max_examples = 10000
out_path = "data/datasets/reverse-t{}-s{}".format(num_tokens, seq_len)
bos = "BOS"

def should_use_sampling(num_tokens, seq_len, max_examples):
    while max_examples > 1:
        max_examples /= num_tokens
        seq_len -= 1
    return seq_len > 0

def simplehash(l):
    return "#".join([str(x) for x in l])

def stringify(l):
    return [str(x) for x in l]

def random_permutation_with_no_fixed_points(l, num_tokens):
    for i in range(1, len(l)):
        new_token = l[i]
        while new_token == l[i]:
            new_token = random.randint(1, num_tokens)
        l[i] = new_token
    return l

data = []

if should_use_sampling(num_tokens, seq_len, max_examples):
    seen = set()
    for _ in range(max_examples):
        while True:
            seq = [bos]
            for _ in range(seq_len):
                seq.append(random.randint(1, num_tokens))
            h = simplehash(seq)
            if h not in seen:
                seen.add(h)
                data.append({
                    "seq": seq,
                    "target": [bos] + list(reversed(seq[1:])),
                })
                break
            
else:
    cur_sequences = [[]]
    for l in range(seq_len):
        next_sequences = []
        for seq in cur_sequences:
            for i in range(1, num_tokens + 1):
                next_sequences.append(seq + [i])
        cur_sequences = next_sequences
    for seq in cur_sequences:
        data.append({
            "seq": [bos] + seq,
            "target": [bos] + list(reversed(seq)),
        })

for i in range(len(data)):
    data[i]["corr_seq"] = random_permutation_with_no_fixed_points(data[i]["seq"].copy(), num_tokens)
    
    data[i]["seq"] = stringify(data[i]["seq"])
    data[i]["target"] = stringify(data[i]["target"])
    data[i]["corr_seq"] = stringify(data[i]["corr_seq"])
    
data = Dataset.from_list(data)
data.save_to_disk(out_path)