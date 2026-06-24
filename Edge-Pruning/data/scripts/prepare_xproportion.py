from datasets import Dataset
import random
import json

non_x_tokens = ["a", "b", "c"]
all_tokens = non_x_tokens + ["x"]
seq_len = 4
max_examples = 10000

def should_use_sampling(all_tokens, seq_len, max_examples):
    while max_examples > 1:
        max_examples /= len(all_tokens)
        seq_len -= 1
    return seq_len > 0

def simplehash(l):
    return "#".join([str(x) for x in l])

def stringify(l):
    return [str(x) for x in l]

def get_target(seq):
    seq[0] = 0
    cur = 0
    count = 0
    for i in range(1, len(seq)):
        count += 1
        if seq[i] == "x":
            cur += 1
        seq[i] = cur / count
    return seq

def random_permutation_with_no_fixed_points(l, all_tokens):
    for i in range(1, len(l)):
        new_token = random.choice(all_tokens)
        while new_token == l[i]:
            new_token = random.choice(all_tokens)
        l[i] = new_token
    return l
        

out_path = "data/datasets/xproportion-t{}-s{}".format(len(all_tokens), seq_len)
bos = "BOS"

data = []

if should_use_sampling(all_tokens, seq_len, max_examples):
    seen = set()
    for _ in range(max_examples):
        while True:
            seq = [bos]
            for _ in range(seq_len):
                seq.append(random.choice(all_tokens))
            h = simplehash(seq)
            if h not in seen:
                seen.add(h)
                data.append({
                    "seq": seq,
                    "target": get_target(seq),
                })
                break
            
else:
    cur_sequences = [[]]
    for l in range(seq_len):
        next_sequences = []
        for seq in cur_sequences:
            for token in all_tokens:
                next_sequences.append(seq + [token])
        cur_sequences = next_sequences
    for seq in cur_sequences:
        data.append({
            "seq": [bos] + seq,
            "target": get_target([bos] + seq),
        })

for i in range(len(data)):
    data[i]["corr_seq"] = random_permutation_with_no_fixed_points(data[i]["seq"].copy(), all_tokens)
    data[i]["seq"] = stringify(data[i]["seq"])
    data[i]["corr_seq"] = stringify(data[i]["corr_seq"])
    data[i]["target"] = data[i]["target"]

print(f"{len(data)} examples")
print(data[0])

data = Dataset.from_list(data)
data.save_to_disk(out_path)