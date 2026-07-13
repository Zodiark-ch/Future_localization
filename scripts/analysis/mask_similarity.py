import argparse

import torch

parser = argparse.ArgumentParser(description="Compare two model masks for similarity.")
parser.add_argument("--mask_path_1", type=str, required=True, help="Joint mask file")
parser.add_argument("--mask_path_2", type=str, required=True, help="Original mask file")

args = parser.parse_args()
joint_mask = torch.load(args.mask_path_1)
orig_mask = torch.load(args.mask_path_2)
total_nums = 0
total_mask = 0
Density = {}
Layer_density = {}
for key in joint_mask.keys():
    num = torch.count_nonzero(joint_mask[key])
    total = joint_mask[key].numel()
    layer_num = key.split("layers.")[-1].split(".")[0]
    conponet_name = key.split("layers.")[-1].replace(".weight", "").replace(f"{layer_num}.", "")
    Density[(layer_num, conponet_name)] = (num/total).item()
    total_nums += total
    total_mask += num
    if layer_num not in Layer_density:
        Layer_density[layer_num] = {}
        Layer_density[layer_num]["total"] = total
        Layer_density[layer_num]["total_mask"] = num
    elif layer_num in Layer_density:
        Layer_density[layer_num]["total"] += total
        Layer_density[layer_num]["total_mask"] += num
for key in Layer_density.keys():
    Layer_density[key]["density"] = Layer_density[key]["total_mask"]/Layer_density[key]["total"]
Density_lst = []
for key in Layer_density.keys():
    if key != "model" and key !="lm_head":
        print(f"layer: {key}, density: {Layer_density[key]['density']}")
        Density_lst.append(Layer_density[key]['density'].item())
print(Density_lst)
print(f"sparsity: {1-total_mask/total_nums:.6f}")






print(Density)
# Function to calculate Jaccard similarity (IoU) between two masks
def jaccard_similarity(mask1, mask2):
    intersection = (mask1 & mask2).float().sum()  # Logical AND
    union = (mask1 | mask2).float().sum()        # Logical OR
    # Avoid division by zero
    return intersection / union if union != 0 else 0

# Calculate similarity for each module and average
similarities = {}
for key in joint_mask.keys():
    # Ensure the key exists in both masks
    if key in orig_mask:
        similarity = jaccard_similarity(joint_mask[key], orig_mask[key])
        similarities[key]=similarity
    else:
        print(f"Warning: {key} not found in both masks.")
print(f"Similarity between the two masks: {similarities}")
# Compute the average similarity
average_similarity = sum(similarities.values()) / len(similarities)
print(f"Average similarity between the two masks: {average_similarity}")
