import os
import sys
sys.path.append(
    os.path.join(
        os.getcwd()
    )
)

from tracrx.rasp import rasp
from tracrx.compiler import compiling
import pickle
import numpy as np
import haiku as hk

import argparse

def make_length():
  all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
  return rasp.SelectorWidth(all_true_selector)

def numpify(x):
    if type(x) == dict:
        return {
            k: numpify(v)
            for k, v in x.items()
        }
    if type(x) == list:
        return [
            numpify(v)
            for v in x
        ]
    return np.asarray(x)

def flatten_dict(x):
    key = list(x.keys())[0]
    if type(x[key]) == dict:
        return {
            f"{k1}_{k2}": v2
            for k1, v1 in x.items()
            for k2, v2 in flatten_dict(v1).items()
        }
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--out_path", type=str, default="data/tracr_models/d-reverse.tracr.pkl")
    parser.add_argument("--max_seq_len", type=int, default=5)
    parser.add_argument("--bos", type=str, default="BOS")
    parser.add_argument("--vocab-len", type=int, default=3)
    
    return parser.parse_args()

def main():
    args = parse_args()

    out_path = args.out_path

    length = make_length()  # `length` is not a primitive in our implementation.
    opp_index = length - rasp.indices - 1
    flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
    reverse = rasp.Aggregate(flip, rasp.tokens)
        
    bos = args.bos
    max_seq_len = args.max_seq_len
    vocab = {i for i in range(1, 1+args.vocab_len)}
    vocab_dict = {v: i for i, v in enumerate(vocab)}
    vocab_dict[bos] = len(vocab_dict)
    vocab_dict["compiler_pad"] = len(vocab_dict)

    model = compiling.compile_rasp_to_model(
        reverse,
        vocab=vocab,
        max_seq_len=max_seq_len,
        compiler_bos=bos,
    )

    model_params = numpify(model.params)
    model_params = flatten_dict(model_params)

    config = {
        "num_heads": model.model_config.num_heads,
        "num_layers": model.model_config.num_layers,
        "mlp_hidden_size": model.model_config.mlp_hidden_size,
        "dropout_rate": model.model_config.dropout_rate,
        "layer_norm": model.model_config.layer_norm,
        "causal": model.model_config.causal,
        "max_seq_len": max_seq_len,
        "bos": bos,
        "pad": "compiler_pad",
    }

    iput = [bos, *vocab]
    out = model.apply(iput)

    pickle.dump({
        "model_params": model_params,
        "config": config,
        "vocab": vocab_dict,
        "unembedding_mtx": out.unembedding_mtx,
    }, open(out_path, "wb"))

    print("EXAMPLE")
    print(iput, "->", out.decoded)
    
if __name__ == "__main__":
    main()