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

def make_frac_prevs(bools):
  bools = rasp.numerical(bools)
  prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
  return rasp.numerical(rasp.Aggregate(prevs, bools,
                                       default=0)).named("frac_prevs")

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
    
    parser.add_argument("--out_path", type=str, default="data/tracr_models/d-xproportion.tracr.pkl")
    parser.add_argument("--max_seq_len", type=int, default=5)
    parser.add_argument("--bos", type=str, default="BOS")
    parser.add_argument("--vocab-len-other-than-x", type=int, default=3)
    
    return parser.parse_args()

def main():
    args = parse_args()
    out_path = args.out_path
    
    frac_prevs = make_frac_prevs(rasp.tokens == "x")

    bos = args.bos
    max_seq_len = args.max_seq_len
    vocab = [chr(ord("a")+i) for i in range(args.vocab_len_other_than_x)] + ["x"]
    vocab_ = {v for v in vocab}
    
    vocab_dict = {bos: 0, "compiler_pad": len(vocab)}
    for i, v in enumerate(vocab):
        if i == len(vocab) - 1:
            vocab_dict[v] = i + 2
        else:
            vocab_dict[v] = i + 1

    model = compiling.compile_rasp_to_model(
        frac_prevs,
        vocab=vocab_,
        max_seq_len=max_seq_len,
        compiler_bos=bos,
    )

    iput = [bos, "x", "a", "c", "x"]
    out = model.apply(iput)

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

    pickle.dump({
        "model_params": model_params,
        "config": config,
        "vocab": vocab_dict,
    }, open(out_path, "wb"))
    
    print("EXAMPLE")
    print(iput, "->", out.decoded)

if __name__ == '__main__':
    main()