from unlearn.generate_mask import GenerateMask

from .FT import (
    TargetFT,
    TargetFT_L1,
    TargetFT_PervasivenessFT,
    TargetFT_PervasivenessKL,
)


def get_finetuning_method(name, *args, **kwargs):
    if name == "TargetFT":
        return TargetFT(*args, **kwargs)
    if name == "TargetFT+PervasivenessFT":
        return TargetFT_PervasivenessFT(*args, **kwargs)
    if name == "TargetFT+PervasivenessKL":
        return TargetFT_PervasivenessKL(use_reference_model=True, *args, **kwargs)
    if name == "TargetFT_L1":
        return TargetFT_L1(*args, **kwargs)
    raise ValueError(f"No finetuning method: {name}")
