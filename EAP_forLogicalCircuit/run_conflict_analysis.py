from __future__ import annotations

import sys
from pathlib import Path

try:
    from EAP_forLogicalCircuit.conflict_analysis import main
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from EAP_forLogicalCircuit.conflict_analysis import main


if __name__ == "__main__":
    main()


'''
cd /ssd_users/chenhang/CSAT

PYTHONPATH=$PWD /home/chenhang/.conda/envs/LLMSFT_BW/bin/python EAP_forLogicalCircuit/run_conflict_analysis.py \
  --task_artifact_dirs bool=files/logical_circuit/bool+IOI+sst2+arithmetic/bool_probing,IOI=files/logical_circuit/bool+IOI+sst2+arithmetic/IOI_probing,sst2=files/logical_circuit/bool+IOI+sst2+arithmetic/sst2_probing,5_digit_arithmetic=files/logical_circuit/bool+IOI+sst2+arithmetic/5_digit_arithmetic_probing \
  --output_dir files/logical_circuit/bool+IOI+sst2+arithmetic/conflict_analysis_probing \
  --min_rank 1 \
  --max_rank 32 \
  --head_to_matrix_aggregation mean \
  --rank_score_source normalized_abs \
  --mask_fill_strategy random \
  --mask_seed 0 \
  --mask_min_keep_ratio 0.1 \
  --mask_max_keep_ratio 0.9 \
  --write_dense_masks
'''