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