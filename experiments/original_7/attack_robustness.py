from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._single_experiment_entry import run_original_7


if __name__ == "__main__":
    raise SystemExit(run_original_7("attack_robustness"))
