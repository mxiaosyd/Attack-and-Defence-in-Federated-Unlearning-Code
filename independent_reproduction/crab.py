from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from independent_reproduction.reproduction_config import main_for_method


if __name__ == "__main__":
    raise SystemExit(main_for_method("crab", sys.argv[1:]))
