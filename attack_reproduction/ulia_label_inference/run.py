from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack_reproduction.ulia_label_inference.attack import run
from attack_reproduction.ulia_label_inference.config import DEFAULT_CONFIG, ULIAConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ULIA label inference reproduction.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = ULIAConfig(**DEFAULT_CONFIG.to_dict())
    if args.device is not None:
        config.device = args.device
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    result = run(config)
    print(result["metrics"])
    print(result["output_dir"])


if __name__ == "__main__":
    main()

