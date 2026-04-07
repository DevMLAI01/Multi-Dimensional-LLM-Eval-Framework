"""
Run all 5 golden dataset generators in sequence, then validate.

Usage:
    uv run python data/golden_dataset/generate_all.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    from data.golden_dataset.generate_correctness_cases import generate_correctness_cases
    from data.golden_dataset.generate_faithfulness_cases import generate_faithfulness_cases
    from data.golden_dataset.generate_latency_cases import generate_latency_cases
    from data.golden_dataset.generate_robustness_cases import generate_robustness_cases
    from data.golden_dataset.generate_safety_cases import generate_safety_cases
    from data.golden_dataset.validate_dataset import validate

    log.info("=== Phase 2: Golden Dataset Generation ===")

    log.info("\n--- Correctness cases (50) ---")
    generate_correctness_cases()

    log.info("\n--- Faithfulness cases (40) ---")
    generate_faithfulness_cases()

    log.info("\n--- Robustness cases (40) ---")
    generate_robustness_cases()

    log.info("\n--- Safety cases (30) ---")
    generate_safety_cases()

    log.info("\n--- Latency cases (40) ---")
    generate_latency_cases()

    log.info("\n--- Validation ---")
    passed = validate()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
