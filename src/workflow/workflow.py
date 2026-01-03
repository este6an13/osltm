"""
Workflow orchestrator for executing steps in sequence.

Usage:
    python -m src.workflow.workflow --params params.json --steps all
    python -m src.workflow.workflow --params params.json --steps 1
    python -m src.workflow.workflow --params params.json --steps 1,3
    python -m src.workflow.workflow --params params.json --steps 1-3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.workflow.steps.step1_sample_dates import run as run_step1
from src.workflow.steps.step2_download_files import run as run_step2

# Registry of all workflow steps
STEPS = {
    1: ("Sample Stratified Dates", run_step1),
    2: ("Download Daily Data", run_step2),
}


def parse_step_selection(step_str: str) -> list[int]:
    """
    Parse step selection string into a list of step numbers.

    Examples:
        "all" -> [1, 2, 3, ...]
        "1" -> [1]
        "1,3" -> [1, 3]
        "1-3" -> [1, 2, 3]
        "1,3-5" -> [1, 3, 4, 5]
    """
    if step_str.lower() == "all":
        return list(STEPS.keys())

    steps = []
    parts = step_str.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Range: "1-3"
            start, end = part.split("-", 1)
            steps.extend(range(int(start), int(end) + 1))
        else:
            # Single step: "1"
            steps.append(int(part))

    return sorted(set(steps))


def load_params(params_path: str | Path) -> dict[str, Any]:
    """Load parameters from JSON file."""
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")

    with open(params_path, "r") as f:
        return json.load(f)


def run_workflow(params: dict[str, Any], step_numbers: list[int]) -> None:
    """Execute workflow steps in sequence."""
    print(f"🚀 Starting workflow with {len(step_numbers)} step(s)")
    print(f"📋 Steps to execute: {step_numbers}\n")

    for step_num in step_numbers:
        if step_num not in STEPS:
            print(f"⚠️  Step {step_num} not found, skipping...")
            continue

        step_name, step_func = STEPS[step_num]
        print(f"{'=' * 60}")
        print(f"▶️  Step {step_num}: {step_name}")
        print(f"{'=' * 60}")

        try:
            step_func(params)
            print(f"✅ Step {step_num} completed successfully\n")
        except Exception as e:
            print(f"❌ Step {step_num} failed: {e}")
            print(f"🛑 Workflow stopped at step {step_num}")
            sys.exit(1)

    print(f"{'=' * 60}")
    print("🎉 All steps completed successfully!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Execute workflow steps in sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all steps
    python -m src.workflow.workflow --params params.json --steps all

    # Run single step
    python -m src.workflow.workflow --params params.json --steps 1

    # Run specific steps
    python -m src.workflow.workflow --params params.json --steps 1,3

    # Run range of steps
    python -m src.workflow.workflow --params params.json --steps 1-3
        """,
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to JSON parameters file",
    )
    parser.add_argument(
        "--steps",
        type=str,
        required=True,
        help="Steps to execute: 'all', single number (e.g., '1'), comma-separated (e.g., '1,3'), or range (e.g., '1-3')",
    )

    args = parser.parse_args()

    # Load parameters
    try:
        params = load_params(args.params)
    except Exception as e:
        print(f"❌ Failed to load params: {e}")
        sys.exit(1)

    # Parse step selection
    try:
        step_numbers = parse_step_selection(args.steps)
    except Exception as e:
        print(f"❌ Invalid step selection: {e}")
        sys.exit(1)

    if not step_numbers:
        print("❌ No valid steps to execute")
        sys.exit(1)

    # Run workflow
    run_workflow(params, step_numbers)


if __name__ == "__main__":
    main()
