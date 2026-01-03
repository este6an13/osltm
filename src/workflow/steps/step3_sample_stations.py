"""
Step 3: Sample stations from check-out files.

This step:
1. Samples a number of check-out files (from sampled dates)
2. Extracts unique stations from those files
3. Creates a reference CSV with all unique stations
4. Samples n stations from the unique set using the seed
"""

import random
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.stations import extract_station_info


def collect_unique_stations(file_paths: list[Path], outs_path: Path) -> dict[str, str]:
    """
    Read check-out files and collect unique stations (by code).

    Returns:
        Dictionary mapping station_code -> station_name
    """
    unique_stations = {}

    for file_path in file_paths:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}, skipping...")
            continue

        print(f"📂 Reading stations from: {file_path.name}")

        try:
            # Read only the Estacion column
            df = pd.read_csv(
                file_path,
                usecols=["Estacion"],
                dtype={"Estacion": str},
            )

            # Extract unique stations
            for station_field in df["Estacion"].unique():
                code, name = extract_station_info(station_field)
                if code and name:
                    # Use first occurrence if duplicate codes (shouldn't happen, but safe)
                    if code not in unique_stations:
                        unique_stations[code] = name

        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")
            continue

    return unique_stations


def save_stations_reference(stations: dict[str, str], output_path: Path) -> None:
    """Save all unique stations to a CSV file for reference."""
    stations_list = [
        {"code": code, "name": name} for code, name in sorted(stations.items())
    ]
    df = pd.DataFrame(stations_list)
    df.to_csv(output_path, index=False)
    print(f"📄 Saved {len(stations_list)} stations to reference file: {output_path}")


def run(params: dict[str, Any]) -> None:
    """
    Execute step 3: sample stations from check-out files.

    Parameters are read from:
        - params['sampled_dates']: List of dates in YYYYMMDD format (from step 1)
        - params['step2']['outs_path']: Path to check-out files directory
        - params['step3']: Configuration dict with:
            - n_files: Number of files to sample (default: all available)
            - n_stations: Number of stations to sample
            - reference_csv_path: Path to save the reference CSV (default: workflow/stations_reference.csv)
    """
    # Get dates from step 1
    sampled_dates = params.get("sampled_dates")
    if not sampled_dates:
        raise ValueError(
            "No sampled_dates found in params. Run step 1 first or provide dates manually."
        )

    # Get check-out path from step 2
    step2_params = params.get("step2", {})
    outs_path = Path(step2_params.get("outs_path", "data/check_outs/daily"))

    # Get step 3 parameters
    step3_params = params.get("step3", {})
    n_files = step3_params.get("n_files")
    n_stations = step3_params.get("n_stations")
    if n_stations is None:
        raise ValueError("step3.n_stations is required in params")

    seed = params.get("seed", 42)
    random.seed(seed)

    # Get reference CSV path
    reference_csv_path = Path(
        step3_params.get("reference_csv_path", "src/workflow/stations_reference.csv")
    )

    print("🔍 Sampling stations from check-out files")
    print(f"   Check-out path: {outs_path}")
    print(f"   Available dates: {len(sampled_dates)}")
    print(f"   Seed: {seed}")

    # Build list of available file paths
    available_files = []
    for date_str in sampled_dates:
        file_path = outs_path / f"{date_str}.csv"
        if file_path.exists():
            available_files.append(file_path)

    if not available_files:
        raise ValueError(
            f"No check-out files found in {outs_path} for the sampled dates."
        )

    print(f"   Found {len(available_files)} available files")

    # Sample files if n_files is specified
    if n_files and n_files < len(available_files):
        files_to_process = random.sample(available_files, k=n_files)
        print(f"   Sampling {n_files} files from {len(available_files)} available")
    else:
        files_to_process = available_files
        print(f"   Using all {len(available_files)} available files")

    # Collect unique stations from sampled files
    print(f"\n📊 Extracting stations from {len(files_to_process)} files...")
    unique_stations = collect_unique_stations(files_to_process, outs_path)

    print(f"\n✅ Found {len(unique_stations)} unique stations")

    # Save reference CSV
    reference_csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_stations_reference(unique_stations, reference_csv_path)

    # Sample stations
    if n_stations > len(unique_stations):
        print(
            f"⚠️  Requested {n_stations} stations but only {len(unique_stations)} available. Using all."
        )
        sampled_station_codes = list(unique_stations.keys())
    else:
        sampled_station_codes = random.sample(
            list(unique_stations.keys()), k=n_stations
        )

    # Store sampled stations in params
    params["sampled_stations"] = [
        {"code": code, "name": unique_stations[code]} for code in sampled_station_codes
    ]

    print(f"\n🎯 Sampled {len(sampled_station_codes)} stations:")
    for station in params["sampled_stations"]:
        print(f"   {station['code']}: {station['name']}")


if __name__ == "__main__":
    # For testing
    import json
    from pathlib import Path

    params_path = Path(__file__).parent.parent / "params.json"
    with open(params_path) as f:
        params = json.load(f)

    # Simulate step 1 output
    if "sampled_dates" not in params or params["sampled_dates"] is None:
        print("⚠️  No sampled_dates in params, using test dates")
        params["sampled_dates"] = ["20240625", "20240628"]

    run(params)
    print(f"\n📋 Sampled stations (for next step): {params.get('sampled_stations')}")
