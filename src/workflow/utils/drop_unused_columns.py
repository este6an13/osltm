"""
Utility script to drop unused columns from check-in and check-out CSV files.

This script processes CSV files in the check-ins and check-outs directories,
keeping only the columns that are actually used in step4_populate_counts.py.
This helps minimize disk space usage.

Columns kept:
- Check-ins: Fecha_Transaccion, Estacion_Parada
- Check-outs: Fecha_Transaccion, Tiempo, Estacion, Salidas_S

Can be run standalone or called from step2 after each download.
"""

from pathlib import Path
from typing import Any

import pandas as pd

# Columns to keep for each file type
CHECKINS_COLUMNS = ["Fecha_Transaccion", "Estacion_Parada"]
CHECKOUTS_COLUMNS = ["Fecha_Transaccion", "Tiempo", "Estacion", "Salidas_S"]


def process_file(file_path: Path, columns_to_keep: list[str], file_type: str) -> bool:
    """
    Process a single CSV file to keep only specified columns.

    Args:
        file_path: Path to the CSV file
        columns_to_keep: List of column names to keep
        file_type: "ins" or "outs" for logging purposes

    Returns:
        True if successful, False otherwise
    """
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return False

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check which columns exist in the file
        existing_columns = set(df.columns)
        columns_to_keep_set = set(columns_to_keep)

        # Find missing columns
        missing_columns = columns_to_keep_set - existing_columns
        if missing_columns:
            print(f"⚠️  Warning: {file_path.name} is missing columns: {missing_columns}")
            # Only keep columns that exist
            columns_to_keep = [
                col for col in columns_to_keep if col in existing_columns
            ]

        # Find columns to drop
        columns_to_drop = existing_columns - columns_to_keep_set
        if not columns_to_drop:
            print(f"⏭️  {file_path.name} already has only required columns")
            return True

        # Keep only the required columns
        df_filtered = df[columns_to_keep]

        # Write back to the same file
        df_filtered.to_csv(file_path, index=False)

        print(
            f"✅ {file_path.name}: Dropped {len(columns_to_drop)} columns "
            f"({', '.join(sorted(columns_to_drop))})"
        )
        return True

    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")
        return False


def process_file_by_date(
    date_str: str,
    file_type: str,
    ins_path: Path,
    outs_path: Path,
) -> bool:
    """
    Process a single file by date and type.

    Args:
        date_str: Date string in YYYYMMDD format
        file_type: "ins" or "outs"
        ins_path: Path to check-ins directory
        outs_path: Path to check-outs directory

    Returns:
        True if successful, False otherwise
    """
    if file_type == "ins":
        file_path = ins_path / f"{date_str}.csv"
        columns_to_keep = CHECKINS_COLUMNS
    elif file_type == "outs":
        file_path = outs_path / f"{date_str}.csv"
        columns_to_keep = CHECKOUTS_COLUMNS
    else:
        print(f"❌ Invalid file_type: {file_type}. Must be 'ins' or 'outs'")
        return False

    return process_file(file_path, columns_to_keep, file_type)


def drop_unused_columns(
    ins_path: Path,
    outs_path: Path,
    process_checkins: bool = True,
    process_checkouts: bool = True,
    date_range_start: str | None = None,
    date_range_end: str | None = None,
    file_list: list[str] | None = None,
) -> None:
    """
    Process CSV files in check-ins and check-outs directories to drop unused columns.

    Args:
        ins_path: Path to check-ins directory
        outs_path: Path to check-outs directory
        process_checkins: Whether to process check-in files
        process_checkouts: Whether to process check-out files
        date_range_start: Start date in YYYYMMDD format (optional, inclusive)
        date_range_end: End date in YYYYMMDD format (optional, inclusive)
        file_list: List of specific date strings (YYYYMMDD) to process (optional)
    """
    print("\n🗜️  Dropping unused columns from CSV files")

    total_processed = 0
    total_errors = 0

    # Determine which files to process
    files_to_process = set()

    if file_list:
        # Process specific files
        files_to_process = set(file_list)
        print(f"📋 Processing {len(files_to_process)} specified files")
    else:
        # Process all files, optionally filtered by date range
        all_files = set()

        if process_checkins and ins_path.exists():
            csv_files = list(ins_path.glob("*.csv"))
            for csv_file in csv_files:
                # Extract date from filename (assuming format YYYYMMDD.csv)
                date_str = csv_file.stem
                if len(date_str) == 8 and date_str.isdigit():
                    all_files.add(date_str)

        if process_checkouts and outs_path.exists():
            csv_files = list(outs_path.glob("*.csv"))
            for csv_file in csv_files:
                date_str = csv_file.stem
                if len(date_str) == 8 and date_str.isdigit():
                    all_files.add(date_str)

        # Filter by date range if provided
        if date_range_start or date_range_end:
            filtered_files = set()
            for date_str in all_files:
                if date_range_start and date_str < date_range_start:
                    continue
                if date_range_end and date_str > date_range_end:
                    continue
                filtered_files.add(date_str)
            files_to_process = filtered_files
            if date_range_start or date_range_end:
                print(
                    f"📅 Filtering by date range: {date_range_start or 'start'} to {date_range_end or 'end'}"
                )
        else:
            files_to_process = all_files

        print(f"   Found {len(files_to_process)} files to process")

    # Process check-ins
    if process_checkins and ins_path.exists():
        print(f"\n📂 Processing check-ins in: {ins_path}")
        for date_str in sorted(files_to_process):
            file_path = ins_path / f"{date_str}.csv"
            if file_path.exists():
                if process_file(file_path, CHECKINS_COLUMNS, "ins"):
                    total_processed += 1
                else:
                    total_errors += 1
    elif process_checkins:
        print(f"⚠️  Check-ins directory not found: {ins_path}")

    # Process check-outs
    if process_checkouts and outs_path.exists():
        print(f"\n📂 Processing check-outs in: {outs_path}")
        for date_str in sorted(files_to_process):
            file_path = outs_path / f"{date_str}.csv"
            if file_path.exists():
                if process_file(file_path, CHECKOUTS_COLUMNS, "outs"):
                    total_processed += 1
                else:
                    total_errors += 1
    elif process_checkouts:
        print(f"⚠️  Check-outs directory not found: {outs_path}")

    print("\n✅ Column dropping complete!")
    print(f"   Processed: {total_processed} files")
    if total_errors > 0:
        print(f"   Errors: {total_errors} files")


def run(params: dict[str, Any]) -> None:
    """
    Execute column dropping based on params.

    Parameters are read from:
        - params['drop_columns']: Configuration dict with:
            - ins_path: Path for check-ins folder (default: from step2.ins_path)
            - outs_path: Path for check-outs folder (default: from step2.outs_path)
            - process_checkins: Whether to process check-ins (default: true)
            - process_checkouts: Whether to process check-outs (default: true)
            - date_range_start: Start date in YYYYMMDD format (optional)
            - date_range_end: End date in YYYYMMDD format (optional)
            - file_list: List of specific date strings (YYYYMMDD) to process (optional)
    """
    drop_params = params.get("drop_columns", {})

    # Get paths from drop_columns params or fallback to step2 params
    step2_params = params.get("step2", {})
    ins_path = Path(
        drop_params.get(
            "ins_path", step2_params.get("ins_path", "data/check_ins/daily")
        )
    )
    outs_path = Path(
        drop_params.get(
            "outs_path", step2_params.get("outs_path", "data/check_outs/daily")
        )
    )
    process_checkins = drop_params.get("process_checkins", True)
    process_checkouts = drop_params.get("process_checkouts", True)
    date_range_start = drop_params.get("date_range_start")
    date_range_end = drop_params.get("date_range_end")
    file_list = drop_params.get("file_list")

    drop_unused_columns(
        ins_path,
        outs_path,
        process_checkins,
        process_checkouts,
        date_range_start,
        date_range_end,
        file_list,
    )


if __name__ == "__main__":
    # For standalone execution
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Drop unused columns from check-in and check-out CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with params.json
    python -m src.workflow.utils.drop_unused_columns --params src/workflow/params.json

    # Process specific files
    python -m src.workflow.utils.drop_unused_columns --params src/workflow/params.json --files 20240625,20240628

    # Process date range
    python -m src.workflow.utils.drop_unused_columns --params src/workflow/params.json --date-start 20240625 --date-end 20240630
        """,
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to JSON parameters file",
    )
    parser.add_argument(
        "--files",
        type=str,
        help="Comma-separated list of date strings (YYYYMMDD) to process",
    )
    parser.add_argument(
        "--date-start",
        type=str,
        help="Start date in YYYYMMDD format (inclusive)",
    )
    parser.add_argument(
        "--date-end",
        type=str,
        help="End date in YYYYMMDD format (inclusive)",
    )

    args = parser.parse_args()

    # Load parameters
    try:
        params = json.load(open(args.params))
    except Exception as e:
        print(f"❌ Failed to load params: {e}")
        sys.exit(1)

    # Override drop_columns params with CLI arguments if provided
    if not params.get("drop_columns"):
        params["drop_columns"] = {}

    if args.files:
        params["drop_columns"]["file_list"] = [f.strip() for f in args.files.split(",")]

    if args.date_start:
        params["drop_columns"]["date_range_start"] = args.date_start

    if args.date_end:
        params["drop_columns"]["date_range_end"] = args.date_end

    run(params)
