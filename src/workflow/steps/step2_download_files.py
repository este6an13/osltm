"""
Step 2: Download daily data files.

This step downloads check-in/out CSV files for the dates sampled in step 1.
Optionally drops unused columns from downloaded files to minimize disk space.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import requests

from src.workflow.utils.drop_unused_columns import process_file_by_date


def download_file(url: str, dest: Path) -> bool:
    """Download a file from a URL to a destination path."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Downloaded: {url}")
        return True
    else:
        print(f"❌ Failed to download: {url} (Status {response.status_code})")
        return False


def unzip_file(zip_path: Path, extract_to: Path) -> Path | None:
    """Unzip a file and return the path of the extracted CSV."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    # Return CSV file path (there should be only one)
    for file in Path(extract_to).glob("*.csv"):
        return file
    return None


def handle_files(
    date_str: str,
    file_type: str,
    out_dir: Path,
    force_redownload: bool = False,
    drop_columns: bool = False,
    ins_path: Path | None = None,
    outs_path: Path | None = None,
) -> None:
    """
    Download, unzip, and move files based on type and date.
    Optionally drops unused columns after download.
    """
    base_urls = {
        "ins": f"https://storage.googleapis.com/validaciones_tmsa/ValidacionTroncal/validacionTroncal{date_str}.zip",
        "outs": f"https://storage.googleapis.com/validaciones_tmsa/Salidas/salidas{date_str}.zip",
    }

    tmp_dir = Path("temp_downloads")
    tmp_dir.mkdir(exist_ok=True)

    # Define expected CSV name (they match the date)
    expected_csv = out_dir / f"{date_str}.csv"
    if expected_csv.exists() and not force_redownload:
        print(
            f"⏭️  Skipping {file_type} for {date_str} — {expected_csv.name} already exists."
        )
        return
    elif expected_csv.exists() and force_redownload:
        print(f"🔄 Force redownload enabled — will overwrite {expected_csv.name}")
        os.remove(expected_csv)

    url = base_urls[file_type]
    zip_path = tmp_dir / f"{file_type}_{date_str}.zip"

    # Download
    if not download_file(url, zip_path):
        return

    # Unzip
    extract_dir = tmp_dir / f"extracted_{file_type}_{date_str}"
    extract_dir.mkdir(exist_ok=True)
    csv_path = unzip_file(zip_path, extract_dir)

    if csv_path and csv_path.exists():
        dest_path = out_dir / csv_path.name
        shutil.move(str(csv_path), dest_path)
        print(f"📁 Moved {csv_path.name} → {dest_path}")

        # Drop unused columns if enabled
        if drop_columns and ins_path and outs_path:
            print(f"🗜️  Dropping unused columns from {dest_path.name}...")
            process_file_by_date(date_str, file_type, ins_path, outs_path)
    else:
        print(f"⚠️  No CSV found in {zip_path}")

    # Cleanup
    os.remove(zip_path)
    shutil.rmtree(extract_dir)
    print(f"🧹 Cleaned up temporary files for {date_str} ({file_type})")


def run(params: dict[str, Any]) -> None:
    """
    Execute step 2: download daily data files.

    Parameters are read from:
        - params['sampled_dates']: List of dates in YYYYMMDD format (from step 1)
        - params['step2']: Configuration dict with:
            - type: "ins", "outs", or "both"
            - ins_path: Path for check-ins folder
            - outs_path: Path for check-outs folder
            - force_redownload: Whether to force redownload existing files (default: false)
            - drop_columns: Whether to drop unused columns after download (default: false)
            - process_checkins: Whether to process check-ins when dropping columns (default: true)
            - process_checkouts: Whether to process check-outs when dropping columns (default: true)
    """
    # Get dates from step 1 or step2 params
    sampled_dates = params.get("sampled_dates")
    if not sampled_dates:
        # Allow dates to be provided directly in step2 params as fallback
        step_params = params.get("step2", {})
        sampled_dates = step_params.get("dates")
        if not sampled_dates:
            raise ValueError(
                "No sampled_dates found in params. Run step 1 first or provide dates in step2.dates."
            )

    step_params = params.get("step2", {})
    download_type = step_params.get("type", "both")
    ins_path = Path(step_params.get("ins_path", "data/check_ins/daily"))
    outs_path = Path(step_params.get("outs_path", "data/check_outs/daily"))
    force_redownload = step_params.get("force_redownload", False)
    drop_columns = step_params.get("drop_columns", False)

    print(f"📥 Downloading files for {len(sampled_dates)} dates")
    print(f"   Type: {download_type}")
    print(f"   Ins path: {ins_path}")
    print(f"   Outs path: {outs_path}")
    print(f"   Force redownload: {force_redownload}")
    print(f"   Drop columns: {drop_columns}")

    # Create output directories if they don't exist
    ins_path.mkdir(parents=True, exist_ok=True)
    outs_path.mkdir(parents=True, exist_ok=True)

    # Download files for each date
    for date_str in sampled_dates:
        print(f"\n📅 Processing date: {date_str}")
        if download_type in ["ins", "both"]:
            handle_files(
                date_str,
                "ins",
                ins_path,
                force_redownload,
                drop_columns,
                ins_path,
                outs_path,
            )
        if download_type in ["outs", "both"]:
            handle_files(
                date_str,
                "outs",
                outs_path,
                force_redownload,
                drop_columns,
                ins_path,
                outs_path,
            )

    print("\n✅ All downloads completed!")


if __name__ == "__main__":
    # For testing
    import json
    from pathlib import Path

    params_path = Path(__file__).parent.parent / "params.json"
    with open(params_path) as f:
        params = json.load(f)

    # Simulate step 1 output
    if "sampled_dates" not in params:
        print("⚠️  No sampled_dates in params, using test dates")
        params["sampled_dates"] = ["20240625", "20240628"]

    run(params)
