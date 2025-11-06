import argparse
import os
import shutil
import zipfile
from pathlib import Path

import requests


def download_file(url, dest):
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


def unzip_file(zip_path, extract_to):
    """Unzip a file and return the path of the extracted CSV."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    # Return CSV file path (there should be only one)
    for file in Path(extract_to).glob("*.csv"):
        return file
    return None


def handle_files(date, file_type, out_dir):
    """Download, unzip, and move files based on type and date."""
    base_urls = {
        "ins": f"https://storage.googleapis.com/validaciones_tmsa/ValidacionTroncal/validacionTroncal{date}.zip",
        "outs": f"https://storage.googleapis.com/validaciones_tmsa/Salidas/salidas{date}.zip",
    }

    tmp_dir = Path("temp_downloads")
    tmp_dir.mkdir(exist_ok=True)

    # Define expected CSV name (they match the date)
    expected_csv = Path(out_dir) / f"{date}.csv"
    if expected_csv.exists():
        print(
            f"⏭️ Skipping {file_type} for {date} — {expected_csv.name} already exists."
        )
        return

    url = base_urls[file_type]
    zip_path = tmp_dir / f"{file_type}_{date}.zip"

    # Download
    if not download_file(url, zip_path):
        return

    # Unzip
    extract_dir = tmp_dir / f"extracted_{file_type}_{date}"
    extract_dir.mkdir(exist_ok=True)
    csv_path = unzip_file(zip_path, extract_dir)

    if csv_path and csv_path.exists():
        dest_path = Path(out_dir) / csv_path.name
        shutil.move(str(csv_path), dest_path)
        print(f"📁 Moved {csv_path.name} → {dest_path}")
    else:
        print(f"⚠️ No CSV found in {zip_path}")

    # Cleanup
    os.remove(zip_path)
    shutil.rmtree(extract_dir)
    print(f"🧹 Cleaned up temporary files for {date} ({file_type})")


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize check-in/out CSVs by date."
    )
    parser.add_argument(
        "dates", nargs="+", help="Dates in format YYYYMMDD (e.g., 20241019 20241020)"
    )
    parser.add_argument(
        "--type",
        choices=["ins", "outs", "both"],
        required=True,
        help="Download type: ins, outs, or both",
    )
    parser.add_argument(
        "--ins_path",
        required=False,
        default="check_ins",
        help="Path for check-ins folder",
    )
    parser.add_argument(
        "--outs_path",
        required=False,
        default="check_outs",
        help="Path for check-outs folder",
    )

    args = parser.parse_args()

    # Create output directories if they don’t exist
    Path(args.ins_path).mkdir(parents=True, exist_ok=True)
    Path(args.outs_path).mkdir(parents=True, exist_ok=True)

    for date in args.dates:
        if args.type in ["ins", "both"]:
            handle_files(date, "ins", args.ins_path)
        if args.type in ["outs", "both"]:
            handle_files(date, "outs", args.outs_path)

    print("✅ All done!")


if __name__ == "__main__":
    main()
