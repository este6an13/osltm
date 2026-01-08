"""
Functional PCA per Station

This script performs Functional Principal Component Analysis (FPCA) on daily count profiles
for each station. Each day's profile is treated as a function/vector of counts across time windows.

The results are projected onto 2 principal components and visualized with points colored by day type.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.workflow.data_loader import load_data

# Color mapping for date types
DATE_TYPE_COLORS = {
    "WD": "#2E86AB",  # Blue
    "SA": "#A23B72",  # Purple
    "SU": "#F18F01",  # Orange
    "HO": "#C73E1D",  # Red
}

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}


def extract_time_series_data(
    df: pd.DataFrame, station_code: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract time series data for a specific station.

    Args:
        df: DataFrame from data_loader with columns: year, month, day, date_type,
            station_code, and time window columns (t_400, t_415, ..., t_2300)
        station_code: Station code to filter by

    Returns:
        Tuple of (time_series_matrix, date_types, date_strs) where:
            - time_series_matrix: DataFrame with rows as days and columns as time windows
            - date_types: Series with date_type for each day
            - date_strs: Series with date strings (YYYY-MM-DD) for each day
    """
    # Filter by station
    station_df = df[df["station_code"] == station_code].copy()

    if station_df.empty:
        return pd.DataFrame(), pd.Series(dtype=str), pd.Series(dtype=str)

    # Get time window columns (all columns starting with 't_')
    time_cols = [col for col in station_df.columns if col.startswith("t_")]
    time_cols = sorted(time_cols)  # Ensure proper ordering

    # Extract time series matrix (each row is a day, columns are time windows)
    time_series = station_df[time_cols].copy()

    # Extract date types
    date_types = station_df["date_type"].copy()

    # Create date identifier for reference
    station_df["date_str"] = (
        station_df["year"].astype(str)
        + "-"
        + station_df["month"].astype(str).str.zfill(2)
        + "-"
        + station_df["day"].astype(str).str.zfill(2)
    )

    return time_series, date_types, station_df["date_str"]


def perform_fpca(
    time_series: pd.DataFrame,
    n_components: int = 2,
    standardize: bool = True,
) -> tuple[np.ndarray, PCA, Optional[StandardScaler]]:
    """
    Perform Functional PCA on time series data.

    Args:
        time_series: DataFrame with rows as days and columns as time windows
        n_components: Number of principal components to extract
        standardize: Whether to standardize the data before PCA

    Returns:
        Tuple of (projected_data, pca_model, scaler)
    """
    if time_series.empty:
        return np.array([]), None, None

    # Remove rows with all NaN
    time_series = time_series.dropna(how="all")

    if len(time_series) < n_components:
        print(
            f"⚠️  Not enough samples ({len(time_series)}) for {n_components} components"
        )
        return np.array([]), None, None

    # Fill remaining NaN with 0 (or could use forward fill, mean, etc.)
    time_series = time_series.fillna(0)

    # Standardize if requested
    scaler = None
    if standardize:
        scaler = StandardScaler()
        time_series_scaled = scaler.fit_transform(time_series)
    else:
        time_series_scaled = time_series.values

    # Perform PCA
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(time_series_scaled)

    return projected, pca, scaler


def plot_fpca_results(
    projected: np.ndarray,
    date_types: pd.Series,
    station_code: str,
    station_name: str,
    date_strs: pd.Series,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """
    Plot FPCA results with points colored by date type.

    Args:
        projected: 2D array of projected data (n_samples, 2)
        date_types: Series with date_type for each sample
        station_code: Station code
        station_name: Station name
        date_strs: Series with date strings for annotation
        output_dir: Optional directory to save the plot
        count_type: Type of counts ("checkins" or "checkouts")
    """
    if len(projected) == 0:
        print(f"⚠️  No data to plot for station {station_code}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points colored by date type
    for date_type in date_types.unique():
        mask = date_types == date_type
        if not mask.any():
            continue

        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=DATE_TYPE_COLORS.get(date_type, "gray"),
            label=DATE_TYPE_LABELS.get(date_type, date_type),
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("First Principal Component (PC1)", fontsize=12)
    ax.set_ylabel("Second Principal Component (PC2)", fontsize=12)
    ax.set_title(
        f"Functional PCA: {station_code} - {station_name}\n({count_type.capitalize()})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Day Type", loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"fpca_{station_code}_{count_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved plot to {filename}")

    plt.show()


def run_fpca_per_station(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    n_components: int = 2,
    standardize: bool = True,
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
) -> dict:
    """
    Run Functional PCA for each station.

    Args:
        count_type: Type of counts to analyze ("checkins" or "checkouts")
        n_components: Number of principal components (default: 2)
        standardize: Whether to standardize data before PCA (default: True)
        output_dir: Directory to save plots (default: src/workflow/results/fpca_results)
        params_path: Path to params.json (default: src/workflow/params.json)
        station_codes: Optional list of station codes to analyze. If None, uses all from data.

    Returns:
        Dictionary with FPCA results per station
    """
    # Load parameters
    if params_path is None:
        params_path = Path("src/workflow/params.json")

    with open(params_path) as f:
        params = json.load(f)

    step4_params = params.get("step4", {})
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)
    time_step = step4_params.get("time_step", 15)

    # Load data
    print(f"📊 Loading {count_type} data...")
    if station_codes:
        print(f"   Filtering to {len(station_codes)} specified stations")
    data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        time_min=time_min,
        time_max=time_max,
        time_step=time_step,
    )

    if count_type not in data:
        raise ValueError(f"No {count_type} data available")

    df = data[count_type]

    if df.empty:
        raise ValueError(f"Empty DataFrame for {count_type}")

    # Get unique stations (if not already specified)
    if station_codes is None:
        station_codes = df["station_code"].unique().tolist()
    else:
        # Filter to only stations that actually have data
        available_stations = set(df["station_code"].unique())
        requested_stations = set(station_codes)
        missing = requested_stations - available_stations
        if missing:
            print(
                f"⚠️  Warning: Some specified stations have no data: {sorted(missing)}"
            )
        station_codes = [sc for sc in station_codes if sc in available_stations]

    print(f"🔍 Analyzing {len(station_codes)} stations...")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/results/fpca_results")
    output_dir = Path(output_dir)

    results = {}

    # Process each station
    for station_code in station_codes:
        station_df = df[df["station_code"] == station_code]
        if station_df.empty:
            print(f"⚠️  No data for station {station_code}, skipping...")
            continue

        station_name = station_df["station_name"].iloc[0]
        print(f"\n📈 Processing station: {station_code} - {station_name}")

        # Extract time series data
        time_series, date_types, date_strs = extract_time_series_data(df, station_code)

        if time_series.empty:
            print(f"⚠️  No time series data for station {station_code}, skipping...")
            continue

        print(f"   Found {len(time_series)} days with data")

        # Perform FPCA
        projected, pca, scaler = perform_fpca(
            time_series, n_components=n_components, standardize=standardize
        )

        if len(projected) == 0:
            print(f"⚠️  FPCA failed for station {station_code}, skipping...")
            continue

        # Print explained variance
        explained_variance = pca.explained_variance_ratio_
        print(
            f"   Explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}"
        )

        # Store results
        results[station_code] = {
            "station_name": station_name,
            "projected": projected,
            "date_types": date_types.values,
            "date_strs": date_strs.values,
            "explained_variance": explained_variance,
            "pca": pca,
            "scaler": scaler,
        }

        # Plot results
        plot_fpca_results(
            projected,
            date_types,
            station_code,
            station_name,
            date_strs,
            output_dir=output_dir,
            count_type=count_type,
        )

    print(f"\n✅ Completed FPCA for {len(results)} stations")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform Functional PCA per station on daily count profiles"
    )
    parser.add_argument(
        "--count_type",
        choices=["checkins", "checkouts"],
        default="checkins",
        help="Type of counts to analyze (default: checkins)",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of principal components (default: 2)",
    )
    parser.add_argument(
        "--no_standardize",
        action="store_true",
        help="Don't standardize data before PCA",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/results/fpca_results",
        help="Directory to save plots (default: src/workflow/results/fpca_results)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="src/workflow/params.json",
        help="Path to params.json (default: src/workflow/params.json)",
    )
    parser.add_argument(
        "--stations",
        type=str,
        nargs="+",
        help="Optional list of station codes to analyze (default: all from data)",
    )

    args = parser.parse_args()

    results = run_fpca_per_station(
        count_type=args.count_type,
        n_components=args.n_components,
        standardize=not args.no_standardize,
        output_dir=Path(args.output_dir),
        params_path=Path(args.params),
        station_codes=args.stations,
    )

    print(f"\n📊 Summary: Analyzed {len(results)} stations")
