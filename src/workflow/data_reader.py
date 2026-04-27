"""
Data reader for workflow: loads and transforms CSV files into pandas DataFrames.

This module provides a standardized way to load transaction data from CSV files
for checkins and checkouts, returning DataFrames ready for analysis.

Each row in the output represents one event with datetime and station information.
"""

import re
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from src.utils.stations import extract_station_info


def load_csv_file(
    csv_path: Path,
    station_codes: Optional[list[str]] = None,
    count_type: Literal["checkins", "checkouts"] = "checkins",
    include_time_components: bool = True,
) -> pd.DataFrame:
    """
    Load a CSV file and extract transaction times.

    Args:
        csv_path: Path to CSV file
        station_codes: Optional list of station codes to filter by
        count_type: Type of counts ("checkins" or "checkouts")
        include_time_components: If True, includes hour, minute, second, hour_float columns.
            If False, only includes hour column (as float). Default: True

    Returns:
        DataFrame with columns:
            - datetime: Datetime of the event
            - station_code: Station code
            - station_name: Station name
            - hour: Hour of day (int if include_time_components=True, float otherwise)
            - minute: Minute of hour (only if include_time_components=True)
            - second: Second of minute (only if include_time_components=True)
            - hour_float: Hour as float (only if include_time_components=True)

        For checkouts, counts are expanded into individual events.
    """
    if count_type == "checkins":
        # Checkins: each row is an event, datetime is in Fecha_Transaccion
        station_col = "Estacion_Parada"
        usecols = ["Fecha_Transaccion", station_col]

        try:
            df = pd.read_csv(
                csv_path,
                usecols=usecols,
                parse_dates=["Fecha_Transaccion"],
            )
        except KeyError:
            # Try alternative column name if first attempt fails
            station_col = "Estacion"
            usecols = ["Fecha_Transaccion", station_col]
            df = pd.read_csv(
                csv_path,
                usecols=usecols,
                parse_dates=["Fecha_Transaccion"],
            )

        # Rename datetime column
        df = df.rename(columns={"Fecha_Transaccion": "datetime"})

        # Filter by stations early using string pattern matching
        if station_codes:
            pattern = "|".join([re.escape(f"({code})") for code in station_codes])
            df = df[
                df[station_col]
                .astype(str)
                .str.contains(pattern, case=False, regex=True)
            ].copy()

        # Extract station code and name
        if df.empty:
            df["station_code"] = ""
            df["station_name"] = ""
        else:
            df[["station_code", "station_name"]] = df[station_col].apply(
                lambda x: pd.Series(extract_station_info(x))
            )

        # Filter by exact station code match (in case pattern matched multiple)
        if station_codes:
            df = df[df["station_code"].isin(station_codes)].copy()

    else:  # checkouts
        # Checkouts: each row is a timestamp with count in Salidas_S
        # Time is split into Fecha_Transaccion (date) and Tiempo (time)
        station_col = "Estacion"
        usecols = ["Fecha_Transaccion", "Tiempo", station_col, "Salidas_S"]

        try:
            df = pd.read_csv(
                csv_path,
                usecols=usecols,
            )
        except KeyError:
            # Try alternative column name if first attempt fails
            station_col = "Estacion_Parada"
            usecols = ["Fecha_Transaccion", "Tiempo", station_col, "Salidas_S"]
            df = pd.read_csv(
                csv_path,
                usecols=usecols,
            )

        # Combine Fecha_Transaccion and Tiempo into datetime
        df["datetime"] = pd.to_datetime(
            df["Fecha_Transaccion"].astype(str) + " " + df["Tiempo"].astype(str)
        )

        # Filter by stations early using string pattern matching
        if station_codes:
            pattern = "|".join([re.escape(f"({code})") for code in station_codes])
            df = df[
                df[station_col]
                .astype(str)
                .str.contains(pattern, case=False, regex=True)
            ].copy()

        # Extract station code and name
        if df.empty:
            df["station_code"] = ""
            df["station_name"] = ""
        else:
            df[["station_code", "station_name"]] = df[station_col].apply(
                lambda x: pd.Series(extract_station_info(x))
            )

        # Filter by exact station code match
        if station_codes:
            df = df[df["station_code"].isin(station_codes)].copy()

        # Group by datetime and station, sum counts (in case of duplicates)
        df = df.groupby(["datetime", "station_code", "station_name"], as_index=False)[
            "Salidas_S"
        ].sum()

        # Expand counts into individual events
        expanded_rows = []
        for _, row in df.iterrows():
            count = int(row["Salidas_S"])
            if count > 0:
                # Create count number of rows with the same datetime
                for _ in range(count):
                    expanded_rows.append(
                        {
                            "datetime": row["datetime"],
                            "station_code": row["station_code"],
                            "station_name": row["station_name"],
                        }
                    )

        if expanded_rows:
            df = pd.DataFrame(expanded_rows)
        else:
            # Return empty dataframe with correct columns
            df = pd.DataFrame(columns=["datetime", "station_code", "station_name"])

    # Extract time components
    if include_time_components:
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["second"] = df["datetime"].dt.second
        df["hour_float"] = df["hour"] + df["minute"] / 60.0 + df["second"] / 3600.0

        return df[
            [
                "datetime",
                "station_code",
                "station_name",
                "hour",
                "minute",
                "second",
                "hour_float",
            ]
        ].copy()
    else:
        # Only include hour as float (for time rescaling analysis)
        df["hour"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0

        return df[["datetime", "station_code", "station_name", "hour"]].copy()
