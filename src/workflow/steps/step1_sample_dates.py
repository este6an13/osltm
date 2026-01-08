"""
Step 1: Stratified sampling of dates.

This step samples dates using stratified sampling and stores the results
in the params dict under 'sampled_dates' for use by subsequent steps.
"""

import random
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.day_type import get_day_type


def sample_stratified_days(
    start_date: date,
    end_date: date,
    n_per_stratum: int = 2,
    random_seed: int = 42,
) -> list[date]:
    """
    Stratified sampling of days between start_date and end_date.
    Ensures at least one weekday and one weekend day per (year, month),
    and includes holidays if they exist.
    """
    random.seed(random_seed)

    # Group all days by (year, month, day_type)
    strata = defaultdict(list)
    current = start_date
    while current <= end_date:
        day_type = get_day_type(current)
        strata[(current.year, current.month, day_type)].append(current)
        current += timedelta(days=1)

    sampled_days = []

    # Iterate over all (year, month) combinations in the range
    year_months = sorted({(y, m) for (y, m, _) in strata.keys()})
    for year, month in year_months:
        # --- Weekdays ---
        weekdays = strata.get((year, month, "WD"), [])
        if weekdays:
            k = min(n_per_stratum, len(weekdays))
            sampled_days.extend(random.sample(weekdays, k=k))

        # --- Weekends (Saturday + Sunday) ---
        weekends = strata.get((year, month, "SA"), []) + strata.get(
            (year, month, "SU"), []
        )
        if weekends:
            k = min(n_per_stratum, len(weekends))
            sampled_days.extend(random.sample(weekends, k=k))
        elif weekdays:
            # fallback if no weekends
            sampled_days.append(random.choice(weekdays))

        # --- Holidays ---
        holidays_in_month = strata.get((year, month, "HO"), [])
        if holidays_in_month:
            k = min(n_per_stratum, len(holidays_in_month))
            sampled_days.extend(random.sample(holidays_in_month, k=k))

    # Sort and deduplicate
    sampled_days = sorted(set(sampled_days))
    return sampled_days


def run(params: dict[str, Any]) -> None:
    """
    Execute step 1: sample stratified dates.

    Parameters are read from params['step1']:
        - start_date: Start date in YYYY-MM-DD format (or null to use default)
        - end_date: End date in YYYY-MM-DD format (or null to use today - days_offset)
        - n_per_stratum: Number of samples per stratum (default: 2)
        - days_offset: Days to subtract from today if end_date is null (default: 2)

    Results are stored in params['sampled_dates'] as a list of date strings (YYYYMMDD).
    """
    step_params = params.get("step1", {})
    seed = params.get("seed", 42)

    # Parse start_date
    start_date_str = step_params.get("start_date")
    if start_date_str:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    else:
        raise ValueError("step1.start_date is required in params")

    # Parse end_date
    end_date_str = step_params.get("end_date")
    if end_date_str:
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    else:
        # Use today - days_offset
        days_offset = step_params.get("days_offset", 2)
        end_date = datetime.now().date() - timedelta(days=days_offset)

    n_per_stratum = step_params.get("n_per_stratum", 2)

    print(f"📅 Sampling dates from {start_date} to {end_date}")
    print(f"   n_per_stratum: {n_per_stratum}, seed: {seed}")

    # Sample dates
    sampled = sample_stratified_days(
        start_date=start_date,
        end_date=end_date,
        n_per_stratum=n_per_stratum,
        random_seed=seed,
    )

    # Store results in params for next steps
    sampled_dates_str = [d.strftime("%Y%m%d") for d in sampled]
    params["sampled_dates"] = sampled_dates_str

    # Save to data CSV
    persistence_dir = Path(
        params.get("step4", {}).get("persistence_dir", "src/workflow/data")
    )
    persistence_dir.mkdir(parents=True, exist_ok=True)
    dates_df = pd.DataFrame({"date": sampled_dates_str})
    dates_file = persistence_dir / "sampled_dates.csv"
    dates_df.to_csv(dates_file, index=False)
    print(f"💾 Saved {len(sampled_dates_str)} dates to {dates_file}")

    print(f"✅ Sampled {len(sampled)} unique dates:")
    for d in sampled:
        day_type = get_day_type(d)
        print(f"   {d} ({day_type})")


if __name__ == "__main__":
    # For testing
    import json
    from pathlib import Path

    params_path = Path(__file__).parent.parent / "params.json"
    with open(params_path) as f:
        params = json.load(f)

    run(params)
    print(f"\n📋 Sampled dates (for next step): {params['sampled_dates']}")
