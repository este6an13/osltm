import random
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Literal

from src.utils.colombian_holidays import is_colombian_holiday


def get_day_type(dt: date) -> Literal["weekday", "saturday", "sunday", "holiday"]:
    """
    Classify a date as weekday / saturday / sunday / holiday.
    Holidays override weekday/weekend classification.
    """
    if is_colombian_holiday(dt):
        return "holiday"
    elif dt.weekday() == 5:
        return "saturday"
    elif dt.weekday() == 6:
        return "sunday"
    else:
        return "weekday"


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

    Parameters
    ----------
    start_date : date
        Beginning of the range (inclusive).
    end_date : date
        End of the range (inclusive).
    n_per_stratum : int
        Number of samples per stratum (if available).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[date]
        List of sampled dates.
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
        weekdays = strata.get((year, month, "weekday"), [])
        if weekdays:
            k = min(n_per_stratum, len(weekdays))
            sampled_days.extend(random.sample(weekdays, k=k))
        else:
            # If no weekdays (extremely rare), skip
            pass

        # --- Weekends (combine Saturday + Sunday) ---
        weekends = strata.get((year, month, "saturday"), []) + strata.get(
            (year, month, "sunday"), []
        )
        if weekends:
            k = min(n_per_stratum, len(weekends))
            sampled_days.extend(random.sample(weekends, k=k))
        else:
            # fallback: if month has no weekends, sample another weekday
            if weekdays:
                sampled_days.append(random.choice(weekdays))

        # --- Holidays ---
        holidays_in_month = strata.get((year, month, "holiday"), [])
        if holidays_in_month:
            k = min(n_per_stratum, len(holidays_in_month))
            sampled_days.extend(random.sample(holidays_in_month, k=k))

    # Sort and deduplicate
    sampled_days = sorted(set(sampled_days))
    return sampled_days


if __name__ == "__main__":
    start = date(2024, 5, 11)
    end = datetime.now().date()

    sampled = sample_stratified_days(start, end, n_per_stratum=2)
    print(f"Sampled {len(sampled)} unique dates between {start} and {end}:")
    for d in sampled:
        print(f"{d} ({get_day_type(d)})")
