import random
from collections import defaultdict
from datetime import date, datetime, timedelta

from src.constants.seed import SEED
from src.utils.day_type import get_day_type


def sample_stratified_days(
    start_date: date,
    end_date: date,
    n_per_stratum: int = 2,
    random_seed: int = SEED,
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


def generate_command(dates: list[date]) -> str:
    """
    Generate a ready-to-run command with the sampled dates.
    """
    date_strs = [d.strftime("%Y%m%d") for d in dates]
    date_args = " ".join(date_strs)

    ins_path = r"D:\dequi\repositories\osltm\data\check_ins\daily"
    outs_path = r"D:\dequi\repositories\osltm\data\check_outs\daily"

    command = (
        f"uv run python -m src.utils.download_daily_data {date_args} "
        f'--type both --ins_path "{ins_path}" --outs_path "{outs_path}"'
    )

    return command


if __name__ == "__main__":
    start = date(2024, 6, 25)
    end = datetime.now().date() - timedelta(days=2)

    sampled = sample_stratified_days(start, end, n_per_stratum=2, random_seed=SEED)
    print(f"Sampled {len(sampled)} unique dates between {start} and {end}:")
    for d in sampled:
        print(f"  {d} ({get_day_type(d)})")

    print("\nREADY-TO-RUN COMMAND:")
    print(generate_command(sampled))
