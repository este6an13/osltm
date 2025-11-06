from datetime import date
from typing import Literal

from src.utils.colombian_holidays import is_colombian_holiday


def get_day_type(dt: date) -> Literal["WD", "SA", "SU", "HO"]:
    """
    Classify a date as weekday / saturday / sunday / holiday.
    Holidays override weekday/weekend classification.
    """
    if is_colombian_holiday(dt):
        return "HO"
    elif dt.weekday() == 5:
        return "SA"
    elif dt.weekday() == 6:
        return "SU"
    else:
        return "WD"
