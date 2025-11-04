"""
Utility for checking whether a given date is a public holiday in Colombia.
Uses the 'holidays' package, which includes movable holidays (per Law 51 of 1983 - Ley Emiliani).
"""

from datetime import date, datetime

import holidays

# Preload Colombian holidays for a wide range of years
_COLOMBIA_HOLIDAYS = holidays.Colombia(years=range(2022, 2026))


def is_colombian_holiday(dt: date | datetime) -> bool:
    """
    Return True if the given date is a Colombian public holiday.
    Works with either a datetime or a date object.
    """
    if isinstance(dt, datetime):
        dt = dt.date()
    return dt in _COLOMBIA_HOLIDAYS


def get_holiday_name(dt: date | datetime) -> str | None:
    """
    Return the name of the holiday if the date is a holiday, else None.
    """
    if isinstance(dt, datetime):
        dt = dt.date()
    return _COLOMBIA_HOLIDAYS.get(dt)


def list_colombian_holidays(year: int) -> dict[date, str]:
    """
    Return all Colombian holidays for a given year as {date: name}.
    Example:
        >>> list_colombian_holidays(2025)
        {datetime.date(2025, 1, 1): 'Año Nuevo', datetime.date(2025, 1, 6): 'Reyes Magos', ...}
    """
    return holidays.Colombia(years=[year])
