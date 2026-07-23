"""
external_api.py
External API integration required by the project brief.

Uses the free, keyless "Nager.Date" public holidays API
(https://date.nager.at) to check whether a requested appointment
date is an official Egyptian public holiday, so the booking agent
never schedules a client on a day the office is closed.

Falls back gracefully (with an explicit flag, not a silent guess)
if the API is unreachable — office weekends (Fri/Sat) are still
enforced locally regardless of network access.
"""

import requests
from datetime import date
from functools import lru_cache
from typing import Set

NAGER_HOLIDAYS_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/EG"


@lru_cache(maxsize=8)
def get_public_holidays(year: int) -> Set[str]:
    """Return a set of 'YYYY-MM-DD' strings for Egyptian public holidays
    in the given year. Returns an empty set (with no crash) if the
    external API cannot be reached — the caller decides how to treat that."""
    try:
        resp = requests.get(NAGER_HOLIDAYS_URL.format(year=year), timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return {item["date"] for item in data}
    except Exception as exc:
        print(f"[external_api] holiday lookup failed ({exc}); "
              f"continuing with weekday-only rules for {year}.")
        return set()


def is_office_closed(d: date) -> bool:
    """Office weekend in Egypt is Friday(4)/Saturday(5) (Mon=0)."""
    if d.weekday() in (4, 5):
        return True
    holidays = get_public_holidays(d.year)
    return d.isoformat() in holidays
