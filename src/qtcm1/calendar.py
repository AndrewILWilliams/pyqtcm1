"""Model calendar: 365-day (default) or 360-day, no leap years.

Port of ``Module Calendar`` + ``TimeManager`` + ``julian`` (calendar.F90).
The former compile-time ``YEAR360`` option is the runtime argument
``days_per_year=360``.

The ``midmonth`` anchor table is the set of interpolation times used by the
boundary-data readers: entry ``m`` is the day-of-year of the middle of month
``m``, with wrap entries ``m=0`` (mid-December of the previous year, -16) and
``m=13`` (mid-January of the next year, 380). The Fortran defines it only for
the 365-day calendar; for 360 days we compute the analogous table
(``15 + 30*(m-1)``) and mark it as an extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MONLEN_365 = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONLEN_360 = np.full(12, 30)
#: midmonth(0:13) from calendar.F90 (365-day calendar)
_MIDMONTH_365 = np.array(
    [-16, 15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349, 380])


@dataclass(frozen=True)
class CalendarState:
    """The Fortran Calendar module's per-day time variables."""

    dayofmodel: int      #: days since run start, 1-based
    yearofmodel: int
    monthofyear: int     #: 1..12
    dayofmonth: int      #: 1..monlen(month)
    dayofyear: int       #: 1..days_per_year

    @property
    def dateofmodel(self) -> int:
        """Date encoded as yyyymmdd (Fortran ``dateofmodel``)."""
        return (self.yearofmodel * 10000 + self.monthofyear * 100
                + self.dayofmonth)


class ModelCalendar:
    """Fixed-length no-leap calendar anchored at (year0, month0, day0).

    Parameters mirror the Fortran ``Input``/``Calendar`` variables. Note the
    cold-start convention discovered in validation: the Fortran ``varinit``
    pins ``day0 = 1`` for non-restart runs; that normalization belongs to the
    model driver, not the calendar, and is *not* applied here.
    """

    def __init__(self, year0: int = 1, month0: int = 1, day0: int = 1,
                 days_per_year: int = 365):
        if days_per_year == 365:
            self.monlen = _MONLEN_365.copy()
            self.midmonth = _MIDMONTH_365.copy()
        elif days_per_year == 360:
            self.monlen = _MONLEN_360.copy()
            self.midmonth = np.concatenate(
                [[-15], 15 + 30 * np.arange(12), [375]])  # 360-day extension
        else:
            raise ValueError(f'days_per_year must be 365 or 360, '
                             f'got {days_per_year}')
        if not 1 <= month0 <= 12:
            raise ValueError(f'month0 out of range: {month0}')
        if not 1 <= day0 <= self.monlen[month0 - 1]:
            raise ValueError(f'day0 out of range for month {month0}: {day0}')
        self.days_per_year = days_per_year
        #: cummonth(m) = days in months before m (Fortran cummonth, 1-based m)
        self.cummonth = np.concatenate([[0], np.cumsum(self.monlen)[:-1]])
        self.year0, self.month0, self.day0 = year0, month0, day0

    # -- TimeManager (calendar.F90) ------------------------------------
    def timemanager(self, dayofmodel: int,
                    perpetual: bool = False) -> CalendarState:
        """Set the calendar for integration day ``dayofmodel`` (1-based).

        Exact port of ``TimeManager``; ``perpetual=True`` reproduces the
        ``SSTmode == 'perpetual'`` branch (clock frozen at the start date).
        """
        if perpetual:
            dayofyear = ((self.day0 - 1 + self.cummonth[self.month0 - 1])
                         % self.days_per_year + 1)
            return CalendarState(dayofmodel=dayofmodel,
                                 yearofmodel=self.year0,
                                 monthofyear=self.month0,
                                 dayofmonth=self.day0,
                                 dayofyear=int(dayofyear))

        day = (dayofmodel - 1 + self.cummonth[self.month0 - 1]
               + self.day0 - 1)                    # days since Jan 1, year0
        yearofmodel = day // self.days_per_year + self.year0
        dayofyear = day % self.days_per_year + 1
        monthofyear = 12
        while self.cummonth[monthofyear - 1] >= dayofyear:
            monthofyear -= 1
        dayofmonth = dayofyear - self.cummonth[monthofyear - 1]
        return CalendarState(dayofmodel=dayofmodel,
                             yearofmodel=int(yearofmodel),
                             monthofyear=int(monthofyear),
                             dayofmonth=int(dayofmonth),
                             dayofyear=int(dayofyear))

    # -- julian (calendar.F90) -----------------------------------------
    def julian(self, date: int) -> int:
        """Days of ``date`` (yyyymmdd) since the run reference date.

        Exact port of ``Integer Function julian`` including its convention
        that the result is relative to (year0, month0, day0).
        """
        year = date // 10000 - self.year0
        month = (date % 10000) // 100
        day = (date % 100) - self.day0 + 1
        return int(day + self.cummonth[month - 1]
                   - self.cummonth[self.month0 - 1]
                   + year * self.days_per_year)


def time_interp(prior: np.ndarray, upcoming: np.ndarray,
                t1: int, t2: int, now: float) -> np.ndarray:
    """Linear interpolation between two boundary-data snapshots.

    Exact port of ``TimeInterp`` (utilities.F90): result =
    prior + (now - t1)/(t2 - t1) * (upcoming - prior), degenerating to
    ``prior`` when ``t2 <= t1``. Times are in the ``midmonth``/julian-day
    convention of the caller.
    """
    divisor = t2 - t1
    if divisor > 0:
        fraction = (now - t1) / divisor
        return prior + fraction * (upcoming - prior)
    return prior.copy()
