"""Calendar tests, pinned to dates printed by the compiled Fortran model.

The fixture pairs (dayofmodel -> dateofmodel) below are copied verbatim from
run logs of the oracle build in this project's sessions (AMIP run and the
three 45-year control members with staggered start months).
"""

import pytest

import numpy as np

from qtcm1.calendar import ModelCalendar, time_interp


def test_oracle_dates_amip_and_controls():
    # (year0, month0, day0), dayofmodel -> dateofmodel, verbatim from logs
    # of the compiled Fortran model (AMIP run; control members a/b/c).
    cases = [
        ((1980, 1, 1), 365, 19801231),
        ((1980, 1, 1), 1044, 19821110),
        ((1980, 1, 1), 4380, 19911231),
        ((1, 1, 1), 2897, 81208),      # log: 00081208
        ((1, 1, 1), 12967, 360711),    # log: 00360711
        ((1, 1, 1), 16425, 451231),    # end of member a
        ((1, 2, 1), 7781, 220527),     # member b (month0=2)
        ((1, 2, 1), 16425, 460131),    # end of member b
        ((1, 3, 1), 7843, 220825),     # member c (month0=3)
        ((1, 3, 1), 16425, 460228),    # end of member c
    ]
    for (y0, m0, d0), day, expected in cases:
        cal = ModelCalendar(year0=y0, month0=m0, day0=d0)
        state = cal.timemanager(day)
        assert state.dateofmodel == expected, \
            f'start={y0}-{m0}-{d0} day={day}'


def test_julian_roundtrip():
    for (y0, m0) in [(1980, 1), (1, 2), (1, 3)]:
        cal = ModelCalendar(year0=y0, month0=m0, day0=1)
        for day in (1, 59, 365, 366, 4380, 16425):
            state = cal.timemanager(day)
            assert cal.julian(state.dateofmodel) == day


def test_day_one_is_start_date():
    cal = ModelCalendar(year0=7, month0=11, day0=1)
    s = cal.timemanager(1)
    assert (s.yearofmodel, s.monthofyear, s.dayofmonth) == (7, 11, 1)


def test_perpetual_freezes_clock():
    cal = ModelCalendar(year0=3, month0=6, day0=15)
    for day in (1, 100, 1000):
        s = cal.timemanager(day, perpetual=True)
        assert (s.yearofmodel, s.monthofyear, s.dayofmonth) == (3, 6, 15)


def test_year360():
    cal = ModelCalendar(days_per_year=360)
    s = cal.timemanager(360)
    assert (s.yearofmodel, s.monthofyear, s.dayofmonth) == (1, 12, 30)
    s = cal.timemanager(361)
    assert (s.yearofmodel, s.monthofyear, s.dayofmonth) == (2, 1, 1)


def test_midmonth_table_matches_fortran():
    cal = ModelCalendar()
    assert cal.midmonth.tolist() == [-16, 15, 45, 74, 105, 135, 166,
                                     196, 227, 258, 288, 319, 349, 380]


def test_time_interp_formula():
    a, b = np.zeros((2, 2)), np.full((2, 2), 31.0)
    # Jan 2 between Dec 15 (t=-16) and Jan 15 (t=15): fraction 18/31
    out = time_interp(a, b, -16, 15, 2.0)
    np.testing.assert_allclose(out, 18.0)
    # degenerate interval returns prior
    np.testing.assert_allclose(time_interp(a, b, 15, 15, 20.0), a)
