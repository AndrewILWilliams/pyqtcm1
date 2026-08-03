"""Golden test: netCDF boundary forcing vs the double-precision oracle.

The daily SST and albedo produced by :class:`qtcm1.io.bnddata.BoundaryData`
must be bitwise-equal to the fields the r8 Fortran integrated with
(captured immediately after its single daily getbnd by
tools/gen_trajectory.py). The 30-day window crosses the February-15
mid-month bracket advance, which exercises the julian-anchor arithmetic
and the bracket state machine.

Pins the anchor-convention finding: the interpolation anchors are
julian(yyyy mm 15) = cumulative month lengths + 15, NOT calendar.F90's
``midmonth`` table -- the two disagree for February (45 vs 46).
"""

import os

import numpy as np
import pytest

from qtcm1.io.bnddata import BoundaryData

_DATA = os.path.expanduser(os.environ.get(
    'QTCM1_BNDDATA', '~/work/data/qtcm1_bnd_r64x42'))
_CTRL = os.path.expanduser('~/work/trajectory_r8/control.npz')

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(_DATA) and os.path.exists(_CTRL)),
    reason='boundary registry / r8 trajectory not found')


@pytest.fixture(scope='module')
def bd():
    return BoundaryData(_DATA)


def test_anchor_convention(bd):
    """February anchor is julian(0215)=46; the midmonth table's 45 is a
    different (unused-here) convention carried by the Fortran."""
    assert bd._anchors.tolist() == [-16, 15, 46, 74, 105, 135, 166, 196,
                                    227, 258, 288, 319, 349, 380]
    assert bd.calendar.midmonth[2] == 45          # the Fortran table, as-is


def test_daily_forcing_bitwise_vs_r8_oracle(bd):
    ctrl = np.load(_CTRL)
    for day in range(33, 63):
        ts0 = np.asarray(ctrl[f'forcing/d{day:03d}/Ts0']).T
        alb = np.asarray(ctrl[f'forcing/d{day:03d}/ALBDs']).T
        np.testing.assert_array_equal(bd.sst(1, day), ts0,
                                      err_msg=f'SST day {day}')
        np.testing.assert_array_equal(bd.albedo(day), alb,
                                      err_msg=f'albedo day {day}')


def test_bracket_advance_day():
    """The Feb-15 bracket switch happens ON day 46 (>= rule)."""
    bd = BoundaryData(_DATA)
    assert bd.bracket(45)[2:] == (15, 46)
    assert bd.bracket(46)[2:] == (46, 74)
