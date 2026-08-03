"""Boundary-data reader tests against the converted netCDF registry.

Set ``QTCM1_DATA`` to the registry directory (default: the session workspace
location). Tests are skipped when the registry is absent.
"""

import os

import numpy as np
import pytest

from qtcm1.calendar import ModelCalendar
from qtcm1.io.bnddata import BoundaryData

DATA = os.environ.get('QTCM1_DATA',
                      os.path.expanduser('~/work/data/qtcm1_bnd_r64x42'))

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, 'sst_reynolds_clim.nc')),
    reason='converted boundary-data registry not found')


@pytest.fixture(scope='module')
def bnd():
    return BoundaryData(DATA, ModelCalendar())


def test_bracket_and_jan2_weight(bnd):
    # Jan 2 sits between mid-December (t=-16) and mid-January (t=15):
    # slots (0, 1), fraction 18/31 -- the convention verified against the
    # compiled oracle (ocean Ts matched to <= 0.06 K with this weight).
    m1, m2, t1, t2 = bnd.bracket(2)
    assert (m1, m2, t1, t2) == (0, 1, -16, 15)
    assert (2 - t1) / (t2 - t1) == pytest.approx(18 / 31)


def test_seasonal_sst_at_anchors_is_exact(bnd):
    np.testing.assert_array_equal(bnd.sst(1, 15, 'seasonal'),
                                  bnd.sst_clim[0])       # mid-January
    np.testing.assert_array_equal(bnd.sst(1, 349, 'seasonal'),
                                  bnd.sst_clim[11])      # mid-December


def test_seasonal_sst_midpoint_is_average(bnd):
    mid = bnd.sst(1, 30, 'seasonal')                     # (30-15)/(45-15)=0.5
    np.testing.assert_allclose(mid, 0.5 * (bnd.sst_clim[0] + bnd.sst_clim[1]),
                               rtol=0, atol=1e-6)


def test_seasonal_sst_year_wrap_continuity(bnd):
    end_of_year = bnd.sst(1, 365, 'seasonal')            # between slots 12, 13
    start_of_year = bnd.sst(2, 0, 'seasonal')            # same physical time
    np.testing.assert_allclose(end_of_year, start_of_year, atol=0.5)


def test_realtime_sst_uses_dated_files(bnd):
    if bnd.sst_dated is None:
        pytest.skip('dated SST not in registry')
    jan1980 = bnd.sst_dated[bnd._dated_index[(1980, 1)]]
    np.testing.assert_array_equal(bnd.sst(1980, 15, 'real_time'), jan1980)
    # December wrap reaches into the previous year's file
    m1, _, t1, t2 = bnd.bracket(360)
    assert m1 == 12
    dec = bnd.sst(1980, 360, 'real_time')
    assert dec.shape == jan1980.shape


def test_sst_is_kelvin(bnd):
    sst = bnd.sst(1, 100, 'seasonal')
    assert 260.0 < sst.min() < 280.0
    assert 295.0 < sst.max() < 310.0


def test_stype_flags(bnd):
    assert set(np.unique(bnd.stype)) <= {0, 1, 2, 3}
    assert (bnd.stype == 0).mean() > 0.5                 # mostly ocean


def test_albedo_range_and_interp(bnd):
    alb = bnd.albedo(200)
    assert 0.0 < alb.min() and alb.max() <= 0.95
    np.testing.assert_array_equal(bnd.albedo(15), bnd.albedo_clim[0])
