"""Tests for qtcm1.surface and the custom-surface plumbing."""

import os
import warnings

import numpy as np
import pytest

from qtcm1 import surface
from qtcm1.config import PACKAGED_DATA, RunConfig
from qtcm1.io.bnddata import BoundaryData

_DATA = os.environ.get('QTCM1_BNDDATA', PACKAGED_DATA)

pytestmark = pytest.mark.skipif(not os.path.isdir(_DATA),
                                reason='boundary registry not found')


def test_real_earth_and_aquaplanet():
    re = surface.real_earth(_DATA)
    aq = surface.aquaplanet(_DATA)
    assert re['stype'].shape == aq['stype'].shape == (42, 64)
    assert set(np.unique(re['stype'].values)) == {0, 1, 2, 3}
    assert (aq['stype'].values == 0).all()
    assert (aq['top'].values == 0.0).all()


def test_paint_box_and_wraparound():
    aq = surface.aquaplanet(_DATA)
    ds = surface.paint(aq, lon=(175.0, 220.0), lat=(-12.0, 12.0),
                       stype=surface.GRASS, top=0.2)
    st, top = ds['stype'].values, ds['top'].values
    lats, lons = aq['lat'].values, aq['lon'].values
    box = ((np.abs(lats) <= 12.0)[:, None]
           & ((lons >= 175.0) & (lons <= 220.0))[None, :])
    assert (st[box] == surface.GRASS).all()
    assert (st[~box] == 0).all()
    assert (top[box] == 0.2).all() and (top[~box] == 0.0).all()
    # dateline-crossing box wraps
    ds2 = surface.paint(aq, lon=(350.0, 10.0), stype=surface.DESERT)
    st2 = ds2['stype'].values
    sel = (lons >= 350.0) | (lons <= 10.0)
    assert (st2[:, sel] == surface.DESERT).all()
    assert (st2[:, ~sel] == 0).all()
    # the input is untouched
    assert (aq['stype'].values == 0).all()


def test_paint_mask_and_errors():
    aq = surface.aquaplanet(_DATA)
    mask = np.zeros((42, 64), dtype=bool)
    mask[20, 30] = True
    ds = surface.paint(aq, mask=mask, stype=surface.FOREST)
    assert ds['stype'].values[20, 30] == surface.FOREST
    assert ds['stype'].values.sum() == surface.FOREST
    with pytest.raises(ValueError, match='not both'):
        surface.paint(aq, lon=(0, 10), mask=mask, stype=1)
    with pytest.raises(ValueError, match='stype'):
        surface.paint(aq, lon=(0, 10), stype=7)


def test_validate_catches_structural_errors():
    ok_st = np.zeros((42, 64), dtype=np.int16)
    ok_top = np.zeros((42, 64))
    surface.validate(ok_st, ok_top, (42, 64))
    with pytest.raises(ValueError, match='shape'):
        surface.validate(ok_st[:-1], ok_top, (42, 64))
    with pytest.raises(ValueError, match='outside'):
        surface.validate(ok_st + 9, ok_top, (42, 64))
    with pytest.raises(ValueError, match='height/10km'):
        surface.validate(ok_st, ok_top + 5.5, (42, 64))   # km passed


def test_boundarydata_override_and_hybrid_albedo():
    ref = BoundaryData(_DATA)
    surf = surface.paint(surface.real_earth(_DATA),
                         lon=(175.0, 220.0), lat=(-12.0, 12.0),
                         stype=surface.GRASS)
    bd = BoundaryData(_DATA, surface=surf, albedo_mode='auto')
    changed = bd.stype != ref.stype
    assert changed.any()
    assert bd.surface_sha256 is not None
    alb_ref, alb = ref.albedo(100), bd.albedo(100)
    # unchanged points keep the Darnell climatology bitwise
    np.testing.assert_array_equal(alb[~changed], alb_ref[~changed])
    # changed points carry the static grassland albedo
    grass_alb = float(ref.albedo_annual[ref.stype == surface.GRASS].mean())
    assert np.allclose(alb[changed], grass_alb)
    # by_stype: static everywhere; darnell: climatology everywhere
    bd2 = BoundaryData(_DATA, surface=surf, albedo_mode='by_stype')
    assert np.ptp(bd2.albedo(1)[bd2.stype == surface.GRASS]) == 0.0
    bd3 = BoundaryData(_DATA, surface=surf, albedo_mode='darnell')
    np.testing.assert_array_equal(bd3.albedo(100), alb_ref)


def test_new_ocean_warns_and_slab_refuses():
    surf = surface.aquaplanet(_DATA)                 # all former land -> ocean
    with pytest.warns(UserWarning, match='under-land'):
        BoundaryData(_DATA, surface=surf)
    from qtcm1.driver import ControlRun
    with pytest.raises(ValueError, match='make_qflux'):
        ControlRun(config=RunConfig(data_path=_DATA,
                                    sst_mode='mixed_layer'),
                   surface=surf)


def test_coerce_roundtrip(tmp_path):
    surf = surface.real_earth(_DATA)
    st, top = surface.coerce(surf)
    st2, top2 = surface.coerce({'stype': st, 'top': top})
    np.testing.assert_array_equal(st, st2)
    p = str(tmp_path / 's.nc')
    surf.to_netcdf(p)
    st3, top3 = surface.coerce(p)
    np.testing.assert_array_equal(st, st3)
    np.testing.assert_array_equal(top, top3)
    with pytest.raises(TypeError):
        surface.coerce(42)
