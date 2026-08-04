"""Tests for qtcm1.basis: the vertical basis functions.

The construction is pinned at three levels: exact identities against the
model's own projection constants (surface values), the quadrature's
internal consistency (level-set independence at table nodes), and the
physical structure of V1 (monotone, one sign change, mid-troposphere
node).
"""

import numpy as np
import pytest

from qtcm1 import load_basis
from qtcm1.constants import (A1HAT, A1PHAT, A1S, T1C_TABLE, V1S, V1Z_TABLE)


def test_surface_values_match_model_constants():
    ds = load_basis()
    # table row 1 is the surface: a1(1000) = a1s, a1+(1000) = 0 exactly
    assert ds['a1'].sel(p=1000.0).item() == A1S
    assert ds['a1p'].sel(p=1000.0).item() == 0.0
    # V1(1000) = -a1phat = V1s -- the identity the model itself uses
    assert ds['V1'].sel(p=1000.0).item() == pytest.approx(V1S, abs=1e-15)
    # and the independent height-level table agrees at its surface row
    assert V1Z_TABLE[0, 1] == V1S


def test_native_levels_reproduce_table():
    ds = load_basis()
    np.testing.assert_array_equal(ds['p'].values, T1C_TABLE[:, 0])
    np.testing.assert_array_equal(ds['a1'].values, T1C_TABLE[:, 3])
    np.testing.assert_array_equal(ds['alpha'].values, T1C_TABLE[:, 1])
    np.testing.assert_array_equal(ds['Tsat'].values, T1C_TABLE[:, 2])


def test_table_means_reproduce_projection_constants():
    """11-level trapezoid p-means match a1phat/a1hat to table resolution.

    The tolerances are the measured discretization error of the coarse
    table against NZ's reference-profile projections (2.4% and 3.8%);
    they document fidelity, not bugs.
    """
    ds = load_basis()
    p = ds['p'].values
    depth = p[0] - p[-1]                                     # 800 hPa
    pm_a1p = np.trapezoid(ds['a1p'].values[::-1], p[::-1]) / depth
    pm_a1 = np.trapezoid(ds['a1'].values[::-1], p[::-1]) / depth
    assert pm_a1p == pytest.approx(A1PHAT, rel=0.03)
    assert pm_a1 == pytest.approx(A1HAT, rel=0.04)


def test_v1_physical_structure():
    ds = load_basis(levels=81)
    V1 = ds['V1'].values                     # descending p = ascending z
    assert np.all(np.diff(V1) > 0), 'V1 must increase monotonically upward'
    signs = np.sign(V1)
    assert signs[0] < 0 < signs[-1]
    assert np.sum(np.diff(signs) != 0) == 1, 'exactly one sign change'
    # the node sits in the mid-troposphere (~500 hPa on the native table)
    node_p = ds['p'].values[np.argmin(np.abs(V1))]
    assert 400.0 < node_p < 600.0


def test_quadrature_is_level_set_independent():
    """Values at table nodes must not depend on the requested level set."""
    native = load_basis()
    fine = load_basis(levels=np.union1d(
        T1C_TABLE[:, 0], np.linspace(210.0, 990.0, 53)))
    on_nodes = fine.sel(p=native['p'].values)
    for v in ('a1', 'a1p', 'V1'):
        np.testing.assert_array_equal(on_nodes[v].values, native[v].values)


def test_int_levels_and_bounds():
    ds = load_basis(levels=41)
    assert ds.sizes['p'] == 41
    assert ds['p'].values[0] == 1000.0 and ds['p'].values[-1] == 200.0
    with pytest.raises(ValueError, match='inside'):
        load_basis(levels=[150.0, 500.0])
    with pytest.raises(ValueError, match='inside'):
        load_basis(levels=[500.0, 1013.0])
    with pytest.raises(ValueError):
        load_basis(levels=1)


def test_height_table_rides_along():
    ds = load_basis()
    np.testing.assert_array_equal(ds['z'].values, V1Z_TABLE[:, 0])
    np.testing.assert_array_equal(ds['V1z'].values, V1Z_TABLE[:, 1])


def test_wind_reconstruction_broadcast():
    """u = u0 + V1(p) u1 broadcasts to (p, lat) and is exact at 1000 hPa."""
    import xarray as xr

    basis = load_basis(levels=21)
    lat = np.linspace(-60, 60, 7)
    u0 = xr.DataArray(np.linspace(-5, 5, lat.size), coords={'lat': lat})
    u1 = xr.DataArray(np.linspace(10, -10, lat.size), coords={'lat': lat})
    u = u0 + basis['V1'] * u1
    assert set(u.dims) == {'p', 'lat'}
    np.testing.assert_allclose(u.sel(p=1000.0).values,
                               (u0 + V1S * u1).values, rtol=0, atol=1e-14)
