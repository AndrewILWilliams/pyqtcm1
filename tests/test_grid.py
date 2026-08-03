"""Grid geometry tests: coordinates, symmetries, conservation identities."""

import numpy as np
import pytest

from qtcm1.grid import Grid


@pytest.fixture(scope='module')
def g():
    return Grid()


def test_t_point_coordinates(g):
    # T-point coordinates as written by the Fortran netCDF output
    # (lat -76.875..76.875 step 3.75; lon 0..354.375 step 5.625)
    np.testing.assert_allclose(g.latt, -76.875 + 3.75 * np.arange(42))
    np.testing.assert_allclose(g.lont, 5.625 * np.arange(64))
    assert g.latt[21] == pytest.approx(1.875)     # first NH row (paper plots)


def test_grid_spacings(g):
    assert g.dx == pytest.approx(2 * np.pi * 6.37e6 / 64)
    assert g.dy == pytest.approx(78.75 / 90 * np.pi * 6.37e6 / 42)
    assert g.dxi * g.dx == pytest.approx(1.0)


def test_coriolis_symmetries(g):
    # fu odd, cosu even about the equator; fv odd on the staggered rows
    np.testing.assert_allclose(g.fu, -g.fu[::-1], atol=1e-18)
    np.testing.assert_allclose(g.cosu, g.cosu[::-1])
    np.testing.assert_allclose(g.fv, -g.fv[::-1], atol=1e-18)
    # equatorial v-row sits exactly on the equator: fv = 0 there
    assert abs(g.fv[20]) < 1e-18 or abs(g.fv[21]) < 1e-18


def test_metric_consistency(g):
    np.testing.assert_allclose(g.dxui, 1.0 / (g.cosu * g.dx))
    np.testing.assert_allclose(g.dyvi, 1.0 / (g.cosv * g.dy))


def test_diffusion_weight_row_sums_vanish(g):
    """Each weight row sums to zero => diffusion conserves a constant field.

    This is the property stated in parinit: for constant T the 2nd/4th-order
    diffusion operators vanish identically.
    """
    ny = g.ny
    for jf in range(2, ny):              # weight2u/4u defined on j=2..ny-1
        assert abs(g.weight2u[jf].sum()) < 1e-12 * abs(g.weight2u[jf]).max()
        assert abs(g.weight4u[jf].sum()) < 1e-12 * abs(g.weight4u[jf]).max()
    for jf in range(1, ny):              # weight2v/4v defined on j=1..ny-1
        assert abs(g.weight2v[jf].sum()) < 1e-12 * abs(g.weight2v[jf]).max()
        assert abs(g.weight4v[jf].sum()) < 1e-12 * abs(g.weight4v[jf]).max()


def test_weight_formulas_spot_values(g):
    """Direct transcription check of the parinit weight expressions."""
    jf = 10                               # arbitrary interior Fortran row
    cosu = np.concatenate([[np.nan], g.cosu])   # 1-based view
    cosv = g.cosv                                # already 0..ny
    assert g.weight2u[jf, 0] == pytest.approx(cosv[jf] / cosu[jf])
    assert g.weight2u[jf, 1] == pytest.approx(
        -(cosv[jf] + cosv[jf - 1]) / cosu[jf])
    assert g.weight2v[jf, 2] == pytest.approx(cosu[jf] / cosv[jf])


def test_periodic_index_arrays(g):
    i = np.arange(g.nx)
    assert g.im1[0] == g.nx - 1 and g.ip1[-1] == 0
    assert g.im2[0] == g.nx - 2 and g.im2[1] == g.nx - 1
    x = np.random.default_rng(0).standard_normal(g.nx)
    np.testing.assert_array_equal(x[g.ip1][g.im1], x)
