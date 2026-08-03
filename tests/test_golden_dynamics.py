"""Tier 1 golden tests: dynamics routines vs the instrumented Fortran oracle.

Fixtures are produced by ``tools/gen_golden.py`` (full visible Fortran state
captured after every wrapped subroutine of one atmospheric time step, at four
seasonal warm states). Point ``QTCM1_FIXTURES`` at the fixture directory;
tests skip if absent.

Comparison policy: fixtures are float32 (the Fortran is single precision);
the port computes in float64 from the float32 inputs. Agreement is required
to ~float32 roundoff of each field's magnitude.
"""

import glob
import os

import numpy as np
import pytest

from qtcm1.grid import Grid
from qtcm1.dynamics.advection import advctTq, advctuv
from qtcm1.dynamics.diffusion import dffus

FIXDIR = os.environ.get('QTCM1_FIXTURES', os.path.expanduser('~/work/fixtures'))
FILES = sorted(glob.glob(os.path.join(FIXDIR, 'step_day*.npz')))

pytestmark = pytest.mark.skipif(not FILES, reason='golden fixtures not found')

GRID = Grid()

#: keys stored on the u/T grid with ghost rows (Fortran (nx, 0:ny+1))
_GHOSTED = {'u1', 'T1', 'q1', 'u0'}


def f2py_to_grid(key, arr):
    """Fortran-layout fixture array -> port convention ((rows, nx))."""
    a = np.asarray(arr).T                      # -> (rows, nx)
    if key in _GHOSTED:
        return a[1:-1]                         # strip ghost rows -> (ny, nx)
    return a


def load(fn, stage):
    z = np.load(fn)
    return {k.split('/', 1)[1]: z[k] for k in z.files
            if k.startswith(stage + '/')}


def assert_field(actual, expected, name, rtol=2e-5, atol_scale=1e-5):
    scale = np.abs(expected).max()
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol_scale * max(scale, 1e-30),
        err_msg=f'{name} (scale={scale:.3e})')


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_advctuv_golden(fn):
    pre = load(fn, 'wsland1')                  # state entering advctuv
    post = load(fn, 'wadvctuv')
    out = advctuv(f2py_to_grid('u1', pre['u1']).astype(np.float64),
                  f2py_to_grid('v1', pre['v1']).astype(np.float64),
                  f2py_to_grid('u0', pre['u0']).astype(np.float64),
                  f2py_to_grid('v0', pre['v0']).astype(np.float64),
                  GRID)
    for key in ['advu0', 'advu1', 'advwu0', 'advwu1', 'div1']:
        assert_field(out[key], f2py_to_grid(key, post[key]), key)
    for key in ['advv0', 'advv1']:             # Fortran writes rows 1..ny-1
        assert_field(out[key][1:-1], f2py_to_grid(key, post[key])[1:-1], key)


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_advctTq_golden(fn):
    pre = load(fn, 'wadvctuv')                 # state entering advctTq
    post = load(fn, 'wadvcttq')
    out = advctTq(f2py_to_grid('T1', pre['T1']).astype(np.float64),
                  f2py_to_grid('q1', pre['q1']).astype(np.float64),
                  f2py_to_grid('u1', pre['u1']).astype(np.float64),
                  f2py_to_grid('v1', pre['v1']).astype(np.float64),
                  f2py_to_grid('u0', pre['u0']).astype(np.float64),
                  f2py_to_grid('v0', pre['v0']).astype(np.float64),
                  GRID)
    for key in ['advT1', 'advq1']:
        assert_field(out[key], f2py_to_grid(key, post[key]), key)


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_dffus_golden(fn):
    pre = load(fn, 'wadvcttq')                 # state entering dffus
    post = load(fn, 'wdffus')
    out = dffus(f2py_to_grid('u1', pre['u1']).astype(np.float64),
                f2py_to_grid('v1', pre['v1']).astype(np.float64),
                f2py_to_grid('u0', pre['u0']).astype(np.float64),
                f2py_to_grid('v0', pre['v0']).astype(np.float64),
                f2py_to_grid('T1', pre['T1']).astype(np.float64),
                f2py_to_grid('q1', pre['q1']).astype(np.float64),
                GRID,
                viscxu1=float(pre['viscxu1']), viscyu1=float(pre['viscyu1']),
                visc4x=float(pre['visc4x']), visc4y=float(pre['visc4y']),
                viscxT=float(pre['viscxT']), viscyT=float(pre['viscyT']),
                viscxq=float(pre['viscxq']), viscyq=float(pre['viscyq']),
                viscxu0=float(pre['viscxu0']), viscyu0=float(pre['viscyu0']))
    for key in ['dfsu1', 'dfsv1', 'dfsu0', 'dfsv0', 'dfsT1', 'dfsq1']:
        assert_field(out[key], f2py_to_grid(key, post[key]), key)


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_prognostics_unchanged_by_tendency_routines(fn):
    """advctuv/advctTq/dffus only fill tendency arrays; prognostics frozen."""
    a = load(fn, 'wsland1')
    b = load(fn, 'wdffus')
    for key in ['u1', 'v1', 'T1', 'q1', 'u0', 'v0']:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


# ---------------------------------------------------------------------------
# Convection (physics/convection.py) -- first physics golden
# ---------------------------------------------------------------------------
from qtcm1.dynamics.filters import PolarFilter          # noqa: E402
from qtcm1.physics.convection import mconvct            # noqa: E402

PFILT = PolarFilter(GRID)


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_mconvct_golden(fn):
    pre = load(fn, 'pre')                      # mconvct is first in atm_step
    post = load(fn, 'wmconvct')
    out = mconvct(f2py_to_grid('T1', pre['T1']).astype(np.float64),
                  f2py_to_grid('q1', pre['q1']).astype(np.float64),
                  float(pre['eps_c']), PFILT)
    # tolerance note: a 1-ulp float32 difference in the T1c lookup tables
    # (~3e-5 K) is amplified by eps_c*Cpg (~1.2e3 W m-2 K-1) into ~0.04 W/m2
    # of Qc noise; atol_scale=1e-4 of the field max covers that floor while
    # still failing loudly (by 3+ orders) on any logic error.
    assert_field(out['Qc'], f2py_to_grid('Qc', post['Qc']), 'Qc',
                 atol_scale=1e-4)


def test_polar_filter_row_extent():
    """js from the Fortran single-precision expression; symmetric rows."""
    assert PFILT.js in (4, 5)                  # document the f32 truncation
    assert PFILT.jn0 == GRID.ny - PFILT.js


# ---------------------------------------------------------------------------
# Cloud, radiation, surface fluxes
# ---------------------------------------------------------------------------
from qtcm1.physics.clouds import cloud                   # noqa: E402
from qtcm1.physics.radiation import radlw, radsw         # noqa: E402
from qtcm1.physics.sfcflux import sfcwind_abl, sflux     # noqa: E402


def _dayofyear(fn):
    return int(os.path.basename(fn)[8:12])       # step_dayNNNN, year 1


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_cloud_golden(fn):
    pre = load(fn, 'wmconvct')
    post = load(fn, 'wcloud')
    out = cloud(f2py_to_grid('Qc', pre['Qc']).astype(np.float64))
    assert_field(out['cl1'], f2py_to_grid('cl1', post['cl1']), 'cl1')


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_radsw_golden(fn):
    pre = load(fn, 'wcloud')
    post = load(fn, 'wradsw')
    cld = cloud(f2py_to_grid('Qc', pre['Qc']).astype(np.float64))['cld']
    out = radsw(cld, f2py_to_grid('ALBDs', pre['ALBDs']).astype(np.float64),
                _dayofyear(fn), GRID)
    for key in ['S0', 'FSWds', 'FSWus', 'FSWut', 'FSW']:
        assert_field(out[key], f2py_to_grid(key, post[key]), key)


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_radlw_golden(fn):
    pre = load(fn, 'wradsw')
    post = load(fn, 'wradlw')
    cld = cloud(f2py_to_grid('Qc', pre['Qc']).astype(np.float64))['cld']
    out = radlw(f2py_to_grid('T1', pre['T1']).astype(np.float64),
                f2py_to_grid('q1', pre['q1']).astype(np.float64),
                f2py_to_grid('Ts', pre['Ts']).astype(np.float64), cld)
    for key in ['FLWds', 'FLWus', 'FLWut', 'FLW']:
        assert_field(out[key], f2py_to_grid(key, post[key]), key)


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_sflux_golden(fn):
    pre = load(fn, 'wradlw')                   # state entering Sflux
    post = load(fn, 'wsflux')
    wind = sfcwind_abl(
        f2py_to_grid('u1', pre['u1']).astype(np.float64),
        f2py_to_grid('v1', pre['v1']).astype(np.float64),
        f2py_to_grid('u0', pre['u0']).astype(np.float64),
        f2py_to_grid('v0', pre['v0']).astype(np.float64),
        f2py_to_grid('us', pre['us']).astype(np.float64),
        f2py_to_grid('vs', pre['vs']).astype(np.float64),
        f2py_to_grid('dphisdx', pre['dphisdx']).astype(np.float64),
        f2py_to_grid('dphisdy', pre['dphisdy']).astype(np.float64),
        f2py_to_grid('CDN', pre['CDN']).astype(np.float64),
        np.asarray(GRID.fu, dtype=np.float64),
        weml=float(pre['weml']), ziml=float(pre['ziml']),
        vvsmin=float(pre['VVsmin']))
    for key in ['us', 'vs']:
        assert_field(wind[key], f2py_to_grid(key, post[key]), key,
                     atol_scale=3e-5)          # Newton fixed point, f32 floor
    out = sflux(f2py_to_grid('T1', pre['T1']).astype(np.float64),
                f2py_to_grid('q1', pre['q1']).astype(np.float64),
                f2py_to_grid('Ts', pre['Ts']).astype(np.float64),
                f2py_to_grid('STYPE', pre['STYPE']),
                f2py_to_grid('CDN', pre['CDN']).astype(np.float64), wind)
    for key in ['CV', 'taux', 'Evap', 'FTs']:
        assert_field(out[key], f2py_to_grid(key, post[key]), key,
                     atol_scale=3e-5)
    assert_field(out['tauy'][:-1], f2py_to_grid('tauy', post['tauy'])[:-1],
                 'tauy', atol_scale=3e-5)


# ---------------------------------------------------------------------------
# Land, baroclinic and barotropic mode updates
# ---------------------------------------------------------------------------
from qtcm1.physics.land import sland1                    # noqa: E402
from qtcm1.dynamics.baroclinic import barcl              # noqa: E402
from qtcm1.dynamics.barotropic import bartr, gradphis    # noqa: E402
from qtcm1.dynamics.elliptic import PoissonDirichlet     # noqa: E402

POISSON = PoissonDirichlet(GRID)


def _g64(stage_dict, key):
    return f2py_to_grid(key, stage_dict[key]).astype(np.float64)


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_sland1_golden(fn):
    pre = load(fn, 'wsflux')
    post = load(fn, 'wsland1')
    out = sland1(_g64(pre, 'Ts'), _g64(pre, 'WD'),
                 f2py_to_grid('STYPE', pre['STYPE']),
                 _g64(pre, 'Qc'), _g64(pre, 'Evap'), _g64(pre, 'FTs'),
                 _g64(pre, 'FSWds'), _g64(pre, 'FSWus'),
                 _g64(pre, 'FLWds'), _g64(pre, 'FLWus'),
                 _g64(pre, 'CV'), float(pre['dt']))
    land = f2py_to_grid('STYPE', pre['STYPE']) != 0
    for key in ['Ts', 'WD', 'Evap', 'Evapi', 'wet', 'Runs', 'Runf']:
        exp = _g64(post, key)
        act = out[key]
        assert_field(np.where(land, act, exp), exp, key)
    # ocean untouched
    np.testing.assert_array_equal(out['Ts'][~land],
                                  _g64(pre, 'Ts')[~land])


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_barcl_golden(fn):
    pre = load(fn, 'wdffus')
    post = load(fn, 'wbarcl')
    out = barcl(_g64(pre, 'u1'), _g64(pre, 'v1'), _g64(pre, 'T1'),
                _g64(pre, 'q1'),
                taux=_g64(pre, 'taux'), tauy=_g64(pre, 'tauy'),
                advu1=_g64(pre, 'advu1'), advv1=_g64(pre, 'advv1'),
                advT1=_g64(pre, 'advT1'), advq1=_g64(pre, 'advq1'),
                dfsu1=_g64(pre, 'dfsu1'), dfsv1=_g64(pre, 'dfsv1'),
                dfsT1=_g64(pre, 'dfsT1'), dfsq1=_g64(pre, 'dfsq1'),
                Qc=_g64(pre, 'Qc'), FSW=_g64(pre, 'FSW'),
                FLW=_g64(pre, 'FLW'), FTs=_g64(pre, 'FTs'),
                Evap=_g64(pre, 'Evap'),
                grid=GRID, polar_filter=PFILT, dt=float(pre['dt']))
    for key in ['u1', 'v1', 'T1', 'q1', 'div1', 'GMs1', 'GMq1']:
        assert_field(out[key], _g64(post, key), key)


def _identify_ab3(pre_r, post_r):
    """Return (rhs_written, [prev1, prev2] candidates) from AB3 slices."""
    delta = [np.abs(post_r[..., s] - pre_r[..., s]).max() for s in range(3)]
    w = int(np.argmax(delta))
    others = [s for s in range(3) if s != w]
    return w, others


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_bartr_golden(fn):
    pre = load(fn, 'wsavebartr')
    post = load(fn, 'wbartr')
    pre_r, post_r = pre['rhsvort0'], post['rhsvort0']    # (nx, ny, 3)
    w, others = _identify_ab3(pre_r, post_r)

    common = dict(taux=_g64(pre, 'taux'), tauy=_g64(pre, 'tauy'),
                  advu0=_g64(pre, 'advu0'), advv0=_g64(pre, 'advv0'),
                  dfsu0=_g64(pre, 'dfsu0'), dfsv0=_g64(pre, 'dfsv0'),
                  grid=GRID, polar_filter=PFILT, poisson=POISSON,
                  dt=float(pre['dt']), mt0=1)
    exp_vort = _g64(post, 'vort0')
    best = None
    for o1, o2 in (others, others[::-1]):
        hist = [pre_r[..., o1].T.astype(np.float64)[: GRID.ny - 1],
                pre_r[..., o2].T.astype(np.float64)[: GRID.ny - 1]]
        bhist = [float(pre['rhsu0bar'][o1]), float(pre['rhsu0bar'][o2])]
        out = bartr(_g64(pre, 'vort0'), float(pre['u0bar']),
                    _g64(pre, 'v0'), hist, bhist, **common)
        err = np.abs(out['vort0'] - exp_vort).max()
        if best is None or err < best[0]:
            best = (err, out)
    err, out = best
    # my new rhs must match the slice the Fortran wrote
    assert_field(out['rhs_hist'][0][: GRID.ny - 1],
                 post_r[..., w].T.astype(np.float64)[: GRID.ny - 1],
                 'rhsvort0[new]')
    # tolerance note: psi0 is O(1e8) in float32, and u0/v0 differentiate
    # it - cancellation amplifies the f32 storage/solver roundoff to a few
    # 1e-5 relative. Logic errors show up orders of magnitude above this.
    for key in ['vort0', 'psi0', 'u0', 'v0']:
        assert_field(out[key], _g64(post, key), key, atol_scale=5e-5)
    assert abs(out['u0bar'] - float(post['u0bar'])) <= \
        max(1e-5 * abs(float(post['u0bar'])), 1e-8)


@pytest.mark.parametrize('fn', FILES, ids=[os.path.basename(f) for f in FILES])
def test_gradphis_golden(fn):
    sav = load(fn, 'wsavebartr')               # winds entering bartr
    pre = load(fn, 'wbartr')                   # state entering gradphis
    post = load(fn, 'wgradphis')
    out = gradphis(_g64(pre, 'u0'), _g64(pre, 'v0'),
                   _g64(sav, 'u0'), _g64(sav, 'v0')[1:],
                   _g64(pre, 'T1'),
                   taux=_g64(pre, 'taux'), tauy=_g64(pre, 'tauy'),
                   advu0=_g64(pre, 'advu0'), advv0=_g64(pre, 'advv0'),
                   dfsu0=_g64(pre, 'dfsu0'), dfsv0=_g64(pre, 'dfsv0'),
                   grid=GRID, dt=float(pre['dt']), mt0=1)
    for key in ['dphisdx', 'dphisdy']:
        assert_field(out[key], _g64(post, key), key)
    assert_field(out['ps'], _g64(post, 'ps'), 'ps', atol_scale=3e-5)
