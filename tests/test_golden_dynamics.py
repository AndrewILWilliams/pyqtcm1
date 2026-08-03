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
