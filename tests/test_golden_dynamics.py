"""Tier 1 golden tests: ported routines vs the instrumented Fortran oracle.

Fixtures are produced by ``tools/gen_golden.py`` (full visible Fortran state
captured after every wrapped subroutine of one atmospheric time step). Two
fixture sets are supported, and every test runs against whichever are
present:

* ``QTCM1_FIXTURES`` (default ``~/work/fixtures``): the **standard
  single-precision build**, at four seasonal warm states. The port runs
  with float32-mode init constants (polar filter, hsat/T1c tables) and
  float64 arithmetic on the float32 inputs, so agreement is required to
  ~float32 roundoff of each field's magnitude; a few documented tolerances
  are wider where f32 table/storage noise is amplified.
* ``QTCM1_FIXTURES_R8`` (default ``~/work/fixtures_r8``): the same Fortran
  compiled with 8-byte reals (``-fdefault-real-8``). The port shares every
  bit of state and every init constant with this build, so agreement is
  required at solver/libm roundoff (~1e-13 relative) -- this is the sharp
  equation-level test; any port bug fails it by many orders of magnitude.

Tests skip if neither fixture set is found.
"""

import glob
import os

import numpy as np
import pytest

from qtcm1.grid import Grid
from qtcm1.dynamics.advection import advctTq, advctuv
from qtcm1.dynamics.baroclinic import barcl
from qtcm1.dynamics.barotropic import bartr, gradphis
from qtcm1.dynamics.diffusion import dffus
from qtcm1.dynamics.elliptic import PoissonDirichlet
from qtcm1.dynamics.filters import PolarFilter
from qtcm1.physics.clouds import cloud
from qtcm1.physics.convection import get_tables, mconvct
from qtcm1.physics.land import sland1
from qtcm1.physics.radiation import radlw, radsw
from qtcm1.physics.sfcflux import sfcwind_abl, sflux

FIXDIR = os.path.abspath(os.path.expanduser(
    os.environ.get('QTCM1_FIXTURES', '~/work/fixtures')))
FIXDIR_R8 = os.path.abspath(os.path.expanduser(
    os.environ.get('QTCM1_FIXTURES_R8', '~/work/fixtures_r8')))
FILES32 = sorted(glob.glob(os.path.join(FIXDIR, 'step_day*.npz')))
FILES_R8 = sorted(glob.glob(os.path.join(FIXDIR_R8, 'step_day*.npz')))
FILES = FILES32 + FILES_R8

pytestmark = pytest.mark.skipif(not FILES, reason='golden fixtures not found')

GRID = Grid()
POISSON = PoissonDirichlet(GRID)

#: per-oracle init constants: like must be compared with like
_SETUP = {
    'f32': dict(pfilter=PolarFilter(GRID, dtype=np.float32),
                tables=get_tables(np.float32)),
    'r8': dict(pfilter=PolarFilter(GRID, dtype=np.float64),
               tables=get_tables(np.float64)),
}


def mode_of(fn):
    return 'r8' if os.path.abspath(os.path.dirname(fn)) == FIXDIR_R8 else 'f32'


_IDS = [f'{mode_of(f)}-{os.path.basename(f)}' for f in FILES]

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


#: r8-mode tolerances: the audited max rel diff over all routines is
#: <= ~1e-13 (elliptic-solver FFT roundoff); 1e-11 gives two orders of
#: headroom while still catching any real defect (which shows at >=1e-6).
R8_RTOL = 1e-11
R8_ATOL_SCALE = 1e-12


def assert_field(actual, expected, name, mode='f32', rtol=None,
                 atol_scale=None):
    """Compare with per-oracle tolerance defaults.

    ``rtol``/``atol_scale`` override the f32 defaults only (documented f32
    noise floors); r8 mode always uses the tight uniform tolerances.
    """
    if mode == 'r8':
        rtol, atol_scale = R8_RTOL, R8_ATOL_SCALE
    else:
        rtol = 2e-5 if rtol is None else rtol
        atol_scale = 1e-5 if atol_scale is None else atol_scale
    scale = np.abs(expected).max()
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol_scale * max(scale, 1e-30),
        err_msg=f'{name} (scale={scale:.3e}, mode={mode})')


def _g64(stage_dict, key):
    return f2py_to_grid(key, stage_dict[key]).astype(np.float64)


# ---------------------------------------------------------------------------
# Tendency routines
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_advctuv_golden(fn):
    mode = mode_of(fn)
    pre = load(fn, 'wsland1')                  # state entering advctuv
    post = load(fn, 'wadvctuv')
    out = advctuv(_g64(pre, 'u1'), _g64(pre, 'v1'),
                  _g64(pre, 'u0'), _g64(pre, 'v0'), GRID)
    for key in ['advu0', 'advu1', 'advwu0', 'advwu1', 'div1']:
        assert_field(out[key], _g64(post, key), key, mode)
    for key in ['advv0', 'advv1']:             # Fortran writes rows 1..ny-1
        assert_field(out[key][1:-1], _g64(post, key)[1:-1], key, mode)


@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_advctTq_golden(fn):
    mode = mode_of(fn)
    pre = load(fn, 'wadvctuv')                 # state entering advctTq
    post = load(fn, 'wadvcttq')
    out = advctTq(_g64(pre, 'T1'), _g64(pre, 'q1'),
                  _g64(pre, 'u1'), _g64(pre, 'v1'),
                  _g64(pre, 'u0'), _g64(pre, 'v0'), GRID)
    for key in ['advT1', 'advq1']:
        assert_field(out[key], _g64(post, key), key, mode)


@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_dffus_golden(fn):
    mode = mode_of(fn)
    pre = load(fn, 'wadvcttq')                 # state entering dffus
    post = load(fn, 'wdffus')
    out = dffus(_g64(pre, 'u1'), _g64(pre, 'v1'),
                _g64(pre, 'u0'), _g64(pre, 'v0'),
                _g64(pre, 'T1'), _g64(pre, 'q1'), GRID,
                viscxu1=float(pre['viscxu1']), viscyu1=float(pre['viscyu1']),
                visc4x=float(pre['visc4x']), visc4y=float(pre['visc4y']),
                viscxT=float(pre['viscxT']), viscyT=float(pre['viscyT']),
                viscxq=float(pre['viscxq']), viscyq=float(pre['viscyq']),
                viscxu0=float(pre['viscxu0']), viscyu0=float(pre['viscyu0']))
    for key in ['dfsu1', 'dfsv1', 'dfsu0', 'dfsv0', 'dfsT1', 'dfsq1']:
        assert_field(out[key], _g64(post, key), key, mode)


@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_prognostics_unchanged_by_tendency_routines(fn):
    """advctuv/advctTq/dffus only fill tendency arrays; prognostics frozen."""
    a = load(fn, 'wsland1')
    b = load(fn, 'wdffus')
    for key in ['u1', 'v1', 'T1', 'q1', 'u0', 'v0']:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


# ---------------------------------------------------------------------------
# Convection and polar filter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_mconvct_golden(fn):
    mode = mode_of(fn)
    pre = load(fn, 'pre')                      # mconvct is first in atm_step
    post = load(fn, 'wmconvct')
    out = mconvct(_g64(pre, 'T1'), _g64(pre, 'q1'), float(pre['eps_c']),
                  _SETUP[mode]['pfilter'], tables=_SETUP[mode]['tables'])
    # f32 tolerance note: a 1-ulp float32 difference in the T1c lookup
    # tables (~3e-5 K) is amplified by eps_c*Cpg (~1.2e3 W m-2 K-1) into
    # ~0.04 W/m2 of Qc noise; atol_scale=1e-4 of the field max covers that
    # floor while still failing loudly (by 3+ orders) on any logic error.
    assert_field(out['Qc'], _g64(post, 'Qc'), 'Qc', mode, atol_scale=1e-4)


def test_polar_filter_row_extent():
    """The filtered-row count sits on a truncation knife edge.

    (1 - 60/YB)*ny/2 is exactly 5 for the standard grid; the REAL*4 build
    computes 4.99999952 -> js=4, an r8 build 5.000000000000001 -> js=5.
    Pin both, and the north/south symmetry.
    """
    pf32 = PolarFilter(GRID, dtype=np.float32)
    pf64 = PolarFilter(GRID, dtype=np.float64)
    assert pf32.js == 4
    assert pf64.js == 5
    for pf in (pf32, pf64):
        assert pf.jn0 == GRID.ny - pf.js


# ---------------------------------------------------------------------------
# Cloud, radiation, surface fluxes
# ---------------------------------------------------------------------------
def _dayofyear(fn):
    return int(os.path.basename(fn)[8:12])       # step_dayNNNN, year 1


@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_cloud_golden(fn):
    mode = mode_of(fn)
    pre = load(fn, 'wmconvct')
    post = load(fn, 'wcloud')
    out = cloud(_g64(pre, 'Qc'))
    assert_field(out['cl1'], _g64(post, 'cl1'), 'cl1', mode)


@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_radsw_golden(fn):
    mode = mode_of(fn)
    pre = load(fn, 'wcloud')
    post = load(fn, 'wradsw')
    cld = cloud(_g64(pre, 'Qc'))['cld']
    out = radsw(cld, _g64(pre, 'ALBDs'), _dayofyear(fn), GRID)
    for key in ['S0', 'FSWds', 'FSWus', 'FSWut', 'FSW']:
        assert_field(out[key], _g64(post, key), key, mode)


@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_radlw_golden(fn):
    mode = mode_of(fn)
    pre = load(fn, 'wradsw')
    post = load(fn, 'wradlw')
    cld = cloud(_g64(pre, 'Qc'))['cld']
    out = radlw(_g64(pre, 'T1'), _g64(pre, 'q1'), _g64(pre, 'Ts'), cld)
    for key in ['FLWds', 'FLWus', 'FLWut', 'FLW']:
        assert_field(out[key], _g64(post, key), key, mode)


@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_sflux_golden(fn):
    mode = mode_of(fn)
    pre = load(fn, 'wradlw')                   # state entering Sflux
    post = load(fn, 'wsflux')
    wind = sfcwind_abl(
        _g64(pre, 'u1'), _g64(pre, 'v1'), _g64(pre, 'u0'), _g64(pre, 'v0'),
        _g64(pre, 'us'), _g64(pre, 'vs'),
        _g64(pre, 'dphisdx'), _g64(pre, 'dphisdy'),
        _g64(pre, 'CDN'), np.asarray(GRID.fu, dtype=np.float64),
        weml=float(pre['weml']), ziml=float(pre['ziml']),
        vvsmin=float(pre['VVsmin']))
    for key in ['us', 'vs']:
        assert_field(wind[key], _g64(post, key), key, mode,
                     atol_scale=3e-5)          # f32: Newton fixed point floor
    out = sflux(_g64(pre, 'T1'), _g64(pre, 'q1'), _g64(pre, 'Ts'),
                f2py_to_grid('STYPE', pre['STYPE']), _g64(pre, 'CDN'),
                wind, tables=_SETUP[mode]['tables'])
    for key in ['CV', 'taux', 'Evap', 'FTs']:
        assert_field(out[key], _g64(post, key), key, mode, atol_scale=3e-5)
    assert_field(out['tauy'][:-1], _g64(post, 'tauy')[:-1], 'tauy', mode,
                 atol_scale=3e-5)


# ---------------------------------------------------------------------------
# Land, baroclinic and barotropic mode updates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_sland1_golden(fn):
    mode = mode_of(fn)
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
        assert_field(np.where(land, act, exp), exp, key, mode)
    # ocean untouched
    np.testing.assert_array_equal(out['Ts'][~land], _g64(pre, 'Ts')[~land])


@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_barcl_golden(fn):
    mode = mode_of(fn)
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
                grid=GRID, polar_filter=_SETUP[mode]['pfilter'],
                dt=float(pre['dt']))
    for key in ['u1', 'v1', 'T1', 'q1', 'div1', 'GMs1', 'GMq1']:
        assert_field(out[key], _g64(post, key), key, mode)


def _identify_ab3(pre_r, post_r):
    """Return (written slice, [r_{n-1}, r_{n-2}] slices) of the AB3 history.

    The written slice is the one that changed; the Fortran rotates slots
    write->oldest (qtcm.F90: ktemp=k_2; k_2=k_1; k_1=k; k=ktemp), so the
    previous write sits at (w+1)%3 and the one before at (w+2)%3. (An
    ascending-order guess here is wrong for one phase in three - it showed
    up as a period-3 bartr failure in the day-46 step bisection.)
    """
    delta = [np.abs(post_r[..., s] - pre_r[..., s]).max() for s in range(3)]
    w = int(np.argmax(delta))
    return w, [(w + 1) % 3, (w + 2) % 3]


@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_bartr_golden(fn):
    mode = mode_of(fn)
    pre = load(fn, 'wsavebartr')
    post = load(fn, 'wbartr')
    pre_r, post_r = pre['rhsvort0'], post['rhsvort0']    # (nx, ny, 3)
    w, others = _identify_ab3(pre_r, post_r)

    common = dict(taux=_g64(pre, 'taux'), tauy=_g64(pre, 'tauy'),
                  advu0=_g64(pre, 'advu0'), advv0=_g64(pre, 'advv0'),
                  dfsu0=_g64(pre, 'dfsu0'), dfsv0=_g64(pre, 'dfsv0'),
                  grid=GRID, polar_filter=_SETUP[mode]['pfilter'],
                  poisson=POISSON, dt=float(pre['dt']), mt0=1)
    o1, o2 = others                            # slot-rotation rule
    hist = [pre_r[..., o1].T.astype(np.float64)[: GRID.ny - 1],
            pre_r[..., o2].T.astype(np.float64)[: GRID.ny - 1]]
    bhist = [float(pre['rhsu0bar'][o1]), float(pre['rhsu0bar'][o2])]
    out = bartr(_g64(pre, 'vort0'), float(pre['u0bar']),
                _g64(pre, 'v0'), hist, bhist, **common)
    # my new rhs must match the slice the Fortran wrote
    assert_field(out['rhs_hist'][0][: GRID.ny - 1],
                 post_r[..., w].T.astype(np.float64)[: GRID.ny - 1],
                 'rhsvort0[new]', mode)
    # f32 tolerance note: psi0 is O(1e8) in float32, and u0/v0 differentiate
    # it - cancellation amplifies the f32 storage/solver roundoff to a few
    # 1e-5 relative. Logic errors show up orders of magnitude above this.
    for key in ['vort0', 'psi0', 'u0', 'v0']:
        assert_field(out[key], _g64(post, key), key, mode, atol_scale=5e-5)
    tol = (R8_RTOL if mode == 'r8' else 1e-5) * abs(float(post['u0bar']))
    assert abs(out['u0bar'] - float(post['u0bar'])) <= max(tol, 1e-8)


@pytest.mark.parametrize('fn', FILES, ids=_IDS)
def test_gradphis_golden(fn):
    mode = mode_of(fn)
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
        assert_field(out[key], _g64(post, key), key, mode)
    assert_field(out['ps'], _g64(post, 'ps'), 'ps', mode, atol_scale=3e-5)
