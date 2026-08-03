#!/usr/bin/env python3
"""Per-routine max-relative-difference audit against a golden fixture.

Prints, for every routine of one atmospheric time step, the maximum
absolute difference of the port's outputs from the oracle's, normalized by
the field's max magnitude. Used to localize discrepancies; the pass/fail
version of this comparison lives in tests/test_golden_dynamics.py.

Usage:
    python audit_vs_fixture.py [fixture.npz ...]
    (defaults to the r8 day-33 fixture; mode is inferred from the path
     exactly as in the tests: f32-build constants for QTCM1_FIXTURES,
     float64 for QTCM1_FIXTURES_R8)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
import test_golden_dynamics as T                              # noqa: E402
from qtcm1.dynamics.advection import advctTq, advctuv         # noqa: E402
from qtcm1.dynamics.baroclinic import barcl                   # noqa: E402
from qtcm1.dynamics.barotropic import bartr, gradphis         # noqa: E402
from qtcm1.dynamics.diffusion import dffus                    # noqa: E402
from qtcm1.physics.clouds import cloud                        # noqa: E402
from qtcm1.physics.convection import mconvct                  # noqa: E402
from qtcm1.physics.land import sland1                         # noqa: E402
from qtcm1.physics.radiation import radlw, radsw              # noqa: E402
from qtcm1.physics.sfcflux import sfcwind_abl, sflux          # noqa: E402


def relmax(a, e):
    e = np.asarray(e, dtype=np.float64)
    return np.abs(np.asarray(a) - e).max() / max(np.abs(e).max(), 1e-300)


def audit(fn, loader=None, verbose=True, mode=None, dayofyear=None):
    mode = mode or T.mode_of(fn)
    S = T._SETUP[mode]
    g = T._g64
    L = loader or (lambda stage: T.load(fn, stage))
    if dayofyear is None:
        dayofyear = int(os.path.basename(fn)[8:12])
    rows = []

    pre, post = L('pre'), L('wmconvct')
    out = mconvct(g(pre, 'T1'), g(pre, 'q1'), float(pre['eps_c']),
                  S['pfilter'], tables=S['tables'])
    rows.append(('mconvct Qc', relmax(out['Qc'], g(post, 'Qc'))))

    pre, post = post, L('wcloud')
    cl = cloud(g(pre, 'Qc'))
    rows.append(('cloud cl1', relmax(cl['cl1'], g(post, 'cl1'))))

    pre, post = post, L('wradsw')
    sw = radsw(cl['cld'], g(pre, 'ALBDs'), dayofyear, T.GRID)
    rows.append(('radsw', max(relmax(sw[k], g(post, k))
                              for k in ['S0', 'FSWds', 'FSWus', 'FSWut',
                                        'FSW'])))

    pre, post = post, L('wradlw')
    lw = radlw(g(pre, 'T1'), g(pre, 'q1'), g(pre, 'Ts'), cl['cld'])
    rows.append(('radlw', max(relmax(lw[k], g(post, k))
                              for k in ['FLWds', 'FLWus', 'FLWut', 'FLW'])))

    pre, post = post, L('wsflux')
    wind = sfcwind_abl(g(pre, 'u1'), g(pre, 'v1'), g(pre, 'u0'),
                       g(pre, 'v0'), g(pre, 'us'), g(pre, 'vs'),
                       g(pre, 'dphisdx'), g(pre, 'dphisdy'), g(pre, 'CDN'),
                       np.asarray(T.GRID.fu, dtype=np.float64),
                       weml=float(pre['weml']), ziml=float(pre['ziml']),
                       vvsmin=float(pre['VVsmin']))
    rows.append(('sflux us/vs', max(relmax(wind[k], g(post, k))
                                    for k in ['us', 'vs'])))
    fx = sflux(g(pre, 'T1'), g(pre, 'q1'), g(pre, 'Ts'),
               T.f2py_to_grid('STYPE', pre['STYPE']), g(pre, 'CDN'), wind,
               tables=S['tables'])
    rows.append(('sflux flux', max(
        [relmax(fx[k], g(post, k)) for k in ['CV', 'taux', 'Evap', 'FTs']]
        + [relmax(fx['tauy'][:-1], g(post, 'tauy')[:-1])])))

    pre, post = post, L('wsland1')
    ld = sland1(g(pre, 'Ts'), g(pre, 'WD'),
                T.f2py_to_grid('STYPE', pre['STYPE']), g(pre, 'Qc'),
                g(pre, 'Evap'), g(pre, 'FTs'), g(pre, 'FSWds'),
                g(pre, 'FSWus'), g(pre, 'FLWds'), g(pre, 'FLWus'),
                g(pre, 'CV'), float(pre['dt']))
    landm = T.f2py_to_grid('STYPE', pre['STYPE']) != 0
    rows.append(('sland1', max(
        relmax(np.where(landm, ld[k], g(post, k)), g(post, k))
        for k in ['Ts', 'WD', 'Evap', 'Evapi', 'wet', 'Runs', 'Runf'])))

    pre, post = post, L('wadvctuv')
    adv = advctuv(g(pre, 'u1'), g(pre, 'v1'), g(pre, 'u0'), g(pre, 'v0'),
                  T.GRID)
    rows.append(('advctuv', max(
        [relmax(adv[k], g(post, k))
         for k in ['advu0', 'advu1', 'advwu0', 'advwu1', 'div1']]
        + [relmax(adv[k][1:-1], g(post, k)[1:-1])
           for k in ['advv0', 'advv1']])))

    pre, post = post, L('wadvcttq')
    tq = advctTq(g(pre, 'T1'), g(pre, 'q1'), g(pre, 'u1'), g(pre, 'v1'),
                 g(pre, 'u0'), g(pre, 'v0'), T.GRID)
    rows.append(('advctTq', max(relmax(tq[k], g(post, k))
                                for k in ['advT1', 'advq1'])))

    pre, post = post, L('wdffus')
    df = dffus(g(pre, 'u1'), g(pre, 'v1'), g(pre, 'u0'), g(pre, 'v0'),
               g(pre, 'T1'), g(pre, 'q1'), T.GRID,
               **{k: float(pre[k]) for k in
                  ['viscxu1', 'viscyu1', 'visc4x', 'visc4y', 'viscxT',
                   'viscyT', 'viscxq', 'viscyq', 'viscxu0', 'viscyu0']})
    rows.append(('dffus', max(relmax(df[k], g(post, k))
                              for k in ['dfsu1', 'dfsv1', 'dfsu0', 'dfsv0',
                                        'dfsT1', 'dfsq1'])))

    pre, post = post, L('wbarcl')
    bc = barcl(g(pre, 'u1'), g(pre, 'v1'), g(pre, 'T1'), g(pre, 'q1'),
               taux=g(pre, 'taux'), tauy=g(pre, 'tauy'),
               advu1=g(pre, 'advu1'), advv1=g(pre, 'advv1'),
               advT1=g(pre, 'advT1'), advq1=g(pre, 'advq1'),
               dfsu1=g(pre, 'dfsu1'), dfsv1=g(pre, 'dfsv1'),
               dfsT1=g(pre, 'dfsT1'), dfsq1=g(pre, 'dfsq1'),
               Qc=g(pre, 'Qc'), FSW=g(pre, 'FSW'), FLW=g(pre, 'FLW'),
               FTs=g(pre, 'FTs'), Evap=g(pre, 'Evap'), grid=T.GRID,
               polar_filter=S['pfilter'], dt=float(pre['dt']))
    rows.append(('barcl', max(relmax(bc[k], g(post, k))
                              for k in ['u1', 'v1', 'T1', 'q1', 'div1',
                                        'GMs1', 'GMq1'])))

    sav = L('wsavebartr')
    pre, post = sav, L('wbartr')
    w, _ = T._identify_ab3(pre['rhsvort0'], post['rhsvort0'])
    # Fortran bartr rotates slots write->oldest (ktemp=k_2; k_2=k_1; k_1=k;
    # k=ktemp), so r_{n-1} = slice (w+1)%3 and r_{n-2} = slice (w+2)%3.
    o1, o2 = (w + 1) % 3, (w + 2) % 3
    bt = bartr(g(pre, 'vort0'), float(pre['u0bar']), g(pre, 'v0'),
               [pre['rhsvort0'][..., o1].T.astype(np.float64)[: T.GRID.ny - 1],
                pre['rhsvort0'][..., o2].T.astype(np.float64)[: T.GRID.ny - 1]],
               [float(pre['rhsu0bar'][o1]), float(pre['rhsu0bar'][o2])],
               taux=g(pre, 'taux'), tauy=g(pre, 'tauy'),
               advu0=g(pre, 'advu0'), advv0=g(pre, 'advv0'),
               dfsu0=g(pre, 'dfsu0'), dfsv0=g(pre, 'dfsv0'), grid=T.GRID,
               polar_filter=S['pfilter'], poisson=T.POISSON,
               dt=float(pre['dt']), mt0=1)
    rows.append(('bartr', max(relmax(bt[k], g(post, k))
                              for k in ['vort0', 'psi0', 'u0', 'v0'])))

    pre2, post2 = post, L('wgradphis')
    gp = gradphis(g(pre2, 'u0'), g(pre2, 'v0'), g(sav, 'u0'),
                  g(sav, 'v0')[1:], g(pre2, 'T1'),
                  taux=g(pre2, 'taux'), tauy=g(pre2, 'tauy'),
                  advu0=g(pre2, 'advu0'), advv0=g(pre2, 'advv0'),
                  dfsu0=g(pre2, 'dfsu0'), dfsv0=g(pre2, 'dfsv0'),
                  grid=T.GRID, dt=float(pre2['dt']), mt0=1)
    rows.append(('gradphis', max(relmax(gp[k], g(post2, k))
                                 for k in ['dphisdx', 'dphisdy', 'ps'])))

    if verbose:
        print(f'\n{os.path.basename(os.path.dirname(fn))}/'
              f'{os.path.basename(fn)}  [mode={mode}]')
        for name, val in rows:
            print(f'  {name:<12s} max rel diff = {val:.2e}')
    return rows


if __name__ == '__main__':
    files = sys.argv[1:] or [os.path.expanduser(
        '~/work/fixtures_r8/step_day0033.npz')]
    for fn in files:
        audit(fn)
