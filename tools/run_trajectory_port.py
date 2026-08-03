#!/usr/bin/env python3
"""Run the pure-Python model along a Fortran oracle's trajectory (Tier 2).

Initializes ModelState from a day-33 golden fixture 'pre' state (with the
AB3 history ordering identified from the fixture's own bartr step), then
integrates N days with the oracle-captured daily forcing, saving daily
states, and prints divergence-vs-time statistics.

Two modes, matching the two oracle builds:

* ``--mode f32``: the standard single-precision Fortran. The port runs
  with float32-mode init constants; agreement is limited by the oracle's
  own f32 state storage (compare against the continuous-noise twin null).
* ``--mode r8`` (default): the double-precision build, which is
  bit-deterministic. The port shares every constant with it, so the RMS
  difference measures pure equation-level fidelity and should sit at
  accumulated float64 roundoff.

Writes <traj>/port.npz.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
import test_golden_dynamics as T
from qtcm1.model import Model, ModelState

MODES = {
    'f32': dict(traj='~/work/trajectory',
                fix='~/work/fixtures/step_day0033.npz',
                init_dtype=np.float32, save_dtype=np.float32),
    'r8': dict(traj='~/work/trajectory_r8',
               fix='~/work/fixtures_r8/step_day0033.npz',
               init_dtype=np.float64, save_dtype=np.float64),
}
KEYS_CMP = ['u1', 'v1', 'T1', 'q1', 'u0', 'v0', 'Ts', 'WD', 'Qc']

FIX = None                                     # set by state_from_fixture


def g64(stage, key, fn=None):
    fn = fn or FIX                      # resolve at call time, not def time
    return T.f2py_to_grid(key, T.load(fn, stage)[key]).astype(np.float64)


def state_from_fixture(fix, init_dtype=np.float64):
    global FIX
    FIX = os.path.expanduser(fix)
    pre = T.load(FIX, 'pre')
    post_r = T.load(FIX, 'wbartr')['rhsvort0']
    pre_r = T.load(FIX, 'wsavebartr')['rhsvort0']
    w = int(np.argmax([np.abs(post_r[..., s] - pre_r[..., s]).max()
                       for s in range(3)]))
    # Fortran bartr rotates slots write->oldest, so the previous write (our
    # r_{n-1}) is slice (w+1)%3 and r_{n-2} is (w+2)%3 (qtcm.F90 ktemp swap).
    o1, o2 = (w + 1) % 3, (w + 2) % 3
    ny = T.GRID.ny
    model = Model(stype=T.f2py_to_grid('STYPE', pre['STYPE']),
                  cdn=g64('pre', 'CDN'),
                  init_dtype=init_dtype,
                  params=dict(dt=float(pre['dt']),
                              eps_c=float(pre['eps_c']),
                              viscxu0=float(pre['viscxu0']),
                              viscyu0=float(pre['viscyu0']),
                              viscxu1=float(pre['viscxu1']),
                              viscyu1=float(pre['viscyu1']),
                              visc4x=float(pre['visc4x']),
                              visc4y=float(pre['visc4y']),
                              viscxT=float(pre['viscxT']),
                              viscyT=float(pre['viscyT']),
                              viscxq=float(pre['viscxq']),
                              viscyq=float(pre['viscyq']),
                              weml=float(pre['weml']),
                              ziml=float(pre['ziml']),
                              vvsmin=float(pre['VVsmin'])))
    state = ModelState(
        u1=g64('pre', 'u1'), v1=g64('pre', 'v1'),
        T1=g64('pre', 'T1'), q1=g64('pre', 'q1'),
        u0=g64('pre', 'u0'), v0=g64('pre', 'v0'),
        vort0=g64('pre', 'vort0'), u0bar=float(pre['u0bar']),
        rhs_hist=[pre_r[..., o1].T.astype(np.float64)[: ny - 1],
                  pre_r[..., o2].T.astype(np.float64)[: ny - 1]],
        rhsbar_hist=[float(pre['rhsu0bar'][o1]),
                     float(pre['rhsu0bar'][o2])],
        Ts=g64('pre', 'Ts'), WD=g64('pre', 'WD'),
        us=g64('pre', 'us'), vs=g64('pre', 'vs'),
        dphisdx=g64('pre', 'dphisdx'), dphisdy=g64('pre', 'dphisdy'))
    return model, state


def run(mode: str, ndays: int, save: bool = True, init_dtype=None):
    """Integrate ndays from the mode's fixture; init_dtype overrides the
    mode's init-constant precision (used to demonstrate the mismatch)."""
    cfg = MODES[mode]
    traj = os.path.expanduser(cfg['traj'])
    ctrl = np.load(os.path.join(traj, 'control.npz'))
    model, state = state_from_fixture(cfg['fix'],
                                      init_dtype or cfg['init_dtype'])
    days = list(range(33, 33 + ndays))
    out = {}
    t0 = time.time()
    for day in days:
        sst = np.asarray(ctrl[f'forcing/d{day:03d}/Ts0']).T.astype(np.float64)
        alb = np.asarray(ctrl[f'forcing/d{day:03d}/ALBDs']).T.astype(np.float64)
        state, diags = model.step_day(state, sst, alb, dayofyear=day)
        snap = dict(u1=state.u1, v1=state.v1, T1=state.T1, q1=state.q1,
                    u0=state.u0, v0=state.v0, Ts=state.Ts, WD=state.WD,
                    Qc=diags['Qc'])
        for k, v in snap.items():
            out[f'd{day:03d}/{k}'] = v.astype(cfg['save_dtype'])
    wall = time.time() - t0
    if save:
        np.savez_compressed(os.path.join(traj, 'port.npz'), **out)
    print(f'port[{mode}]: {ndays} days in {wall:.1f} s '
          f'({wall / ndays * 365:.0f} s/sim-year)')
    return out


def rms_curves(mode, port, ndays):
    """Daily RMS differences vs control (and vs twins, f32 mode only)."""
    cfg = MODES[mode]
    traj = os.path.expanduser(cfg['traj'])
    ctrl = np.load(os.path.join(traj, 'control.npz'))
    twin_fn = os.path.join(traj, 'twin.npz')
    twin = np.load(twin_fn) if os.path.exists(twin_fn) else None
    days = list(range(33, 33 + ndays))
    curves = {}
    for key in KEYS_CMP:
        cp, ct = [], []
        for day in days:
            c = T.f2py_to_grid(key, ctrl[f'd{day:03d}/{key}']).astype(float)
            p = port[f'd{day:03d}/{key}'].astype(float)
            cp.append(np.sqrt(np.mean((p - c) ** 2)))
            if twin is not None:
                w = T.f2py_to_grid(key,
                                   twin[f'd{day:03d}/{key}']).astype(float)
                ct.append(np.sqrt(np.mean((w - c) ** 2)))
        curves[key] = (np.array(cp), np.array(ct) if twin is not None
                       else None)
    return days, curves


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=list(MODES), default='r8')
    p.add_argument('ndays', type=int, nargs='?', default=30)
    args = p.parse_args()
    port = run(args.mode, args.ndays)
    days, curves = rms_curves(args.mode, port, args.ndays)
    for key in KEYS_CMP:
        cp, ct = curves[key]
        line = (f'{key:>3s}: port-vs-ctrl day1={cp[0]:.2e} '
                f'day{args.ndays}={cp[-1]:.2e}')
        if ct is not None:
            line += f' | twin-vs-ctrl day1={ct[0]:.2e} day{args.ndays}={ct[-1]:.2e}'
        print(line)
