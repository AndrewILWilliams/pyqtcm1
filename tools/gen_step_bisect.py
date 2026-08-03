#!/usr/bin/env python3
"""Capture every routine of every step of ONE model day from the oracle.

Replicates the trajectory control's exact construction (32-day run_session
warmup + manual daily loop), discards days up to ``--day``-1, then executes
day ``--day`` as 72 manual atm_step wrapcall sequences, capturing the full
visible Fortran state after every wrapped subroutine. Every (step, stage)
is then a golden fixture; used to bisect trajectory divergence events to a
single routine application.

Self-validates: end-of-previous-day and end-of-target-day states must match
the stored trajectory control bitwise.

Output: <out>/step_bisect_day{day}.npz with keys s{step:02d}/{stage}/{key}
(stage 'pre' = state entering the step).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gen_golden import ARRAY_KEYS, SCALAR_REAL_KEYS, ATM_STEP, make_capture

CHECK_KEYS = ['u1', 'v1', 'T1', 'q1', 'u0', 'v0', 'vort0', 'psi0', 'Ts',
              'WD', 'us', 'vs', 'Qc', 'rhsvort0', 'rhsu0bar']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--oracle', default=os.path.expanduser('~/work/pyqtcm_r8'))
    p.add_argument('--bnddir', default=os.path.expanduser(
        '~/work/qtcm-master/test/bnddir/r64x42'))
    p.add_argument('--traj', default=os.path.expanduser('~/work/trajectory_r8'))
    p.add_argument('--out', default=os.path.expanduser('~/work/trajectory_r8'))
    p.add_argument('--day', type=int, default=46)
    p.add_argument('--warmup', type=int, default=32)
    args = p.parse_args()

    sys.path.insert(0, args.oracle)
    from qtcm import Qtcm
    import tempfile

    ctrl = np.load(os.path.join(args.traj, 'control.npz'))

    with tempfile.TemporaryDirectory() as tmp:
        m = Qtcm(compiled_form='parts', dt=1200., landon=1,
                 SSTmode='seasonal', bnddir=args.bnddir,
                 SSTdir=os.path.join(args.bnddir, 'SST_Reynolds'),
                 outdir=tmp, runname='bisect', year0=1, month0=1, day0=1,
                 lastday=args.warmup, ntout=0, ntouti=0, mrestart=0)
        m.run_session()
        sp = m._Qtcm__qtcm.setbypy
        capture = make_capture(m._Qtcm__qtcm)

        # manual daily loop, exactly as gen_trajectory's control branch
        for day in range(args.warmup + 1, args.day):
            m.run_list([{'__qtcm.wrapcall.wtimemanager': [day]},
                        {'__qtcm.wrapcall.wocean': [1, day]},
                        '__qtcm.wrapcall.wgetbnd'])
            m.qtcm()

        # bitwise check vs control at end of day-1
        state = capture()
        for key in CHECK_KEYS:
            ref = ctrl[f'd{args.day - 1:03d}/{key}']
            if not np.array_equal(state[key], ref):
                mx = np.abs(np.asarray(state[key], float)
                            - np.asarray(ref, float)).max()
                print(f'WARNING: day-{args.day - 1} mismatch {key}: {mx:.3e}')

        # target day: boundary update, then 72 manual step sequences
        m.run_list([{'__qtcm.wrapcall.wtimemanager': [args.day]},
                    {'__qtcm.wrapcall.wocean': [1, args.day]},
                    '__qtcm.wrapcall.wgetbnd'])
        m.set_qtcm_item('nastep', int(round(86400.0 / m.dt.value)))

        payload = {}
        for step in range(1, 73):
            for key, val in capture().items():
                payload[f's{step:02d}/pre/{key}'] = val
            for wname in ATM_STEP:
                m.run_list([f'__qtcm.wrapcall.{wname}'])
                for key, val in capture().items():
                    payload[f's{step:02d}/{wname}/{key}'] = val
            print(f'step {step} captured', flush=True)

        # bitwise check vs control at end of target day
        state = capture()
        ok = True
        for key in CHECK_KEYS:
            ref = ctrl[f'd{args.day:03d}/{key}']
            if not np.array_equal(state[key], ref):
                mx = np.abs(np.asarray(state[key], float)
                            - np.asarray(ref, float)).max()
                print(f'WARNING: day-{args.day} mismatch {key}: {mx:.3e}')
                ok = False
        print('end-of-day check:', 'BITWISE MATCH' if ok else 'MISMATCH (see above)')

        fn = os.path.join(args.out, f'step_bisect_day{args.day:04d}.npz')
        np.savez_compressed(fn, **payload)
        print(f'wrote {fn} ({len(payload)} arrays)')
        del m


if __name__ == '__main__':
    main()
