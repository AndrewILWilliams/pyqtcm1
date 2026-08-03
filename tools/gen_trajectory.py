#!/usr/bin/env python3
"""Oracle trajectories for Tier-2 testing: control + perturbed twin.

Warms the instrumented parts-form oracle 32 days (bit-identical to the
golden-fixture warmup), then integrates N more days capturing daily states
and the daily boundary forcing (SST, albedo). The twin branch perturbs T1
by a relative 1e-6 at the branch point; its divergence from the control
defines the acceptance envelope for the Python port.

Output: ~/work/trajectory/{control,twin}.npz with per-day arrays
``d{day:03d}/<key>`` plus ``forcing/d{day:03d}/{Ts0,ALBDs}``.
"""

import argparse
import os
import sys
import tempfile

import numpy as np

KEYS = ['u1', 'v1', 'T1', 'q1', 'u0', 'v0', 'vort0', 'psi0', 'Ts', 'WD',
        'us', 'vs', 'Qc', 'rhsvort0', 'rhsu0bar', 'u0bar']


def get_array(sp, key):
    sp.getitem_real_array(key)
    assert sp.is_readable, key
    for attr in ('real_rank1_array', 'real_rank2_array', 'real_rank3_array'):
        a = getattr(sp, attr)
        if a is not None:
            out = np.array(a, copy=True)
            setattr(sp, attr, None)
            return out
    raise RuntimeError(key)


def run_branch(Qtcm, bnddir, warmup, ndays, perturb, out_fn,
               daily_noise=0.0, perturb_rel=1.0e-6):
    with tempfile.TemporaryDirectory() as tmp:
        m = Qtcm(compiled_form='parts', dt=1200., landon=1,
                 SSTmode='seasonal', bnddir=bnddir,
                 SSTdir=os.path.join(bnddir, 'SST_Reynolds'),
                 outdir=tmp, runname='traj', year0=1, month0=1, day0=1,
                 lastday=warmup, ntout=0, ntouti=0, mrestart=0)
        m.run_session()
        sp = m._Qtcm__qtcm.setbypy

        if perturb:
            T1 = get_array(sp, 'T1')
            m.set_qtcm_item('T1', T1 * (1.0 + perturb_rel))

        rng = np.random.default_rng(12345)
        payload = {}
        for day in range(warmup + 1, warmup + 1 + ndays):
            if daily_noise > 0.0:            # continuous-injection twin
                T1 = get_array(sp, 'T1')
                m.set_qtcm_item('T1', T1 + daily_noise
                                * rng.standard_normal(T1.shape)
                                .astype(np.float32))
            m.run_list([{'__qtcm.wrapcall.wtimemanager': [day]},
                        {'__qtcm.wrapcall.wocean': [1, day]},
                        '__qtcm.wrapcall.wgetbnd'])
            payload[f'forcing/d{day:03d}/Ts0'] = get_array(sp, 'Ts0')
            payload[f'forcing/d{day:03d}/ALBDs'] = get_array(sp, 'ALBDs')
            # Step the day manually with a SINGLE getbnd. (m.qtcm() calls
            # wgetbnd again; on mid-month bracket days a same-day second
            # call corrupts the interpolation - bndry1's guard restores
            # time1/time2 but var_next has already advanced, so var_prior
            # gets next month's data. The standard full-form driver calls
            # getbnd once per day; this matches it, and the captured
            # forcing above is then exactly what the day integrates.)
            nastep = int(round(86400.0 / m.dt.value))
            m.set_qtcm_item('nastep', nastep)
            for m.it.value in range(1, nastep + 1):
                m.run_list(['atm_step'])
            for key in KEYS:
                if key == 'u0bar':
                    payload[f'd{day:03d}/{key}'] = np.float64(
                        sp.getitem_real('u0bar'))
                else:
                    payload[f'd{day:03d}/{key}'] = get_array(sp, key)
        np.savez_compressed(out_fn, **payload)
        print(f'wrote {out_fn}')
        del m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--oracle', default=os.path.expanduser('~/work/pyqtcm_ext'))
    p.add_argument('--bnddir', default=os.path.expanduser(
        '~/work/qtcm-master/test/bnddir/r64x42'))
    p.add_argument('--out', default=os.path.expanduser('~/work/trajectory'))
    p.add_argument('--warmup', type=int, default=32)
    p.add_argument('--ndays', type=int, default=30)
    p.add_argument('--branches', default='control,twin,twin_cont',
                   help='comma list: control, twin, twin_cont')
    p.add_argument('--perturb-rel', type=float, default=1.0e-6,
                   help='relative T1 perturbation of the twin branch')
    args = p.parse_args()
    sys.path.insert(0, args.oracle)
    from qtcm import Qtcm
    os.makedirs(args.out, exist_ok=True)
    branches = args.branches.split(',')
    if 'control' in branches:
        run_branch(Qtcm, args.bnddir, args.warmup, args.ndays, False,
                   os.path.join(args.out, 'control.npz'))
    if 'twin' in branches:
        run_branch(Qtcm, args.bnddir, args.warmup, args.ndays, True,
                   os.path.join(args.out, 'twin.npz'),
                   perturb_rel=args.perturb_rel)
    if 'twin_cont' in branches:
        run_branch(Qtcm, args.bnddir, args.warmup, args.ndays, False,
                   os.path.join(args.out, 'twin_cont.npz'),
                   daily_noise=2.0e-5)


if __name__ == '__main__':
    main()
