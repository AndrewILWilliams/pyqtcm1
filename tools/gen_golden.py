#!/usr/bin/env python3
"""Generate per-routine golden fixtures from the instrumented Fortran oracle.

Strategy ("piggyback on the live sequence"): warm the compiled QTCM1 up for
N days with `compiled_form='parts'`, then execute the next day's atmospheric
time step **one wrapped subroutine at a time**, capturing the full visible
Fortran state after each call. Each routine's fixture is (state before it =
state after its predecessor, state after it); the Python port of routine X
must map before -> after for the fields X writes.

Requires the *extended* oracle build (setbypy patched with tendency-array
getters; see tools/oracle/README.md). Fixture files are ~4 MB/day compressed
and live outside the repo; point tests at them with QTCM1_FIXTURES.

Usage:
    python gen_golden.py --oracle ~/work/pyqtcm_ext --bnddir <...>/r64x42 \
                         --out ~/work/fixtures [--days 32 104 196 287]
"""

import argparse
import os
import sys
import tempfile

import numpy as np

#: All gettable 2-D/3-D real arrays (original + extended setbypy).
ARRAY_KEYS = [
    # prognostic + related (include ghost rows as stored in Fortran)
    'u1', 'v1', 'T1', 'q1', 'u0', 'v0', 'vort0', 'psi0', 'rhsvort0',
    'rhsu0bar', 'Ts', 'WD',
    # fluxes / physics outputs
    'Qc', 'FLWds', 'FLWus', 'FSWds', 'FSWus', 'Evap', 'FTs', 'taux', 'tauy',
    'FLWut', 'FLW', 'S0', 'FSWut', 'FSW',
    # extended getters: tendencies + diagnostics
    'advu1', 'advv1', 'advT1', 'advq1', 'advwu1', 'advwv1',
    'dfsu1', 'dfsv1', 'dfsT1', 'dfsq1', 'div1', 'chi1', 'GMq1', 'GMs1',
    'advu0', 'advv0', 'advwu0', 'advwv0', 'dfsu0', 'dfsv0', 'div0', 'chi0',
    'us', 'vs', 'cl1', 'CV', 'CDN', 'ALBDs', 'Ts0', 'ps',
    'dphisdx', 'dphisdy', 'Evapi', 'Runf', 'Runs', 'wet',
    # static
    'STYPE',
]

SCALAR_REAL_KEYS = ['dt', 'eps_c', 'u0bar', 'V1b',
                    'viscxu0', 'viscyu0', 'viscxu1', 'viscyu1',
                    'visc4x', 'visc4y', 'viscxT', 'viscyT',
                    'viscxq', 'viscyq', 'weml', 'ziml', 'VVsmin']

#: The atm_step sequence (wrapcall names), in execution order.
ATM_STEP = ['wmconvct', 'wcloud', 'wradsw', 'wradlw', 'wsflux', 'wsland1',
            'wadvctuv', 'wadvcttq', 'wdffus', 'wbarcl',
            'wsavebartr', 'wbartr', 'wgradphis']


def make_capture(ext):
    sp = ext.setbypy

    def get_array(key):
        sp.getitem_real_array(key)
        if not sp.is_readable:
            raise KeyError(key)
        for attr in ('real_rank1_array', 'real_rank2_array',
                     'real_rank3_array'):
            a = getattr(sp, attr)
            if a is not None:
                out = np.array(a, copy=True)
                setattr(sp, attr, None)
                return out
        raise RuntimeError(f'no rank array allocated for {key}')

    def capture():
        state = {k: get_array(k) for k in ARRAY_KEYS}
        for k in SCALAR_REAL_KEYS:
            state[k] = np.float64(sp.getitem_real(k))
        return state

    return capture


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--oracle', required=True,
                   help='dir containing the extended qtcm package')
    p.add_argument('--bnddir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--days', type=int, nargs='+',
                   default=[32, 104, 196, 287])
    args = p.parse_args()

    sys.path.insert(0, os.path.expanduser(args.oracle))
    from qtcm import Qtcm

    os.makedirs(args.out, exist_ok=True)
    for warmup in args.days:
        with tempfile.TemporaryDirectory() as tmp:
            m = Qtcm(compiled_form='parts', dt=1200., landon=1,
                     SSTmode='seasonal', bnddir=args.bnddir,
                     SSTdir=os.path.join(args.bnddir, 'SST_Reynolds'),
                     outdir=tmp, runname='golden', year0=1, month0=1,
                     day0=1, lastday=warmup, ntout=0, ntouti=0, mrestart=0)
            m.run_session()
            capture = make_capture(m._Qtcm__qtcm)

            day = warmup + 1
            m.run_list([{'__qtcm.wrapcall.wtimemanager': [day]},
                        {'__qtcm.wrapcall.wocean': [1, day]},
                        '__qtcm.wrapcall.wgetbnd'])
            m.set_qtcm_item('nastep', int(round(86400.0 / m.dt.value)))

            payload = {}
            for key, val in capture().items():
                payload[f'pre/{key}'] = val
            for wname in ATM_STEP:
                m.run_list([f'__qtcm.wrapcall.{wname}'])
                for key, val in capture().items():
                    payload[f'{wname}/{key}'] = val

            fn = os.path.join(args.out, f'step_day{day:04d}.npz')
            np.savez_compressed(fn, **payload)
            print(f'wrote {fn} ({len(payload)} arrays)')
            del m


if __name__ == '__main__':
    main()
