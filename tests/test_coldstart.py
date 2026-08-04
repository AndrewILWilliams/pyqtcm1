"""Cold-start race: from-nothing init vs a fresh r8 oracle cold start.

Validates the full init sequence in one shot: varinit state, bndinit CDN
(+ ABL first-call mutation), the getbnd first-call SST skip (day 1 runs
with the 295-K varinit ocean), the gradphis first-call return, and the
netCDF boundary machinery. Measured day-1 RMS is ~3e-14 (T1); thresholds
sit ~100x above. Skips unless the registry and the cold-start oracle
trajectory exist (tools/gen_trajectory.py --warmup 0).
"""

import os

from qtcm1.config import PACKAGED_DATA
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

_DATA = os.path.expanduser(os.environ.get(
    'QTCM1_BNDDATA', PACKAGED_DATA))
_CTRL = os.path.expanduser('~/work/trajectory_r8_cold/control.npz')

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(_DATA) and os.path.exists(_CTRL)),
    reason='registry / r8 cold-start trajectory not found')

_TH = {'T1': (5e-12, 5e-11), 'u1': (2e-11, 2e-10), 'Ts': (5e-12, 5e-11),
       'WD': (5e-12, 5e-11), 'us': (2e-11, 2e-10)}


def test_three_day_cold_start_race():
    from qtcm1.driver import ControlRun
    import test_golden_dynamics as T

    ctrl = np.load(_CTRL)
    run = ControlRun(_DATA)
    failures = []
    for day in (1, 2, 3):
        run.advance_day()
        for key, th in _TH.items():
            c = T.f2py_to_grid(key, ctrl[f'd{day:03d}/{key}']).astype(float)
            rms = float(np.sqrt(np.mean((getattr(run.state, key) - c) ** 2)))
            lim = th[0] if day == 1 else th[1]
            if rms > lim:
                failures.append(f'{key} day{day} RMS {rms:.2e} > {lim:.0e}')
    assert not failures, '; '.join(failures)
