"""Tier 2: short trajectory race against the double-precision oracle.

The r8 Fortran build is bit-deterministic and shares every init constant
with the port, so the port's RMS distance from it measures pure
equation-level fidelity. After one coupling day (72 full time steps) the
measured distance is ~1e-13 RMS on all prognostics (accumulated float64
roundoff); a port defect shows up 6+ orders above the thresholds here
(e.g. the pre-fix f32-table mismatch gave day-1 T1 RMS of 6e-3 K).

Thresholds carry ~100x margin over measured values and account for the
~x2.6/day chaotic amplification of the roundoff seed.

Skips unless the r8 fixture and trajectory files exist (see
tools/gen_golden.py, tools/gen_trajectory.py).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

_FIX = os.path.expanduser('~/work/fixtures_r8/step_day0033.npz')
_CTRL = os.path.expanduser('~/work/trajectory_r8/control.npz')

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_FIX) and os.path.exists(_CTRL)),
    reason='r8 fixture/trajectory not found')

#: (day-1 threshold, day-3 threshold); Qc is wider by eps_c*Cpg amplification
_THRESH = {'u1': (5e-12, 1e-9), 'v1': (5e-12, 1e-9), 'T1': (5e-12, 1e-9),
           'q1': (5e-12, 1e-9), 'u0': (5e-12, 1e-9), 'v0': (5e-12, 1e-9),
           'Ts': (5e-12, 1e-9), 'WD': (5e-12, 1e-9), 'Qc': (1e-9, 1e-7)}


def test_three_day_race_vs_r8_oracle():
    import run_trajectory_port as R

    port = R.run('r8', 3, save=False)
    _, curves = R.rms_curves('r8', port, 3)
    failures = []
    for key, (t1, t3) in _THRESH.items():
        cp, _ = curves[key]
        if cp[0] > t1:
            failures.append(f'{key} day1 RMS {cp[0]:.2e} > {t1:.0e}')
        if cp[2] > t3:
            failures.append(f'{key} day3 RMS {cp[2]:.2e} > {t3:.0e}')
    assert not failures, '; '.join(failures)
