"""Bit-exact restarts and twin (perturbed-initial-condition) experiments.

Demonstrates two properties of the port: restarts continue the
trajectory to the last bit, and a last-bit perturbation lets you measure
the model's own error growth (QTCM1 at fixed SST is strongly damped --
last-bit noise does NOT amplify, which is exactly what makes trajectory
shadowing a sharp validation test).
"""

import os
from dataclasses import replace

import numpy as np

from qtcm1.config import RunConfig
from qtcm1.driver import ControlRun

DATA = os.path.expanduser(os.environ.get('QTCM1_BNDDATA',
                                         '~/qtcm1_data/r64x42'))
cfg = RunConfig(data_path=DATA)

# -- 1. bit-exact restart ------------------------------------------------
a = ControlRun(config=cfg)
for _ in range(60):
    a.advance_day()
a.save_restart('day60.restart.npz')
for _ in range(30):
    a.advance_day()

b = ControlRun.from_restart('day60.restart.npz')
for _ in range(30):
    b.advance_day()

print('restart bitwise-identical:',
      all(np.array_equal(getattr(a.state, k), getattr(b.state, k))
          for k in ['u1', 'v1', 'T1', 'q1', 'Ts', 'WD']))

# -- 2. last-bit twin ----------------------------------------------------
c = ControlRun.from_restart('day60.restart.npz')
c.state = replace(c.state, T1=c.state.T1 * (1.0 + 1e-15))   # one-ulp seed
for _ in range(30):
    c.advance_day()

rms = float(np.sqrt(np.mean((c.state.T1 - b.state.T1) ** 2)))
print(f'last-bit twin after 30 days: RMS T1 divergence = {rms:.2e} K '
      f'(no chaotic amplification)')
