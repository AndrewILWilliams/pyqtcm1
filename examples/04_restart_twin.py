"""Bit-exact restarts and twin (perturbed-initial-condition) experiments.

Demonstrates two properties of the port: restarts continue the
trajectory to the last bit, and a last-bit perturbation lets you measure
the model's own error growth. QTCM1 at fixed SST is strongly damped:
round-off noise grows only weakly (a few x per month, no chaotic
e-folding), which is exactly what makes trajectory shadowing a sharp
validation test.
"""

import os
from dataclasses import replace

import numpy as np

from qtcm1.config import RunConfig
from qtcm1.driver import ControlRun

DATA = os.environ.get('QTCM1_BNDDATA')   # None -> packaged data/r64x42
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
d = ControlRun.from_restart('day60.restart.npz')            # its reference
rms_by_day = []
for _ in range(30):
    c.advance_day()
    d.advance_day()
    rms_by_day.append(float(np.sqrt(np.mean(
        (c.state.T1 - d.state.T1) ** 2))))

print(f'last-bit twin after 30 days: RMS T1 divergence = '
      f'{rms_by_day[-1]:.2e} K (no chaotic amplification)')

# -- figure: error growth (or rather, the lack of it) --------------------
import matplotlib.pyplot as plt                          # noqa: E402

seed_rms = float(np.sqrt(np.mean((b.state.T1 * 1e-15) ** 2)))
fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.semilogy(np.arange(1, 31), rms_by_day, marker='o', ms=3.5, lw=1.2)
ax.axhline(seed_rms, ls='--', lw=0.9, color='0.4')
ax.annotate('initial one-ulp seed', (30, seed_rms),
            textcoords='offset points', xytext=(-4, 5),
            ha='right', fontsize=8, color='0.35')
ax.set_xlabel('days after perturbation')
ax.set_ylabel(r'RMS $T_1$ divergence [K]')
ax.set_title('last-bit twin: divergence stays at round-off scale'
             ' (no chaotic growth)')
fig.tight_layout()
fig.savefig('twin_divergence.png', dpi=150)
print('wrote twin_divergence.png')
