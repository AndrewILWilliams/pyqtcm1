"""Figure: Tier-2 trajectory race against the double-precision oracle.

RMS(x - r8 control) vs time for the pure-Python port, compared against the
sharpest possible null: the same r8 Fortran restarted with a one-part-in-
1e15 T1 perturbation (a last-bit difference). The r8 build is bit-
deterministic, so any distance above accumulated roundoff is equation-level
error; the port stays at the twin's noise floor for the full 30 days.
The grey curve shows the port run with the f32 build's init constants
(lookup tables, polar-filter extent/factors) - the configuration mismatch
found by this test - whose day-1 offset is 9 orders of magnitude larger.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import test_golden_dynamics as T

TRAJ = os.path.expanduser('~/work/trajectory_r8')
ctrl = np.load(f'{TRAJ}/control.npz')
twin = np.load(f'{TRAJ}/twin.npz')
port = np.load(f'{TRAJ}/port.npz')
pre = np.load(f'{TRAJ}/port_f32consts.npz')

DAYS = list(range(33, 63))
FIELDS = [('T1', 'K', 'Temperature mode $T_1$'),
          ('u1', 'm s$^{-1}$', 'Baroclinic wind $u_1$'),
          ('Ts', 'K', 'Surface temperature $T_s$'),
          ('Qc', 'W m$^{-2}$', 'Convective heating $Q_c$')]


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def curve(src, key, fortran_layout):
    vals = []
    for day in DAYS:
        c = T.f2py_to_grid(key, ctrl[f'd{day:03d}/{key}']).astype(float)
        x = src[f'd{day:03d}/{key}']
        x = (T.f2py_to_grid(key, x) if fortran_layout else x).astype(float)
        vals.append(rms(x, c))
    return np.array(vals)


fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
t = np.array(DAYS) - 32
for ax, (key, unit, title) in zip(axes.flat, FIELDS):
    cp = curve(port, key, False)
    cw = curve(twin, key, True)
    cf = curve(pre, key, False)
    a = T.f2py_to_grid(key, ctrl[f'd{DAYS[5]:03d}/{key}']).astype(float)
    b = T.f2py_to_grid(key, ctrl[f'd{DAYS[-1]:03d}/{key}']).astype(float)
    sat = rms(a, b)

    ax.semilogy(t, cp, '-', color='#8a4048', lw=2.2,
                label='pure-Python port (f64 constants)')
    ax.semilogy(t, cw, '-', color='#4878a8', lw=1.6,
                label='r8 Fortran twin, $10^{-15}$ rel. $T_1$ seed\n'
                      '(last-bit-perturbation null)')
    ax.semilogy(t, cf, '--', color='0.55', lw=1.4,
                label='port with f32-build init constants\n(tables + '
                      'filter row extent mismatched)')
    ax.axhline(sat, color='0.55', lw=1, ls=':')
    ax.text(1.2, sat * 1.3, 'saturation (decorrelated states)',
            fontsize=7.5, color='0.4')
    ax.set_title(f'{title}  [RMS vs r8 control, {unit}]', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.spines[['top', 'right']].set_visible(False)
axes[1, 0].set_xlabel('days since branch point')
axes[1, 1].set_xlabel('days since branch point')
axes[0, 0].legend(fontsize=7.5, loc='center right', framealpha=0.9)
fig.suptitle('Tier-2 exactness race: over 30 days (2160 time steps) the '
             'pure-Python port stays at the roundoff\nnoise floor of a '
             'bit-deterministic double-precision Fortran build - '
             'indistinguishable from a last-bit perturbation',
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fn = os.path.expanduser('~/work/run/fig_trajectory_r8.png')
fig.savefig(fn, dpi=150)
print(f'wrote {fn}')

for key, _, _ in FIELDS:
    cp = curve(port, key, False)
    cw = curve(twin, key, True)
    cf = curve(pre, key, False)
    for name, c in [('port', cp), ('twin', cw), ('f32c', cf)]:
        print(f'{key:>3s} {name}: d1={c[0]:.2e} d5={c[4]:.2e} '
              f'd15={c[14]:.2e} d30={c[-1]:.2e}')
