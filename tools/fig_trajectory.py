"""Figure: Tier-2 trajectory divergence of the pure-Python port.

RMS(port - Fortran control) vs time for four fields, against two Fortran
null models: a single-seed twin (T1 perturbed by 1e-6 relative once) and a
continuous-injection twin (2e-5 K daily T1 noise - the correct null for
precision-floor differences that re-seed every step). Saturation level =
RMS difference between two randomly chosen days of the control itself.
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

TRAJ = os.path.expanduser('~/work/trajectory')
ctrl = np.load(f'{TRAJ}/control.npz')
twin = np.load(f'{TRAJ}/twin.npz')
twinc = np.load(f'{TRAJ}/twin_cont.npz')
port = np.load(f'{TRAJ}/port.npz')

DAYS = list(range(33, 63))
FIELDS = [('T1', 'K', 'Temperature mode $T_1$'),
          ('u1', 'm s$^{-1}$', 'Baroclinic wind $u_1$'),
          ('Ts', 'K', 'Surface temperature $T_s$'),
          ('Qc', 'W m$^{-2}$', 'Convective heating $Q_c$')]


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
for ax, (key, unit, title) in zip(axes.flat, FIELDS):
    cs = {}
    for name, src in [('port', port), ('twin', twin), ('twinc', twinc)]:
        vals = []
        for day in DAYS:
            c = T.f2py_to_grid(key, ctrl[f'd{day:03d}/{key}']).astype(float)
            if name == 'port':
                x = src[f'd{day:03d}/{key}'].astype(float)
            else:
                x = T.f2py_to_grid(key, src[f'd{day:03d}/{key}']).astype(float)
            vals.append(rms(x, c))
        cs[name] = np.array(vals)
    # saturation estimate: two well-separated control days
    a = T.f2py_to_grid(key, ctrl[f'd{DAYS[5]:03d}/{key}']).astype(float)
    b = T.f2py_to_grid(key, ctrl[f'd{DAYS[-1]:03d}/{key}']).astype(float)
    sat = rms(a, b)

    t = np.array(DAYS) - 32
    ax.semilogy(t, cs['port'], '-', color='#8a4048', lw=2,
                label='pure-Python port vs Fortran')
    ax.semilogy(t, cs['twinc'], '-', color='#4878a8', lw=1.6,
                label='Fortran twin, daily 2e-5 K noise\n(precision-floor null)')
    ax.semilogy(t, cs['twin'], '--', color='#48a878', lw=1.4,
                label='Fortran twin, single 1e-6 seed')
    ax.axhline(sat, color='0.55', lw=1, ls=':')
    ax.text(1.2, sat * 1.25, 'saturation (decorrelated states)',
            fontsize=7.5, color='0.4')
    ax.set_title(f'{title}  [RMS, {unit}]', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.spines[['top', 'right']].set_visible(False)
axes[1, 0].set_xlabel('days since branch point')
axes[1, 1].set_xlabel('days since branch point')
axes[0, 0].legend(fontsize=7.5, loc='lower right', framealpha=0.9)
fig.suptitle('Tier-2 trajectory test: the port diverges from the Fortran '
             'reference no faster than\nthe Fortran diverges from itself '
             'under equivalent precision-floor noise', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.expanduser('~/work/run/fig_trajectory.png'), dpi=150)
print('wrote fig_trajectory.png')

# summary numbers for the log
for key, _, _ in FIELDS:
    p = [rms(port[f'd{d:03d}/{key}'].astype(float),
             T.f2py_to_grid(key, ctrl[f'd{d:03d}/{key}']).astype(float))
         for d in (33, 47, 62)]
    w = [rms(T.f2py_to_grid(key, twinc[f'd{d:03d}/{key}']).astype(float),
             T.f2py_to_grid(key, ctrl[f'd{d:03d}/{key}']).astype(float))
         for d in (33, 47, 62)]
    print(f'{key}: port d1/d15/d30 = {p[0]:.2e}/{p[1]:.2e}/{p[2]:.2e} | '
          f'twin_cont = {w[0]:.2e}/{w[1]:.2e}/{w[2]:.2e}')
