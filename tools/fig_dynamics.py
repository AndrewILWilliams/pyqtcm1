"""Figure: the ported dynamics updates (barcl + bartr + gradphis) and the
full-step agreement ledger over all 13 atm_step routines."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import test_golden_dynamics as T
from qtcm1.dynamics.barotropic import bartr, gradphis
from qtcm1.dynamics.baroclinic import barcl

g = T.GRID
OUT = os.path.expanduser('~/work/run')
fn = os.path.expanduser('~/work/fixtures/step_day0105.npz')


def g64(stage, key):
    return T.f2py_to_grid(key, T.load(fn, stage)[key]).astype(np.float64)


# ---- run the ported dynamics from fixture inputs --------------------------
pre_b = T.load(fn, 'wdffus')
out_barcl = barcl(g64('wdffus', 'u1'), g64('wdffus', 'v1'),
                  g64('wdffus', 'T1'), g64('wdffus', 'q1'),
                  taux=g64('wdffus', 'taux'), tauy=g64('wdffus', 'tauy'),
                  advu1=g64('wdffus', 'advu1'), advv1=g64('wdffus', 'advv1'),
                  advT1=g64('wdffus', 'advT1'), advq1=g64('wdffus', 'advq1'),
                  dfsu1=g64('wdffus', 'dfsu1'), dfsv1=g64('wdffus', 'dfsv1'),
                  dfsT1=g64('wdffus', 'dfsT1'), dfsq1=g64('wdffus', 'dfsq1'),
                  Qc=g64('wdffus', 'Qc'), FSW=g64('wdffus', 'FSW'),
                  FLW=g64('wdffus', 'FLW'), FTs=g64('wdffus', 'FTs'),
                  Evap=g64('wdffus', 'Evap'), grid=g, polar_filter=T.PFILT,
                  dt=float(pre_b['dt']))

pre = T.load(fn, 'wsavebartr')
pre_r, post_r = pre['rhsvort0'], T.load(fn, 'wbartr')['rhsvort0']
w = int(np.argmax([np.abs(post_r[..., s] - pre_r[..., s]).max()
                   for s in range(3)]))
o1, o2 = [s for s in range(3) if s != w]
hist = [pre_r[..., o1].T.astype(np.float64)[:g.ny - 1],
        pre_r[..., o2].T.astype(np.float64)[:g.ny - 1]]
bh = [float(pre['rhsu0bar'][o1]), float(pre['rhsu0bar'][o2])]
out_bartr = bartr(g64('wsavebartr', 'vort0'), float(pre['u0bar']),
                  g64('wsavebartr', 'v0'), hist, bh,
                  taux=g64('wsavebartr', 'taux'),
                  tauy=g64('wsavebartr', 'tauy'),
                  advu0=g64('wsavebartr', 'advu0'),
                  advv0=g64('wsavebartr', 'advv0'),
                  dfsu0=g64('wsavebartr', 'dfsu0'),
                  dfsv0=g64('wsavebartr', 'dfsv0'),
                  grid=g, polar_filter=T.PFILT, poisson=T.POISSON,
                  dt=float(pre['dt']), mt0=1)
out_gp = gradphis(g64('wbartr', 'u0'), g64('wbartr', 'v0'),
                  g64('wsavebartr', 'u0'), g64('wsavebartr', 'v0')[1:],
                  g64('wbartr', 'T1'),
                  taux=g64('wbartr', 'taux'), tauy=g64('wbartr', 'tauy'),
                  advu0=g64('wbartr', 'advu0'), advv0=g64('wbartr', 'advv0'),
                  dfsu0=g64('wbartr', 'dfsu0'), dfsv0=g64('wbartr', 'dfsv0'),
                  grid=g, dt=float(pre['dt']), mt0=1)

# ---- 4-panel figure -------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
X, Y = np.meshgrid(g.lont, g.latt)
Xv, Yv = np.meshgrid(g.lont, g.latv)


def basemap(ax):
    m = Basemap(projection='cyl', llcrnrlat=-78.75, urcrnrlat=78.75,
                llcrnrlon=0, urcrnrlon=360, resolution='c', ax=ax)
    m.drawcoastlines(linewidth=0.4, color='0.35')


ax = axes[0, 0]
basemap(ax)
psi_full = out_bartr['psi0']
psi = (psi_full - psi_full.mean(axis=1, keepdims=True)) / 1e6  # zonal anom.
d = np.abs(psi_full - g64('wbartr', 'psi0')).max()
pc = ax.pcolormesh(Xv, Yv, psi, cmap='RdBu_r',
                   vmin=-np.abs(psi).max(), vmax=np.abs(psi).max(),
                   shading='nearest')
plt.colorbar(pc, ax=ax, shrink=0.8, label=r'$10^6$ m$^2$ s$^{-1}$')
ax.set_title(f'Barotropic streamfunction $\\psi_0$ (zonal anomaly; '
             f'FATD + AB3)\n'
             f'max |$\\Delta$| vs Fortran = {d:.1f} m$^2$/s '
             f'({d/np.abs(psi_full).max():.1e} rel)', fontsize=10)

ax = axes[0, 1]
basemap(ax)
u0 = out_bartr['u0']
d = np.abs(u0 - g64('wbartr', 'u0')).max()
pc = ax.pcolormesh(X, Y, u0, cmap='RdBu_r', vmin=-30, vmax=30,
                   shading='nearest')
plt.colorbar(pc, ax=ax, shrink=0.8, label='m s$^{-1}$')
ax.set_title(f'Barotropic zonal wind $u_0 = -\\partial_y\\psi_0$\n'
             f'max |$\\Delta$| = {d:.1e} m/s', fontsize=10)

ax = axes[1, 0]
basemap(ax)
dT = (out_barcl['T1'] - g64('wdffus', 'T1')) * 86400.0 / float(pre_b['dt'])
d = np.abs(out_barcl['T1'] - g64('wbarcl', 'T1')).max()
vmax = np.percentile(np.abs(dT), 99)
pc = ax.pcolormesh(X, Y, dT, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   shading='nearest')
plt.colorbar(pc, ax=ax, shrink=0.8, label='K day$^{-1}$')
ax.set_title(f'Baroclinic temperature tendency $\\partial_t T_1$ (barcl)\n'
             f'max |$\\Delta T_1$| after step = {d:.1e} K', fontsize=10)

ax = axes[1, 1]
basemap(ax)
ps = (out_gp['ps'] - 101325.0) / 100.0
d = np.abs(out_gp['ps'] - g64('wgradphis', 'ps')).max()
pc = ax.pcolormesh(X, Y, ps, cmap='RdBu_r', vmin=-np.abs(ps).max(),
                   vmax=np.abs(ps).max(), shading='nearest')
plt.colorbar(pc, ax=ax, shrink=0.8, label='hPa')
ax.set_title(f'Surface pressure anomaly (gradphis + dphiint)\n'
             f'max |$\\Delta$| = {d:.1e} Pa', fontsize=10)

fig.suptitle('pyqtcm1 ported dynamics, one time step (day-105 fixture) - '
             'all fields computed by the Python port', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f'{OUT}/fig_dynamics.png', dpi=150)
print('wrote fig_dynamics.png')
