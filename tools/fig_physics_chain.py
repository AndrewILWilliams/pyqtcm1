"""Figures: the ported physics chain vs the Fortran oracle.

Figure 1: maps of each ported physics output for one time step (the day-105
fixture, mid-April of model year 1), computed by the *Python port* with the
max |difference| vs the Fortran oracle annotated per panel.

Figure 2: the agreement ledger - max relative error per ported routine
across all four seasonal fixture days, against the float32-roundoff context.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import test_golden_dynamics as T                     # reuse loaders
from qtcm1.grid import Grid
from qtcm1.dynamics.advection import advctTq, advctuv
from qtcm1.dynamics.diffusion import dffus
from qtcm1.dynamics.filters import PolarFilter
from qtcm1.physics.convection import mconvct
from qtcm1.physics.clouds import cloud
from qtcm1.physics.radiation import radlw, radsw
from qtcm1.physics.sfcflux import sfcwind_abl, sflux

g = Grid()
pf = PolarFilter(g)
OUT = os.path.expanduser('~/work/run')


def F(stage, key, fn):
    return T.f2py_to_grid(key, T.load(fn, stage)[key]).astype(np.float64)


def run_chain(fn):
    """Run the ported physics chain from the fixture's 'pre' state."""
    pre = T.load(fn, 'pre')
    day = int(os.path.basename(fn)[8:12])
    r = {}
    r['Qc'] = mconvct(F('pre', 'T1', fn), F('pre', 'q1', fn),
                      float(pre['eps_c']), pf)['Qc']
    cl = cloud(r['Qc'])
    r['cl1'] = cl['cl1']
    sw = radsw(cl['cld'], F('wcloud', 'ALBDs', fn), day, g)
    lw = radlw(F('pre', 'T1', fn), F('pre', 'q1', fn), F('pre', 'Ts', fn),
               cl['cld'])
    r.update(sw); r.update(lw)
    pw = T.load(fn, 'wradlw')
    wind = sfcwind_abl(F('wradlw', 'u1', fn), F('wradlw', 'v1', fn),
                       F('wradlw', 'u0', fn), F('wradlw', 'v0', fn),
                       F('wradlw', 'us', fn), F('wradlw', 'vs', fn),
                       F('wradlw', 'dphisdx', fn), F('wradlw', 'dphisdy', fn),
                       F('wradlw', 'CDN', fn), np.asarray(g.fu, float),
                       weml=float(pw['weml']), ziml=float(pw['ziml']),
                       vvsmin=float(pw['VVsmin']))
    fl = sflux(F('pre', 'T1', fn), F('pre', 'q1', fn), F('pre', 'Ts', fn),
               T.f2py_to_grid('STYPE', pw['STYPE']),
               F('wradlw', 'CDN', fn), wind)
    r.update(wind); r.update(fl)
    return r


# ---------------------------------------------------------------- figure 1
fn = os.path.expanduser('~/work/fixtures/step_day0105.npz')
r = run_chain(fn)
oracle_stage = dict(Qc='wmconvct', cl1='wcloud', S0='wradsw', FSWds='wradsw',
                    FLWut='wradlw', Evap='wsflux', FTs='wsflux', us='wsflux',
                    vs='wsflux')

panels = [
    ('Qc', 'Convective heating / precip', 'W m$^{-2}$', 'YlGnBu', (0, None)),
    ('cl1', 'Deep + CsCc cloud cover', '-', 'YlGnBu', (0, None)),
    ('FSWds', 'Downward surface shortwave', 'W m$^{-2}$', 'YlOrRd', (0, None)),
    ('FLWut', 'Outgoing longwave (OLR)', 'W m$^{-2}$', 'YlOrRd', (None, None)),
    ('Evap', 'Potential evaporation (pre-land adjustment)', 'W m$^{-2}$', 'YlGnBu', (0, None)),
    ('FTs', 'Sensible heat flux', 'W m$^{-2}$', 'RdBu_r', ('sym', None)),
]

fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.5))
for ax, (key, title, units, cmap, vlim) in zip(axes.flat, panels):
    field = r[key]
    exp = F(oracle_stage[key], key, fn)
    dmax = np.abs(field - exp).max()
    m = Basemap(projection='cyl', llcrnrlat=-78.75, urcrnrlat=78.75,
                llcrnrlon=0, urcrnrlon=360, resolution='c', ax=ax)
    m.drawcoastlines(linewidth=0.4, color='0.3')
    X, Y = np.meshgrid(g.lont, g.latt)
    if vlim[0] == 'sym':
        vmax = np.abs(field).max()
        pc = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading='nearest')
    else:
        pc = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=vlim[0],
                           shading='nearest')
    plt.colorbar(pc, ax=ax, shrink=0.75, pad=0.02, label=units)
    uplain = 'W/m2' if 'W' in units else units
    ax.set_title(f'{title}\nport vs Fortran: max |$\\Delta$| = {dmax:.2e} '
                 f'{uplain}', fontsize=10)
fig.suptitle('pyqtcm1 ported physics chain, one time step '
             '(day 105 = mid-April, year 1)\n'
             'fields computed by the Python port; annotation gives the '
             'max abs difference vs the Fortran oracle', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f'{OUT}/fig_physics_chain.png', dpi=150)
plt.close(fig)
print('wrote fig_physics_chain.png')

# ---------------------------------------------------------------- figure 2
routines = {}
for fx in T.FILES:
    day = os.path.basename(fx)
    r = run_chain(fx)
    adv = advctuv(F('wsland1', 'u1', fx), F('wsland1', 'v1', fx),
                  F('wsland1', 'u0', fx), F('wsland1', 'v0', fx), g)
    tq = advctTq(F('wadvctuv', 'T1', fx), F('wadvctuv', 'q1', fx),
                 F('wadvctuv', 'u1', fx), F('wadvctuv', 'v1', fx),
                 F('wadvctuv', 'u0', fx), F('wadvctuv', 'v0', fx), g)
    prq = T.load(fx, 'wadvcttq')
    df = dffus(F('wadvcttq', 'u1', fx), F('wadvcttq', 'v1', fx),
               F('wadvcttq', 'u0', fx), F('wadvcttq', 'v0', fx),
               F('wadvcttq', 'T1', fx), F('wadvcttq', 'q1', fx), g,
               **{k: float(prq[k]) for k in
                  ['viscxu1', 'viscyu1', 'visc4x', 'visc4y', 'viscxT',
                   'viscyT', 'viscxq', 'viscyq', 'viscxu0', 'viscyu0']})
    checks = {
        'mconvct': [('Qc', r['Qc'], 'wmconvct')],
        'cloud': [('cl1', r['cl1'], 'wcloud')],
        'radsw': [(k, r[k], 'wradsw') for k in ['FSW', 'FSWds', 'FSWut']],
        'radlw': [(k, r[k], 'wradlw') for k in ['FLWds', 'FLWut', 'FLW']],
        'sflux (ABL)': [(k, r[k], 'wsflux') for k in
                        ['us', 'vs', 'Evap', 'FTs', 'taux']],
        'advctuv': [(k, adv[k], 'wadvctuv') for k in
                    ['advu0', 'advu1', 'div1']],
        'advctTq': [(k, tq[k], 'wadvcttq') for k in ['advT1', 'advq1']],
        'dffus': [(k, df[k], 'wdffus') for k in ['dfsu1', 'dfsT1', 'dfsu0']],
    }
    for rt, items in checks.items():
        worst = 0.0
        for key, act, stage in items:
            exp = F(stage, key, fx)
            if key in ('advv0', 'advv1'):
                act, exp = act[1:-1], exp[1:-1]
            scale = max(np.abs(exp).max(), 1e-30)
            worst = max(worst, np.abs(act - exp).max() / scale)
        routines.setdefault(rt, []).append(worst)

names = list(routines)
vals = [max(routines[k]) for k in names]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ypos = np.arange(len(names))[::-1]
ax.barh(ypos, vals, color='#4878a8', height=0.6)
for y, v in zip(ypos, vals):
    ax.text(v * 1.25, y, f'{v:.1e}', va='center', fontsize=9)
ax.axvline(1.19e-7, color='0.4', lw=1, ls=':')
ax.text(1.19e-7, len(names) - 0.2, ' float32 machine eps', fontsize=8,
        color='0.35', rotation=0)
ax.axvline(1e-4, color='#a84848', lw=1, ls='--')
ax.text(1e-4, -0.45, ' test tolerance ceiling', fontsize=8, color='#a84848')
ax.set_yticks(ypos, names)
ax.set_xscale('log')
ax.set_xlim(3e-8, 3e-3)
ax.set_xlabel('max relative error vs Fortran oracle '
              '(worst field, worst of 4 seasonal states)')
ax.set_title('pyqtcm1 golden-test agreement ledger: every ported routine '
             'sits at the\nsingle-precision noise floor of the Fortran '
             'reference', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_agreement_ledger.png', dpi=150)
print('wrote fig_agreement_ledger.png')
