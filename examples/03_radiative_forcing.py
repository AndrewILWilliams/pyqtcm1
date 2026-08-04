"""Idealized radiative forcing (a CO2-like perturbation), fixed SST.

QTCM1 v2.3 has no explicit CO2 parameter; greenhouse-gas experiments
are done by perturbing the longwave budget. Every physics routine here
is a pure function, so until first-class intervention hooks land the
pattern is to wrap the routine at the point the model imports it: below,
``radlw`` gets +F W/m2 added to the column longwave heating FLW (and
removed from OLR), i.e. an idealized top-of-atmosphere greenhouse
forcing.

NOTE the scientific caveat: with prescribed SST the ocean cannot warm,
so this measures the fast/atmospheric-and-land response only. The full
equilibrium response needs the slab (mixed-layer) ocean, which is part
of the original model's option set not yet ported (see the roadmap in
the docs).
"""

import os

import numpy as np

import qtcm1.model
from qtcm1.config import RunConfig
from qtcm1.driver import ControlRun
from qtcm1.physics.radiation import radlw as radlw_orig

DATA = os.environ.get('QTCM1_BNDDATA')   # None -> packaged data/r64x42
FORCING = 4.0                              # W/m2, ~2xCO2-like


def radlw_forced(T1, q1, Ts, cld):
    out = radlw_orig(T1, q1, Ts, cld)
    out['FLW'] = out['FLW'] + FORCING      # column LW heating
    out['FLWut'] = out['FLWut'] - FORCING  # OLR reduced by the forcing
    return out


def run_years(forced, nyears=2):
    qtcm1.model.radlw = radlw_forced if forced else radlw_orig
    run = ControlRun(config=RunConfig(data_path=DATA))
    run.run_years(nyears)
    last12 = run.monthly[-12:]
    return {k: np.mean([m[k] for _, _, m in last12], axis=0)
            for k in ['T1', 'Ts', 'Qc']}


ctrl = run_years(False)
pert = run_years(True)
qtcm1.model.radlw = radlw_orig             # restore

land = ControlRun(config=RunConfig(data_path=DATA)).model.stype > 0
print(f'fast response to +{FORCING:.0f} W/m2 (fixed SST):')
print(f'  tropospheric T1:  {(pert["T1"] - ctrl["T1"]).mean():+.3f} K')
print(f'  land surface Ts:  {(pert["Ts"] - ctrl["Ts"])[land].mean():+.3f} K')
print(f'  precipitation:    '
      f'{(pert["Qc"] - ctrl["Qc"]).mean() * 86400 / 2.43e6:+.4f} mm/day')

# -- figure: the fast response pattern (land warms, ocean cannot) --------
import matplotlib.pyplot as plt                          # noqa: E402

bd = ControlRun(config=RunConfig(data_path=DATA)).bd
lat, lon = bd.lat, bd.lon
dTs = pert['Ts'] - ctrl['Ts']
dprec = (pert['Qc'] - ctrl['Qc']) * 86400.0 / 2.43e6
fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True)
lim = float(np.abs(dTs).max())
cf0 = axes[0].contourf(lon, lat, dTs, levels=np.linspace(-lim, lim, 17),
                       cmap='RdBu_r')
axes[0].set_title(f'surface temperature response to +{FORCING:.0f} W/m$^2$'
                  ' (fixed SST: land only)')
fig.colorbar(cf0, ax=axes[0], label=r'$\Delta T_s$ [K]')
lim = float(np.abs(dprec).max())
cf1 = axes[1].contourf(lon, lat, dprec, levels=np.linspace(-lim, lim, 17),
                       cmap='RdBu_r')
axes[1].set_title('precipitation response')
fig.colorbar(cf1, ax=axes[1],
             label=r'$\Delta$ precipitation [mm day$^{-1}$]')
for ax in axes:
    ax.contour(lon, lat, (bd.stype > 0).astype(float), levels=[0.5],
               colors='0.35', linewidths=0.7)
    ax.set_ylabel('latitude [degrees north]')
axes[1].set_xlabel('longitude [degrees east]')
fig.tight_layout()
fig.savefig('radiative_forcing_fast_response.png', dpi=150)
print('wrote radiative_forcing_fast_response.png')
