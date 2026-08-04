"""Prescribed-SST anomaly experiment (El Nino-like warm patch).

Because SST is a boundary condition, anomaly experiments need no model
changes: drive the day loop yourself (mirroring ControlRun.advance_day)
and add the anomaly to the climatological SST before it is applied.
One spin-up year + one analysis year each for control and anomaly runs.
"""

import os

import numpy as np

from qtcm1.config import RunConfig
from qtcm1.driver import ControlRun

DATA = os.environ.get('QTCM1_BNDDATA')   # None -> packaged data/r64x42

# -- an idealized +2 K warm patch, central/eastern equatorial Pacific ----
_bd = ControlRun(config=RunConfig(data_path=DATA)).bd
lat, lon = _bd.lat, _bd.lon
LAT, LON = np.meshgrid(lat, lon, indexing='ij')
anom = (2.0 * np.exp(-(LAT / 10.0) ** 2)
        * np.exp(-((LON - 210.0) / 40.0) ** 2))


def run_years(sst_anomaly, nyears=2):
    """Custom day loop = ControlRun.advance_day with perturbed SST."""
    run = ControlRun(config=RunConfig(data_path=DATA))
    acc, n = 0.0, 0
    for day in range(1, nyears * 365 + 1):
        run.dayofmodel += 1
        date = run.calendar.timemanager(run.dayofmodel)
        doy = date.dayofyear
        alb = np.asarray(run.bd.albedo(doy))
        if run._getbnd_virgin:                 # getbnd first-call SST skip
            run._getbnd_virgin = False
        else:
            sst = run.bd.sst(date.yearofmodel, doy) + sst_anomaly
            run.state, _ = run.model.apply_boundary(run.state, sst, alb)
        for it in range(1, 73):                # one coupling day
            run.state, diags = run.model.step(run.state, alb, doy, it)
        if day > (nyears - 1) * 365:           # accumulate final year
            acc, n = acc + diags['Qc'], n + 1
    return acc / n


qc_ctrl = run_years(0.0)
qc_nino = run_years(anom)
dprec = (qc_nino - qc_ctrl) * 86400.0 / 2.43e6            # mm/day

box = (np.abs(lat) <= 5)[:, None] & ((lon >= 180) & (lon <= 260))[None, :]
print(f'Nino-region precipitation response: {dprec[box].mean():+.2f} mm/day')
print(f'global-mean response:               {dprec.mean():+.3f} mm/day')

# -- figure: precipitation response with the SST anomaly overlaid --------
import matplotlib.pyplot as plt                          # noqa: E402

land = _bd.stype > 0
lim = float(np.abs(dprec).max())
fig, ax = plt.subplots(figsize=(8.6, 4.0))
cf = ax.contourf(lon, lat, dprec, levels=np.linspace(-lim, lim, 17),
                 cmap='RdBu_r')
cs = ax.contour(lon, lat, anom, levels=[0.5, 1.0, 1.5],
                colors='k', linewidths=0.8)
ax.clabel(cs, fmt='%.1f K', fontsize=7)
ax.contour(lon, lat, land.astype(float), levels=[0.5],
           colors='0.35', linewidths=0.7)
ax.set_title('year-2 precipitation response to the warm patch')
ax.set_xlabel('longitude [degrees east]')
ax.set_ylabel('latitude [degrees north]')
fig.colorbar(cf, ax=ax, label=r'$\Delta$ precipitation [mm day$^{-1}$]')
fig.tight_layout()
fig.savefig('sst_anomaly_response.png', dpi=150)
print('wrote sst_anomaly_response.png')
