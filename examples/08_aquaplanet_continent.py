"""A lone equatorial continent on an otherwise water-covered planet.

Start from ``surface.aquaplanet()`` (all ocean, flat), paint a single
grassland continent in the equatorial Pacific, and make the ocean
zonally symmetric too: ``sst_mode='zonal'`` replaces the SST with its
ocean-masked zonal mean (observed meridional structure and seasonal
cycle, no warm pool, no under-land fill values). Every deviation from
zonal symmetry in the result is then *caused by the continent* — which
is the point of the experiment.

The land is fully interactive (prognostic ground temperature, soil
moisture, runoff); its albedo is the static grassland value diagnosed
from the packaged climatology (``albedo_mode='auto'``, and on an
aquaplanet every point counts as "changed", so the whole planet uses
per-type values - ocean albedo included).

The figure maps December-mean precipitation (last month of a one-year
run) and its deviation from the zonal mean: rain organizes over and
around the lone landmass while the rest of the ITCZ stays a symmetric
band. Runtime ~2 min.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from qtcm1 import surface
from qtcm1.config import RunConfig
from qtcm1.driver import ControlRun

DATA = os.environ.get('QTCM1_BNDDATA')   # None -> packaged data/r64x42

# -- geography: water world + one continent ------------------------------
surf = surface.aquaplanet(DATA)
surf = surface.paint(surf, lon=(175.0, 220.0), lat=(-12.0, 12.0),
                     stype=surface.GRASS, top=0.0)

# -- one year, cold start, zonally symmetric ocean -----------------------
run = ControlRun(config=RunConfig(data_path=DATA, sst_mode='zonal'),
                 surface=surf)
run.run_years(1, progress=lambda day, date: print(f'day {day}', end='\r'))

# -- December-mean precipitation and its zonal anomaly -------------------
ds = run.to_datasets()['monthly']
prec = ds['Qc'].isel(time=-1) / 28.125          # W/m2 -> mm/day
anom = prec - prec.mean('lon')
lon, lat = ds['lon'], ds['lat']
land = (surf['stype'].values > 0).astype(float)

fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True)
cf0 = axes[0].contourf(lon, lat, prec, levels=np.arange(0, 17, 1),
                       cmap='viridis', extend='max')
axes[0].set_title('December-mean precipitation - aquaplanet + one continent')
fig.colorbar(cf0, ax=axes[0], label='precipitation [mm day$^{-1}$]')

lim = float(np.abs(anom).max())
cf1 = axes[1].contourf(lon, lat, anom, levels=np.linspace(-lim, lim, 17),
                       cmap='RdBu_r')
axes[1].set_title('deviation from the zonal mean (the continent signal)')
fig.colorbar(cf1, ax=axes[1], label='precipitation anomaly [mm day$^{-1}$]')

for ax in axes:
    ax.contour(lon, lat, land, levels=[0.5], colors='k', linewidths=0.9)
    ax.set_ylabel('latitude [degrees north]')
axes[1].set_xlabel('longitude [degrees east]')
fig.tight_layout()
fig.savefig('aquaplanet_continent_precip.png', dpi=150)
print('\nwrote aquaplanet_continent_precip.png')
