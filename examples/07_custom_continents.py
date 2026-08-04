"""Custom continents: paint a Pacific "Atlantis", map its rainfall.

Geography is an input in pyqtcm1: ``stype`` (0 ocean, 1 forest, 2 grass,
3 desert) and ``top`` on the model grid are all the model knows about
continents. ``qtcm1.surface`` builds and edits that pair; passing the
result to ``ControlRun(surface=...)`` re-derives everything downstream
(land/ocean split, drag, land-model parameters). Albedo over the *new*
land uses static per-type values diagnosed from the packaged climatology
(``albedo_mode='auto'``); the rest of the planet keeps the observed
seasonal cycle.

This script paints a flat grassland continent across the central
equatorial Pacific (the dry tongue east of the dateline), runs one year
from a cold start, and maps the December-mean precipitation - the ITCZ
splits around the new landmass and rain organizes along its coasts.

Runtime ~2 min. For restart-friendly declarative runs, save the surface
(``surf.to_netcdf('atlantis.nc')``) and use ``RunConfig(surface=path)``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from qtcm1 import surface
from qtcm1.config import RunConfig
from qtcm1.driver import ControlRun

DATA = os.environ.get('QTCM1_BNDDATA')   # None -> packaged data/r64x42

# -- build the geography: real Earth + a painted continent ---------------
surf = surface.real_earth(DATA)
surf = surface.paint(surf, lon=(175.0, 220.0), lat=(-12.0, 12.0),
                     stype=surface.GRASS, top=0.0)

# -- one year, cold start, monthly means ---------------------------------
run = ControlRun(config=RunConfig(data_path=DATA), surface=surf)
run.run_years(1, progress=lambda day, date: print(f'day {day}', end='\r'))

# -- December-mean precipitation (last month of the year) ----------------
ds = run.to_datasets()['monthly']
prec = ds['Qc'].isel(time=-1) / 28.125          # W/m2 -> mm/day
lon, lat = ds['lon'], ds['lat']

fig, ax = plt.subplots(figsize=(8.6, 4.0))
cf = ax.contourf(lon, lat, prec, levels=np.arange(0, 17, 1),
                 cmap='viridis', extend='max')
# coastlines of the *modified* surface (thick where painted)
ax.contour(lon, lat, (surf['stype'].values > 0).astype(float),
           levels=[0.5], colors='w', linewidths=0.9)
ax.set_xlabel('longitude [degrees east]')
ax.set_ylabel('latitude [degrees north]')
ax.set_title('December-mean precipitation, year 1 - with a Pacific continent')
fig.colorbar(cf, ax=ax, label='precipitation [mm day$^{-1}$]')
fig.tight_layout()
fig.savefig('custom_continent_precip.png', dpi=150)
print('\nwrote custom_continent_precip.png')
