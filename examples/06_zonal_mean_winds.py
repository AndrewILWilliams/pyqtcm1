"""Zonal-mean zonal wind as a function of latitude and pressure.

The prognostic winds are *mode amplitudes*: the barotropic ``u0`` and the
baroclinic ``u1``. Full profiles are Galerkin reconstructions (NZ 3.10)

    u(x, y, p) = u0(x, y) + V1(p) u1(x, y),

so a latitude-pressure section costs one broadcast against the basis
function. ``load_basis`` returns V1(p) (and the a1 family it derives
from) as an xarray Dataset; V1 < 0 in the lower troposphere and > 0
aloft, with its node near 500 hPa - trades against westerlies flip sign
there, giving the classic first-baroclinic structure.

Runtime ~2-4 min (2 years); the annual mean of year 2 is shown.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from qtcm1 import load_basis
from qtcm1.config import RunConfig
from qtcm1.driver import ControlRun

DATA = os.environ.get('QTCM1_BNDDATA')   # None -> packaged data/r64x42
YEARS = 2

run = ControlRun(config=RunConfig(data_path=DATA))
run.run_years(YEARS, progress=lambda day, date: print(f'day {day}', end='\r'))

# annual + zonal mean of the final year's monthly mode amplitudes
ds = run.to_datasets()['monthly'].isel(time=slice(-12, None)).mean('time')
ubar0 = ds['u0'].mean('lon')             # (lat,)
ubar1 = ds['u1'].mean('lon')

# reconstruct u(p, lat) = u0 + V1(p) u1 on 41 levels
basis = load_basis(levels=41)
u = ubar0 + basis['V1'] * ubar1
u.attrs.update(units='m s-1', long_name='zonal-mean zonal wind')

fig, ax = plt.subplots(figsize=(7.0, 4.2))
step = 5.0                                          # m/s, clean contour step
lim = step * np.ceil(float(abs(u).max()) / step)
cf = ax.contourf(u['lat'], u['p'], u.transpose('p', 'lat'),
                 levels=np.arange(-lim, lim + step / 2, step), cmap='RdBu_r')
ax.contour(u['lat'], u['p'], u.transpose('p', 'lat'),
           levels=[0.0], colors='0.2', linewidths=0.8)
ax.invert_yaxis()                        # pressure decreases upward
ax.set_xlabel('latitude [degrees north]')
ax.set_ylabel('pressure [hPa]')
ax.set_title(f'Zonal-mean zonal wind, year-{YEARS} mean')
fig.colorbar(cf, ax=ax, label='u [m s$^{-1}$]')
fig.tight_layout()
fig.savefig('zonal_mean_winds.png', dpi=150)
print('\nwrote zonal_mean_winds.png')
