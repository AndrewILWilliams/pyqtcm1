"""Control run: cold start, climatological seasonal SST, monthly output.

The standard QTCM1 experiment. ~1 min per simulated year single-core.
"""

import os

from qtcm1.config import RunConfig
from qtcm1.driver import ControlRun

DATA = os.environ.get('QTCM1_BNDDATA')   # None -> packaged data/r64x42

cfg = RunConfig(data_path=DATA)            # build='f64' (recommended)
run = ControlRun(config=cfg)
run.run_years(3, progress=lambda day, date: print(f'  year {day // 365}'))
run.save_monthly('control_monthly.npz')
run.save_restart('control_y3.restart.npz')

# quick look: tropical-mean precipitation seasonal cycle, year 3
import numpy as np                                       # noqa: E402

lat = run.bd.lat
w = np.cos(np.deg2rad(lat))
tropics = np.abs(lat) <= 15.0
for year, month, mean in run.monthly[-12:]:
    prec = mean['Qc'] * 86400.0 / 2.43e6                 # W/m2 -> mm/day
    pm = (prec[tropics] * w[tropics, None]).sum() / (w[tropics].sum()
                                                     * prec.shape[1])
    print(f'{year:04d}-{month:02d}  tropical Prec = {pm:5.2f} mm/day')

# -- figure: January / July precipitation, final year (NZ Fig. 2 view) ---
import matplotlib.pyplot as plt                          # noqa: E402

lon = run.bd.lon
land = run.model.stype > 0
months = {m: mean for _, m, mean in run.monthly[-12:]}
fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True)
for ax, m, name in [(axes[0], 1, 'January'), (axes[1], 7, 'July')]:
    prec = months[m]['Qc'] * 86400.0 / 2.43e6
    cf = ax.contourf(lon, lat, prec, levels=np.arange(0, 17, 1),
                     cmap='viridis', extend='max')
    ax.contour(lon, lat, land.astype(float), levels=[0.5],
               colors='w', linewidths=0.9)
    ax.set_title(f'{name}-mean precipitation, year 3')
    ax.set_ylabel('latitude [degrees north]')
    fig.colorbar(cf, ax=ax, label='precipitation [mm day$^{-1}$]')
axes[1].set_xlabel('longitude [degrees east]')
fig.tight_layout()
fig.savefig('control_precip_janjul.png', dpi=150)
print('wrote control_precip_janjul.png')
