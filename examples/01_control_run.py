"""Control run: cold start, climatological seasonal SST, monthly output.

The standard QTCM1 experiment. ~1 min per simulated year single-core.
"""

import os

from qtcm1.config import RunConfig
from qtcm1.driver import ControlRun

DATA = os.path.expanduser(os.environ.get('QTCM1_BNDDATA',
                                         '~/qtcm1_data/r64x42'))

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
