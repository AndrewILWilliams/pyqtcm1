"""Slab ocean with an idealized greenhouse (CO2-like) forcing.

The full version of the experiment example 03 could only approximate:
with the mixed-layer ocean the SST responds to the forcing. Workflow:

1. spin up with fixed climatological SST;
2. branch a slab-ocean control (its Q-flux, diagnosed from a fixed-SST
   control, keeps the slab on the control climatology);
3. branch a slab run with +F W/m2 added to the column longwave budget;
4. difference the two equilibria.

Runtime ~15 min with the default lengths; increase them for smoother
statistics.
"""

import os

import numpy as np

import qtcm1.model
from qtcm1.config import RunConfig
from qtcm1.driver import ControlRun
from qtcm1.physics.radiation import radlw as radlw_orig

DATA = os.path.expanduser(os.environ.get('QTCM1_BNDDATA',
                                         '~/qtcm1_data/r64x42'))
FORCING = 4.0                              # W/m2
SPINUP_YEARS, SLAB_YEARS = 2, 2            # increase for production


def radlw_forced(T1, q1, Ts, cld):
    out = radlw_orig(T1, q1, Ts, cld)
    out['FLW'] = out['FLW'] + FORCING
    out['FLWds'] = out['FLWds'] + FORCING  # surface sees the forcing too
    out['FLWut'] = out['FLWut'] - FORCING
    return out


# 1. fixed-SST spin-up ---------------------------------------------------
spin = ControlRun(config=RunConfig(data_path=DATA))
spin.run_years(SPINUP_YEARS)
spin.save_restart('spinup.restart.npz')


def slab_run(forced):
    qtcm1.model.radlw = radlw_forced if forced else radlw_orig
    run = ControlRun.from_restart('spinup.restart.npz', config=RunConfig(
        data_path=DATA, sst_mode='mixed_layer'))
    run.run_years(SLAB_YEARS)
    last12 = run.monthly[-12:]
    qtcm1.model.radlw = radlw_orig
    return {k: np.mean([m[k] for _, _, m in last12], axis=0)
            for k in ['Ts', 'T1', 'Qc']}


ctrl = slab_run(False)
pert = slab_run(True)

ocean = ControlRun(config=RunConfig(data_path=DATA)).model.stype == 0
dTs = pert['Ts'] - ctrl['Ts']
print(f'slab response to +{FORCING:.0f} W/m2 after {SLAB_YEARS} yr:')
print(f'  ocean-mean SST:   {dTs[ocean].mean():+.2f} K')
print(f'  tropospheric T1:  {(pert["T1"] - ctrl["T1"]).mean():+.2f} K')
print(f'  precipitation:    '
      f'{(pert["Qc"] - ctrl["Qc"]).mean() * 86400 / 2.43e6:+.3f} mm/day')
print('(a 50 m slab e-folds in ~1-2 yr; run longer for equilibrium)')
