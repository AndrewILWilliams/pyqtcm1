#!/usr/bin/env python3
"""Diagnose the slab-ocean Q-flux from a fixed-SST control (aveflux port).

fsn = monthly climatology of the control's net surface heat flux
      FSWds - FSWus + FLWds - FLWus - Evap - FTs;
dts = monthly climatology of Cmx * dTs/dt, with the tendency as the
      backward month-to-month difference of monthly-mean Ts divided by
      30*86400 s (first record zero), exactly as aveflux.f.

Usage:
  python make_qflux.py --qm ~/work/run/ctrl_a/qm_ctrl_a.nc \
                       --out ~/work/data/qtcm1_bnd_r64x42/qflux.nc \
                       [--y0 6] [--y1 45]
"""

import argparse
import os

import netCDF4
import numpy as np

CMX = 4.18e6 * 50.0
DATAINT = 30.0 * 86400.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qm', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--y0', type=int, default=6)
    p.add_argument('--y1', type=int, default=45)
    args = p.parse_args()

    ds = netCDF4.Dataset(os.path.expanduser(args.qm))
    n = ds['Ts'].shape[0]
    fs = (np.array(ds['FSWds'][:]) - np.array(ds['FSWus'][:])
          + np.array(ds['FLWds'][:]) - np.array(ds['FLWus'][:])
          - np.array(ds['Evap'][:]) - np.array(ds['FTs'][:]))
    ts = np.array(ds['Ts'][:])
    lat, lon = np.array(ds['lat'][:]), np.array(ds['lon'][:])
    ds.close()

    dts = np.zeros_like(ts)
    dts[1:] = (ts[1:] - ts[:-1]) / DATAINT * CMX     # aveflux backward diff

    k0, k1 = (args.y0 - 1) * 12, min(args.y1 * 12, n)
    sl = slice(k0, k1)
    months = np.arange(n) % 12                       # record 0 = January
    fsn_clim = np.stack([fs[sl][months[sl] == m].mean(axis=0)
                         for m in range(12)])
    dts_clim = np.stack([dts[sl][months[sl] == m].mean(axis=0)
                         for m in range(12)])

    out = netCDF4.Dataset(os.path.expanduser(args.out), 'w')
    out.createDimension('month', 12)
    out.createDimension('lat', lat.size)
    out.createDimension('lon', lon.size)
    for name, val in [('lat', lat), ('lon', lon)]:
        v = out.createVariable(name, 'f4', (name,))
        v[:] = val
    for name, val, units in [('fsn', fsn_clim, 'W m-2'),
                             ('dts', dts_clim, 'W m-2')]:
        v = out.createVariable(name, 'f8', ('month', 'lat', 'lon'))
        v[:] = val
        v.units = units
    out.source = (f'aveflux-equivalent from {os.path.basename(args.qm)}, '
                  f'years {args.y0}-{args.y1}; Cmx={CMX:.3e}')
    out.close()
    print(f'wrote {args.out}: fsn mean {fsn_clim.mean():+.2f} W/m2, '
          f'dts rms {np.sqrt((dts_clim**2).mean()):.2f} W/m2')


if __name__ == '__main__':
    main()
