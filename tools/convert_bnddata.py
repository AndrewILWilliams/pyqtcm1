#!/usr/bin/env python3
"""Convert QTCM1 ASCII boundary data (bnddir) to CF-style netCDF.

Lossless: values are parsed exactly as Fortran list-directed reads do
(whitespace tokens, D-exponents normalized, Fortran column-major order,
grid (nx=64, ny=42)) and stored unmodified. Units reflect the *verified*
content of the files (e.g. SST files are Kelvin despite a stale "Celsius"
comment in ocean.F90).

Outputs one netCDF per dataset plus a sha256 manifest of sources/products.
"""
import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np
import netCDF4

NX, NY = 64, 42
LON = 0.0 + 5.625 * np.arange(NX)                      # T-point longitudes
LAT = -76.875 + 3.75 * np.arange(NY)                   # T-point latitudes
MONTH_DAYS15 = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]


def read_ascii_field(path, expect=NX * NY):
    """Parse a Fortran list-directed ASCII array file -> (ny, nx) array."""
    txt = open(path).read()
    txt = re.sub(r'([0-9.])[dD]([+-]?[0-9])', r'\1E\2', txt)   # D-exp -> E-exp
    toks = txt.split()
    assert not any('*' in t for t in toks), f'repeat syntax in {path}'
    a = np.array([float(t) for t in toks])
    assert a.size == expect, f'{path}: {a.size} values, expected {expect}'
    return a.reshape((NX, NY), order='F').T                     # -> (ny, nx)


def nc_create(path, title, extra_dims=None):
    ds = netCDF4.Dataset(path, 'w')
    ds.title = title
    ds.source = 'QTCM1 v2.3 bnddir (r64x42), converted from ASCII'
    ds.Conventions = 'CF-1.8'
    ds.history = 'convert_bnddata.py (pyqtcm1 project)'
    ds.createDimension('lat', NY)
    ds.createDimension('lon', NX)
    for name, size in (extra_dims or {}).items():
        ds.createDimension(name, size)
    la = ds.createVariable('lat', 'f4', ('lat',))
    la[:] = LAT; la.units = 'degrees_north'; la.long_name = 'latitude (T points)'
    lo = ds.createVariable('lon', 'f4', ('lon',))
    lo[:] = LON; lo.units = 'degrees_east'; lo.long_name = 'longitude (T points)'
    return ds


def add_month_coord(ds):
    m = ds.createVariable('month', 'i4', ('month',))
    m[:] = np.arange(1, 13); m.long_name = 'climatological month'
    return m


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def convert_sst_clim(bnd, out, manifest):
    files = [f'{bnd}/SST_Reynolds/0000{m:02d}15.sst' for m in range(1, 13)]
    data = np.stack([read_ascii_field(f) for f in files])
    ds = nc_create(f'{out}/sst_reynolds_clim.nc',
                   'Reynolds SST, monthly climatology (mid-month)',
                   {'month': 12})
    add_month_coord(ds)
    v = ds.createVariable('sst', 'f8', ('month', 'lat', 'lon'))
    v[:] = data; v.units = 'K'
    v.long_name = ('sea surface temperature, monthly climatology; ocean-'
                   'interpolated over land (Reynolds); mid-month values')
    v.comment = ('Fortran source files are Kelvin despite the "in Celsius" '
                 'comment in ocean.F90:sstin')
    ds.close()
    manifest['sources'] += [{'file': f, 'sha256': sha256(f)} for f in files]


def convert_sst_dated(bnd, out, manifest):
    files = sorted(glob.glob(f'{bnd}/SST_Reynolds/[12]*.sst'))
    dates = [os.path.basename(f)[:8] for f in files]
    yy = np.array([int(d[:4]) for d in dates])
    mm = np.array([int(d[4:6]) for d in dates])
    # verify contiguity of the monthly series
    idx = yy * 12 + (mm - 1)
    assert np.all(np.diff(idx) == 1), 'dated SST series has gaps'
    data = np.stack([read_ascii_field(f) for f in files])
    ds = nc_create(f'{out}/sst_reynolds_1949_2001.nc',
                   'Reynolds SST, observed monthly means (mid-month)',
                   {'time': len(files)})
    t = ds.createVariable('time', 'f8', ('time',))
    t.units = 'days since 1949-01-01'; t.calendar = '365_day'
    t[:] = [(y - 1949) * 365 + MONTH_DAYS15[m - 1] for y, m in zip(yy, mm)]
    v = ds.createVariable('sst', 'f8', ('time', 'lat', 'lon'))
    v[:] = data; v.units = 'K'
    v.long_name = 'sea surface temperature, observed monthly (mid-month values)'
    ds.close()
    manifest['sources'] += [{'file': f, 'sha256': sha256(f)} for f in files]


def convert_sst_perpetual(bnd, out, manifest):
    f = f'{bnd}/00000000.sst'
    ds = nc_create(f'{out}/sst_perpetual.nc', 'Perpetual SST (SSTmode=perpetual)')
    v = ds.createVariable('sst', 'f8', ('lat', 'lon'))
    v[:] = read_ascii_field(f); v.units = 'K'
    ds.close()
    manifest['sources'].append({'file': f, 'sha256': sha256(f)})


def convert_albedo(bnd, out, manifest):
    files = [f'{bnd}/ALBD_Darnell/0000{m:02d}15.alb' for m in range(1, 13)]
    ann = f'{bnd}/ALBD_Darnell/00001315.alb'      # "month 13" = annual mean
    data = np.stack([read_ascii_field(f) for f in files])
    ds = nc_create(f'{out}/albedo_darnell.nc',
                   'Darnell surface albedo, monthly climatology + annual mean',
                   {'month': 12})
    add_month_coord(ds)
    v = ds.createVariable('albedo', 'f8', ('month', 'lat', 'lon'))
    v[:] = data; v.units = '1'
    v.long_name = 'surface albedo, monthly climatology (mid-month values)'
    va = ds.createVariable('albedo_annual', 'f8', ('lat', 'lon'))
    va[:] = read_ascii_field(ann); va.units = '1'
    va.long_name = 'annual-mean surface albedo (Fortran file 00001315.alb)'
    va.comment = 'used by bndinit for initialization'
    ds.close()
    manifest['sources'] += [{'file': f, 'sha256': sha256(f)}
                            for f in files + [ann]]


def convert_surface(bnd, out, manifest):
    ds = nc_create(f'{out}/surface.nc', 'QTCM1 surface type and topography')
    stype = read_ascii_field(f'{bnd}/STYPE')
    v = ds.createVariable('stype', 'i2', ('lat', 'lon'))
    v[:] = stype.astype(np.int16); v.units = '1'
    v.long_name = 'surface type'
    v.flag_values = np.array([0, 1, 2, 3], np.int16)
    v.flag_meanings = 'ocean forest grass desert'
    top = read_ascii_field(f'{bnd}/TOP')
    vt = ds.createVariable('top', 'f8', ('lat', 'lon'))
    vt[:] = top; vt.units = '10 km'
    vt.long_name = 'relative topography height/10km (used by TOPO option)'
    ds.close()
    manifest['sources'] += [{'file': f'{bnd}/{n}', 'sha256': sha256(f'{bnd}/{n}')}
                            for n in ('STYPE', 'TOP')]


def read_cloud_file(path):
    """Cloud files: one value/line, loops n(types), j(rows), i(1..64).

    CLOUD_ISCCP is (3 types, 32 rows) as consumed by readobscloud
    (nyOBS=32, jskip=5). CLOUD_ISCCP7 holds (7 types, 33 rows); no reader
    in QTCM1 v2.3 consumes it -- layout inferred, kept for completeness.
    """
    txt = re.sub(r'([0-9.])[dD]([+-]?[0-9])', r'\1E\2', open(path).read())
    toks = [float(t) for t in txt.split()]
    for nyobs in (32, 33, NY):
        if len(toks) % (NX * nyobs) == 0:
            ntype = len(toks) // (NX * nyobs)
            if ntype in (3, 4, 7):
                return np.array(toks).reshape(ntype, nyobs, NX)
    raise ValueError(f'{path}: cannot infer layout for {len(toks)} values')


def convert_clouds(bnd, out, manifest, sub='CLOUD_ISCCP'):
    files = sorted(glob.glob(f'{bnd}/{sub}/0000[01][0-9]15.cld'))
    monthly = [f for f in files if '0000015' not in os.path.basename(f)]
    data = np.stack([read_cloud_file(f) for f in monthly])
    ntype, nyobs = data.shape[1], data.shape[2]
    ds = nc_create(f'{out}/{sub.lower()}.nc',
                   f'ISCCP cloud-cover climatology ({sub}, OBSCLD option)',
                   {'month': 12, 'cloud_type': ntype, 'lat_obs': nyobs})
    add_month_coord(ds)
    lo = ds.createVariable('lat_obs', 'f4', ('lat_obs',))
    js = (NY - nyobs) // 2
    lo[:] = LAT[js:js + nyobs]; lo.units = 'degrees_north'
    lo.long_name = f'observed-cloud latitudes (rows {js+1}..{js+nyobs} of model grid)'
    v = ds.createVariable('cloud_cover', 'f8',
                          ('month', 'cloud_type', 'lat_obs', 'lon'))
    v[:] = data; v.units = '1'
    v.long_name = 'cloud cover fraction by type'
    v.comment = ('Fortran fills rows outside lat_obs with the boundary-row '
                 'zonal mean at run time (readobscloud); stored here unfilled')
    ds.close()
    manifest['sources'] += [{'file': f, 'sha256': sha256(f)} for f in monthly]


def convert_masks(bnd, out, manifest):
    f = f'{bnd}/ensopac.mask'
    ds = nc_create(f'{out}/masks.nc', 'QTCM1 region masks')
    v = ds.createVariable('ensopac', 'f8', ('lat', 'lon'))
    v[:] = read_ascii_field(f); v.units = '1'
    v.long_name = ('ENSO Pacific mask: 1 = prescribed SST, '
                   '0 = mixed-layer ocean (BLEND_SST option)')
    ds.close()
    manifest['sources'].append({'file': f, 'sha256': sha256(f)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bnddir', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = {'grid': 'r64x42', 'sources': [], 'products': []}
    convert_sst_clim(args.bnddir, args.out, manifest)
    convert_sst_dated(args.bnddir, args.out, manifest)
    convert_sst_perpetual(args.bnddir, args.out, manifest)
    convert_albedo(args.bnddir, args.out, manifest)
    convert_surface(args.bnddir, args.out, manifest)
    for sub in ('CLOUD_ISCCP', 'CLOUD_ISCCP7'):
        if os.path.isdir(os.path.join(args.bnddir, sub)):
            convert_clouds(args.bnddir, args.out, manifest, sub)
    convert_masks(args.bnddir, args.out, manifest)
    for f in sorted(glob.glob(f'{args.out}/*.nc')):
        manifest['products'].append({'file': os.path.basename(f),
                                     'sha256': sha256(f)})
    json.dump(manifest, open(f'{args.out}/manifest.json', 'w'), indent=1)
    print('converted products:')
    for prod in manifest['products']:
        print('  ', prod['file'])


if __name__ == '__main__':
    main()
