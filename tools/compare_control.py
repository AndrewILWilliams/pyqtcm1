#!/usr/bin/env python3
"""Tier-3 initial comparison: Python control vs Fortran control (same window).

Compares years Y0..Y1 (default 2..11) of the Python run's monthly means
against the same years of the Fortran member-a control, using the exact
conventions of the baseline pipeline (control_stats.py): cos(lat)
area weighting, tropics = |lat| <= 15, Prec = Qc*86400/L, annual-mean
climatology maps, area-weighted pattern correlation of anomalies.

Prints a per-field table against the baseline acceptance tolerances
(derived from the 3-member 40-yr spread; the ~x2 column scales them by
sqrt(40/10) for the shorter window). Writes the comparison figure.

Note: the Python run uses float64 init constants (js=5 polar-filter rows);
the Fortran control is the single-precision build (js=4). Differences here
therefore bound {port + build-precision + internal variability}, with the
port itself verified exact at Tier 2.
"""

import os
import sys

import numpy as np
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

Y0, Y1 = 2, 11                              # analysis years, inclusive
HLATENT = 2.43e6
FIELDS = ['Prec', 'Ts', 'T1', 'q1', 'u1', 'v1', 'u0', 'v0',
          'Evap', 'OLR', 'cl1', 'WD']
UNITS = dict(Prec='mm/day', Ts='K', T1='K', q1='K', u1='m/s', v1='m/s',
             u0='m/s', v0='m/s', Evap='W/m2', OLR='W/m2', cl1='-',
             WD='kg/m2')
#: baseline Δ tolerances (claude/qtcm1-control-baseline.md, 40-yr window)
TOL = dict(Prec=(0.0015, 0.0081, 0.049), Ts=(0.0044, 0.00082, 0.021),
           T1=(0.025, 0.0055, 0.086), q1=(0.010, 0.018, 0.074),
           u1=(0.029, 0.057, 0.26), v1=(0.0027, 0.0068, 0.26),
           u0=(0.0038, 0.028, 0.10), v0=(None, None, 0.085),
           Evap=(0.038, 0.13, 0.70), OLR=(0.033, 0.037, 0.21),
           cl1=(3.3e-5, 0.00018, 0.0011), WD=(0.017, 0.091, 2.0))


def load_fortran(fn):
    ds = netCDF4.Dataset(fn)
    lat = np.array(ds['lat'][:])
    k0, k1 = (Y0 - 1) * 12, Y1 * 12
    out = {}
    for f in FIELDS:
        a = np.array(ds[f][k0:k1])
        out[f] = a.reshape(Y1 - Y0 + 1, 12, *a.shape[1:])
    ds.close()
    return lat, out


def load_python(fn):
    z = np.load(fn)
    years, months = z['years'], z['months']
    sel = (years >= Y0) & (years <= Y1)
    idx = np.nonzero(sel)[0]
    nyr = Y1 - Y0 + 1
    assert len(idx) == nyr * 12, f'{len(idx)} months selected'
    out = {}
    for f in FIELDS:
        key = {'Prec': 'Qc'}.get(f, f)
        a = np.stack([z[f'm{i:04d}/{key}'] for i in idx]).astype(np.float64)
        if f in ('v1', 'v0'):
            a = a[:, 1:]                       # v grid -> Fortran rows 1..ny
        out[f] = a.reshape(nyr, 12, *a.shape[1:])
    return out


def aw_mean(x, w, latmask=None):
    if latmask is not None:
        x = x[..., latmask, :]
        w = w[latmask]
    ww = np.broadcast_to(w[:, None], x.shape[-2:])
    return (x * ww).sum(axis=(-2, -1)) / ww.sum()


def field_stats(a, w, tropics):
    if a.ndim == 4:
        annual = a.mean(axis=1)
    gm = aw_mean(annual, w)
    tm = aw_mean(annual, w, tropics)
    return dict(gmean=gm.mean(), gstd=gm.std(ddof=1),
                tmean=tm.mean(), tstd=tm.std(ddof=1),
                amap=annual.mean(axis=0))


def _pair_metrics(sa, sb, w):
    """Scalar diffs + map RMS/pattern-corr between two field_stats dicts."""
    dg = abs(sb['gmean'] - sa['gmean'])
    dtm = abs(sb['tmean'] - sa['tmean'])
    ca, cb = sa['amap'], sb['amap']
    ww = np.broadcast_to(w[:, None], ca.shape)
    rms = np.sqrt((((cb - ca) ** 2) * ww).sum() / ww.sum())
    ai = ca - (ca * ww).sum() / ww.sum()
    bi = cb - (cb * ww).sum() / ww.sum()
    pc = ((ai * bi * ww).sum()
          / np.sqrt((ai ** 2 * ww).sum() * (bi ** 2 * ww).sum()))
    vr = (sb['gstd'] / sa['gstd']) ** 2 if sa['gstd'] > 0 else np.nan
    return dg, dtm, rms, pc, vr


def main():
    fort_fn = os.path.expanduser('~/work/run/ctrl_a/qm_ctrl_a.nc')
    null_fn = os.path.expanduser('~/work/run/ctrl_b/qm_ctrl_b.nc')
    py_fn = os.environ.get('PYCTRL',
                           os.path.expanduser('~/work/run/py_ctrl_monthly.npz'))
    lat, fort = load_fortran(fort_fn)
    _, fortb = load_fortran(null_fn)
    py = load_python(py_fn)
    w = np.cos(np.deg2rad(lat))
    tropics = np.abs(lat) <= 15.0

    print(f'Python (f64 build) vs Fortran member-a (f32 build), years '
          f'{Y0}-{Y1}.\nNull columns: Fortran member-b vs member-a over the '
          f'SAME window (pure internal\nvariability between two realizations '
          f'of the same build). Verdict PASS =\neach py metric within '
          f'max(sqrt(40/N)-scaled 40-yr tolerance, 1.5x null).\n')
    hdr = (f'{"field":>5s} | {"dGmean":>9s} {"null":>9s} | {"dTmean":>9s} '
           f'{"null":>9s} | {"mapRMS":>9s} {"null":>9s} | {"pcorr":>8s} '
           f'{"null":>8s} | {"vr":>5s}  verdict')
    print(hdr)
    results = {}
    for f in FIELDS:
        a = np.array(fort[f], dtype=np.float64)
        b = py[f]
        nb = np.array(fortb[f], dtype=np.float64)
        if f == 'Prec':
            a = a * 86400.0 / HLATENT
            b = b * 86400.0 / HLATENT
            nb = nb * 86400.0 / HLATENT
        sa = field_stats(a, w, tropics)
        sb = field_stats(b, w, tropics)
        sn = field_stats(nb, w, tropics)
        dg, dtm, rms, pc, vr = _pair_metrics(sa, sb, w)
        ndg, ndt, nrms, npc, _ = _pair_metrics(sa, sn, w)
        tg, tt, tr = TOL[f]
        scale = np.sqrt(40.0 / (Y1 - Y0 + 1))    # shorter-window envelope
        lim_g = max(scale * tg, 1.5 * ndg) if tg is not None else None
        lim_t = max(scale * tt, 1.5 * ndt) if tt is not None else None
        lim_r = max(scale * tr, 1.5 * nrms)
        ok = ((lim_g is None or dg <= lim_g)
              and (lim_t is None or dtm <= lim_t) and rms <= lim_r)
        results[f] = dict(fort=sa, py=sb, dg=dg, dt=dtm, rms=rms, pc=pc,
                          vr=vr, null=dict(dg=ndg, dt=ndt, rms=nrms, pc=npc),
                          ok=bool(ok))
        print(f'{f:>5s} | {dg:9.2e} {ndg:9.2e} | {dtm:9.2e} {ndt:9.2e} | '
              f'{rms:9.2e} {nrms:9.2e} | {pc:8.5f} {npc:8.5f} | '
              f'{vr:5.2f}  {"PASS" if ok else "CHECK"}')

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10))
    lon = np.arange(64) * 5.625
    for col, f in enumerate(['Prec', 'Ts']):
        a = np.array(fort[f], dtype=np.float64)
        b = py[f]
        if f == 'Prec':
            a *= 86400.0 / HLATENT
            b *= 86400.0 / HLATENT
        ca = a.mean(axis=(0, 1))
        cb = b.mean(axis=(0, 1))
        vmax = np.percentile(ca, 99)
        vmin = ca.min()
        for row, (fld, ttl) in enumerate(
                [(cb, f'pure-Python (f64), {f}'),
                 (ca, f'Fortran control (f32), {f}')]):
            ax = axes[row, col]
            im = ax.pcolormesh(lon, lat, fld, cmap='viridis',
                               vmin=vmin, vmax=vmax, shading='auto')
            plt.colorbar(im, ax=ax, label=UNITS[f])
            ax.set_title(f'{ttl}  [{Y0}-{Y1} yr mean]', fontsize=10)
        d = cb - ca
        s = np.abs(d).max()
        ax = axes[2, col]
        im = ax.pcolormesh(lon, lat, d, cmap='RdBu_r', vmin=-s, vmax=s,
                           shading='auto')
        plt.colorbar(im, ax=ax, label=UNITS[f])
        rmsd = results[f]['rms']
        ax.set_title(f'Python - Fortran  (map RMS {rmsd:.3g} {UNITS[f]})',
                     fontsize=10)
        ax.axhline(61.875, color='k', lw=0.5, ls=':')
        ax.axhline(-61.875, color='k', lw=0.5, ls=':')
    for ax in axes.flat:
        ax.set_ylim(lat.min(), lat.max())
    npass = sum(r['ok'] for r in results.values())
    fig.suptitle(f'Tier-3 initial check, years {Y0}-{Y1}: pure-Python '
                 f'(float64 build) vs Fortran control (float32 build) - '
                 f'{npass}/{len(results)} fields within envelope',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fn = os.path.expanduser('~/work/run/fig_control10.png')
    fig.savefig(fn, dpi=140)
    print(f'\nwrote {fn}')
    return results


if __name__ == '__main__':
    main()
