"""User-configurable continents and topography.

Geography enters QTCM1 through exactly two fields on the model grid:
``stype`` (surface type: 0 ocean, 1 forest, 2 grassland, 3 desert) and
``top`` (relative topography, units of *height/10 km*, only active under
the ``TOPO`` option). Everything else geographic — the land/ocean split,
drag coefficients, land-model parameters, the slab-ocean domain — is
derived from ``stype`` at run construction, so swapping the surface is
enough to change the continents.

Builders (all return an :class:`xarray.Dataset` with ``stype``/``top``):

* :func:`real_earth` — the packaged surface, as an editable copy;
* :func:`aquaplanet` — all ocean, flat;
* :func:`paint` — set a lat/lon box (or an arbitrary boolean mask) to a
  surface type, optionally with topography.

Pass the result to :class:`qtcm1.driver.ControlRun` via the ``surface=``
argument (in-memory), or save it with ``ds.to_netcdf(path)`` and set
``RunConfig(surface=path)`` for a declarative, restart-friendly run.

Earth-locked couplings, handled explicitly (no silent behavior):

* **Albedo.** The Darnell climatology knows where Earth's continents
  are. Where a custom surface *changes* ``stype``, the run uses a static
  per-type albedo diagnosed from the packaged data over the original
  mask (``albedo_mode='auto'``, the default; ``'by_stype'`` applies the
  static map everywhere, ``'darnell'`` forces the climatology and is
  almost certainly wrong over moved coastlines).
* **SST.** Prescribed SST covers the full grid, including under
  real-Earth land, so ocean created by *removing* land runs with the
  dataset's under-land fill values — smooth but not observationally
  meaningful there. A warning is raised; supply your own SST handling if
  the experiment depends on those regions.
* **Q-flux.** ``qflux.nc`` was diagnosed from the real geography; slab
  ocean modes therefore refuse a custom surface (rerun a fixed-SST
  control on the new geography and ``tools/make_qflux.py`` first).
"""

from __future__ import annotations

import numpy as np

#: surface type codes (bndinit / land.F90)
OCEAN, FOREST, GRASS, DESERT = 0, 1, 2, 3
_STYPES = (OCEAN, FOREST, GRASS, DESERT)


def _packaged_surface_path(data_path: str | None = None) -> str:
    import os

    from .config import PACKAGED_DATA
    return os.path.join(data_path or PACKAGED_DATA, 'surface.nc')


def real_earth(data_path: str | None = None):
    """The packaged (real-Earth) surface as an editable Dataset."""
    import xarray as xr

    ds = xr.open_dataset(_packaged_surface_path(data_path)).load()
    ds.attrs['source'] = 'pyqtcm1 packaged registry (real Earth)'
    return ds


def aquaplanet(data_path: str | None = None):
    """All-ocean, flat surface on the model grid."""
    import xarray as xr

    ds = real_earth(data_path)
    out = xr.Dataset(
        {'stype': (('lat', 'lon'),
                   np.zeros(ds['stype'].shape, dtype=np.int16)),
         'top': (('lat', 'lon'), np.zeros(ds['top'].shape))},
        coords={'lat': ds['lat'], 'lon': ds['lon']})
    out['stype'].attrs.update(ds['stype'].attrs)
    out['top'].attrs.update(ds['top'].attrs)
    out.attrs['source'] = 'qtcm1.surface.aquaplanet'
    return out


def paint(ds, lon=None, lat=None, mask=None, stype=None, top=None):
    """Set a region of the surface to a type (and optionally topography).

    Parameters
    ----------
    ds:
        Surface Dataset (from :func:`real_earth`/:func:`aquaplanet`);
        not modified — a painted copy is returned.
    lon, lat:
        Region bounds in degrees, inclusive of cell centers.
        ``lon=(l0, l1)`` with ``l0 > l1`` wraps across 0°.
        Omitted bound = whole axis.
    mask:
        Alternative to the box: a boolean (lat, lon) array selecting the
        cells to paint. Mutually exclusive with ``lon``/``lat``.
    stype:
        Surface type to paint (0 ocean, 1 forest, 2 grass, 3 desert).
    top:
        Optional topography value (height/10 km units) painted over the
        same region; ``None`` leaves ``top`` untouched.
    """
    if mask is not None and (lon is not None or lat is not None):
        raise ValueError('pass either mask or lon/lat bounds, not both')
    out = ds.copy(deep=True)
    lons = np.asarray(ds['lon'].values)
    lats = np.asarray(ds['lat'].values)
    if mask is None:
        sel_lon = np.ones(lons.size, dtype=bool)
        if lon is not None:
            l0, l1 = (x % 360.0 for x in lon)
            sel_lon = ((lons >= l0) & (lons <= l1) if l0 <= l1
                       else (lons >= l0) | (lons <= l1))
        sel_lat = np.ones(lats.size, dtype=bool)
        if lat is not None:
            p0, p1 = min(lat), max(lat)
            sel_lat = (lats >= p0) & (lats <= p1)
        mask = sel_lat[:, None] & sel_lon[None, :]
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != ds['stype'].shape:
            raise ValueError(f'mask shape {mask.shape} != surface '
                             f'{ds["stype"].shape}')
    if stype is not None:
        if int(stype) not in _STYPES:
            raise ValueError(f'stype must be one of {_STYPES}')
        vals = out['stype'].values.copy()
        vals[mask] = int(stype)
        out['stype'].values = vals
    if top is not None:
        vals = out['top'].values.copy()
        vals[mask] = float(top)
        out['top'].values = vals
    return out


def validate(stype: np.ndarray, top: np.ndarray, shape: tuple):
    """Validate a custom surface; raises ValueError on structural errors."""
    stype, top = np.asarray(stype), np.asarray(top)
    if stype.shape != shape or top.shape != shape:
        raise ValueError(f'surface fields must have shape {shape}; got '
                         f'stype {stype.shape}, top {top.shape}')
    bad = ~np.isin(stype, _STYPES)
    if bad.any():
        raise ValueError(f'stype contains values outside {_STYPES}: '
                         f'{np.unique(stype[bad])}')
    if not np.isfinite(top).all():
        raise ValueError('top contains non-finite values')
    if top.max() > 0.9:
        raise ValueError(
            f'top max {top.max():.3f}: units are height/10km (Everest '
            f'≈ 0.88); a value > 0.9 suggests km or m were passed')
    return stype, top


def coerce(surface):
    """Normalize a surface spec (Dataset | path | dict) to (stype, top).

    Accepts an xarray Dataset with ``stype``/``top``, a path to a netCDF
    file with those variables, or a mapping with those keys.
    """
    if isinstance(surface, str):
        import netCDF4
        with netCDF4.Dataset(surface) as ds:
            return (np.array(ds['stype'][:]).astype(np.int16),
                    np.array(ds['top'][:]))
    if hasattr(surface, 'data_vars'):                  # xarray Dataset
        return (np.asarray(surface['stype'].values).astype(np.int16),
                np.asarray(surface['top'].values, dtype=np.float64))
    if isinstance(surface, dict):
        return (np.asarray(surface['stype']).astype(np.int16),
                np.asarray(surface['top'], dtype=np.float64))
    raise TypeError(f'cannot interpret surface spec of type '
                    f'{type(surface).__name__}')


def sha256(stype: np.ndarray, top: np.ndarray) -> str:
    """Content hash of a surface, for provenance stamping."""
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(stype, dtype=np.int16).tobytes())
    h.update(np.ascontiguousarray(top, dtype=np.float64).tobytes())
    return h.hexdigest()
