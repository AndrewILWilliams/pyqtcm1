"""Configurable model output: per-variable frequency, mean or instantaneous.

A run's output is declared as a dict mapping variable names to requests:

.. code-block:: python

   output = {
       'Ts':  {'freq': 'monthly', 'kind': 'mean'},
       'Qc':  {'freq': 'daily',   'kind': 'mean'},
       'u1':  {'freq': '6h',      'kind': 'inst'},
       'T1':  {'freq': 'step',    'kind': 'inst'},   # every time step
   }

``freq`` is one of ``'step'``, ``'<n>h'`` (any divisor of 24, e.g.
``'1h'``, ``'6h'``, ``'12h'``), ``'daily'``, or ``'monthly'``.  ``kind``
is ``'mean'`` (accumulated every time step over the interval — the
Fortran ``varmean`` semantics) or ``'inst'`` (the field sampled at the
END of the interval).  Every interval is labelled by its end time on a
``cftime`` *noleap* (365-day) axis, matching the model calendar; the
``cell_methods`` attribute records ``time: mean`` or ``time: point``.

Results come back as one :class:`xarray.Dataset` per frequency
(:meth:`OutputManager.to_datasets`), with CF-style units/long names, the
run's provenance in the global attributes, and v-grid variables on their
own ``lat_v`` coordinate.  Partial intervals at the end of a run are
dropped (only complete means/samples are ever written).
"""

from __future__ import annotations

import json

import numpy as np

#: variable registry: name -> (source, units, long_name, vgrid?)
VARIABLES = {
    'u1':   ('state', 'm s-1', 'baroclinic (mode-1) zonal wind', False),
    'v1':   ('state', 'm s-1', 'baroclinic (mode-1) meridional wind', True),
    'T1':   ('state', 'K', 'temperature mode amplitude', False),
    'q1':   ('state', 'K', 'moisture mode amplitude', False),
    'u0':   ('state', 'm s-1', 'barotropic zonal wind', False),
    'v0':   ('state', 'm s-1', 'barotropic meridional wind', True),
    'Ts':   ('state', 'K', 'surface temperature', False),
    'WD':   ('state', 'kg m-2', 'soil moisture', False),
    'us':   ('state', 'm s-1', 'surface zonal wind', False),
    'vs':   ('state', 'm s-1', 'surface meridional wind', False),
    'Qc':   ('diag', 'W m-2', 'convective heating (= precipitation)', False),
    'Evap': ('diag', 'W m-2', 'surface latent heat flux', False),
    'FTs':  ('diag', 'W m-2', 'surface sensible heat flux', False),
    'OLR':  ('diag', 'W m-2', 'outgoing longwave radiation', False),
    'S0':   ('diag', 'W m-2', 'incoming solar at top', False),
    'FSWds': ('diag', 'W m-2', 'downward surface shortwave', False),
    'FSWus': ('diag', 'W m-2', 'upward surface shortwave', False),
    'FLWds': ('diag', 'W m-2', 'downward surface longwave', False),
    'FLWus': ('diag', 'W m-2', 'upward surface longwave', False),
    'cl1':  ('diag', '1', 'deep cloud fraction', False),
    'taux': ('diag', 'N m-2', 'zonal surface stress', False),
    'tauy': ('diag', 'N m-2', 'meridional surface stress', False),
    'div1': ('diag', 's-1', 'mode-1 divergence', False),
    'Runf': ('diag', 'W m-2', 'runoff', False),
    'wet':  ('diag', '1', 'relative soil wetness', False),
}

#: the default request: the standard monthly-mean archive
DEFAULT_OUTPUT = {k: {'freq': 'monthly', 'kind': 'mean'} for k in
                  ['u1', 'v1', 'T1', 'q1', 'u0', 'v0', 'Ts', 'WD', 'Qc',
                   'Evap', 'FTs', 'OLR', 'cl1', 'S0', 'taux']}


def _freq_steps(freq: str, nastep: int):
    """Interval length in steps (None for monthly, handled by calendar)."""
    if freq == 'step':
        return 1
    if freq == 'daily':
        return nastep
    if freq == 'monthly':
        return None
    if freq.endswith('h'):
        hours = int(freq[:-1])
        if 24 % hours:
            raise ValueError(f'{freq!r}: hours must divide 24')
        steps = hours * nastep / 24.0
        if steps != int(steps) or int(steps) == 0:
            raise ValueError(f'{freq!r} is not a multiple of the time step')
        return int(steps)
    raise ValueError(f'unknown output frequency {freq!r}')


class OutputManager:
    """Accumulates requested output during a run; see the module docs.

    Parameters
    ----------
    spec : dict
        ``{variable: {'freq': ..., 'kind': 'mean'|'inst'}}``.
    lat, lon : 1-D coordinate arrays (T grid; lat_v is derived).
    nastep : atmospheric steps per day.
    attrs : global attributes for the datasets (e.g. provenance).
    """

    def __init__(self, spec: dict, lat, lon, nastep: int,
                 attrs: dict | None = None):
        self.lat = np.asarray(lat)
        self.lon = np.asarray(lon)
        dlat = float(self.lat[1] - self.lat[0])
        self.lat_v = np.concatenate([[self.lat[0] - dlat / 2],
                                     self.lat + dlat / 2])
        self.nastep = nastep
        self.attrs = dict(attrs or {})
        self.requests = {}
        for name, req in spec.items():
            if name not in VARIABLES:
                raise KeyError(f'unknown output variable {name!r}; '
                               f'known: {sorted(VARIABLES)}')
            kind = req['kind']
            if kind not in ('mean', 'inst'):
                raise ValueError(f"{name}: kind must be 'mean' or 'inst'")
            self.requests[name] = dict(
                freq=req['freq'], kind=kind,
                steps=_freq_steps(req['freq'], nastep),
                acc=None, n=0, times=[], data=[])
        self._month_key = None
        self._month_stamp = None
        self._month_last = {}

    # ------------------------------------------------------------------
    def _sample(self, name, state, diags):
        src = VARIABLES[name][0]
        return getattr(state, name) if src == 'state' else diags[name]

    def _stamp(self, date, it):
        """cftime noleap time at the end of step ``it`` of ``date``."""
        import cftime
        import datetime
        if it == self.nastep:                     # midnight = next day 00h
            base = cftime.DatetimeNoLeap(date.yearofmodel, date.monthofyear,
                                         date.dayofmonth)
            return base + datetime.timedelta(days=1)
        hour = 24.0 * it / self.nastep
        return (cftime.DatetimeNoLeap(date.yearofmodel, date.monthofyear,
                                      date.dayofmonth)
                + datetime.timedelta(hours=hour))

    def record(self, state, diags, date, it):
        """Call once per atmospheric step (the driver does this).

        Sub-daily and daily intervals close when ``it`` is a multiple of
        the interval length (they always divide the day); monthly
        intervals close on the first step of a new month, using the
        stamp and (for 'inst') the sample from the last step of the old
        month.
        """
        month_key = (date.yearofmodel, date.monthofyear)
        if self._month_key is None:
            self._month_key = month_key
        if month_key != self._month_key:          # month rolled over: flush
            for name, r in self.requests.items():
                if r['steps'] is not None:
                    continue
                if r['kind'] == 'mean' and r['n']:
                    r['times'].append(self._month_stamp)
                    r['data'].append(r['acc'] / r['n'])
                    r['acc'], r['n'] = None, 0
                elif r['kind'] == 'inst' and name in self._month_last:
                    r['times'].append(self._month_stamp)
                    r['data'].append(self._month_last[name])
            self._month_key = month_key

        for name, r in self.requests.items():
            if r['kind'] == 'mean':
                f = self._sample(name, state, diags)
                if r['acc'] is None:
                    r['acc'] = np.zeros_like(f)
                r['acc'] += f
                r['n'] += 1
                if r['steps'] is not None and it % r['steps'] == 0:
                    r['times'].append(self._stamp(date, it))
                    r['data'].append(r['acc'] / r['n'])
                    r['acc'], r['n'] = None, 0
            elif r['steps'] is not None:          # sub-daily/daily inst
                if it % r['steps'] == 0:
                    r['times'].append(self._stamp(date, it))
                    r['data'].append(np.array(
                        self._sample(name, state, diags)))
            elif it == self.nastep:               # monthly inst candidate
                self._month_last[name] = np.array(
                    self._sample(name, state, diags))
        if it == self.nastep:
            self._month_stamp = self._stamp(date, it)

    # ------------------------------------------------------------------
    def to_datasets(self) -> dict:
        """One :class:`xarray.Dataset` per requested frequency.

        The ``nc_time_axis`` converter is registered when :mod:`qtcm1` is
        imported, so both ``.plot()`` and raw matplotlib calls work on the
        cftime noleap time axis out of the box.
        """
        import xarray as xr
        by_freq = {}
        for name, r in self.requests.items():
            by_freq.setdefault(r['freq'], []).append(name)
        out = {}
        for freq, names in by_freq.items():
            data_vars = {}
            coords = dict(lat=('lat', self.lat), lon=('lon', self.lon))
            time = None
            for name in names:
                r = self.requests[name]
                if not r['data']:
                    continue
                if time is None:
                    time = r['times']
                    coords['time'] = ('time', time)
                _, units, long_name, vgrid = VARIABLES[name]
                dims = ('time', 'lat_v', 'lon') if vgrid else \
                       ('time', 'lat', 'lon')
                if vgrid:
                    coords['lat_v'] = ('lat_v', self.lat_v)
                n = min(len(r['data']), len(time))
                data_vars[name] = (dims, np.stack(r['data'][:n]), dict(
                    units=units, long_name=long_name,
                    cell_methods=('time: mean' if r['kind'] == 'mean'
                                  else 'time: point')))
            if data_vars:
                out[freq] = xr.Dataset(data_vars, coords=coords,
                                       attrs=self.attrs)
        return out

    def to_netcdf(self, prefix: str):
        """Write ``<prefix>_<freq>.nc`` per frequency; returns the paths."""
        paths = []
        for freq, ds in self.to_datasets().items():
            fn = f'{prefix}_{freq}.nc'
            ds.to_netcdf(fn)
            paths.append(fn)
        return paths
