"""NetCDF boundary-data reader with QTCM1 mid-month time interpolation.

Replaces the Fortran ASCII readers (``bndry.F90``, ``ocean.F90:sstin`` /
``readsst`` / ``getSST``) with readers over the netCDF registry produced by
``tools/convert_bnddata.py``. The interpolation reproduces the Fortran
scheme: monthly snapshots valid at mid-month anchors (``midmonth`` table in
calendar.F90, including the previous-December ``-16`` and next-January
``380`` wrap anchors), linearly interpolated with ``TimeInterp``.

The interpolation instant is the coupling-interval midpoint,
``thistime = dayofmodel + ndays/2`` (readsst; bndry1 hardwires ndays=1),
so daily forcing is valid at ``dayofyear + 0.5``. With the registry stored
at float64 (ASCII decimals parse identically in Python and Fortran) the
daily fields are bitwise-equal to the double-precision oracle's
(tests/test_boundary.py pins 30 days including a mid-month bracket
advance). Note two Fortran driver quirks live OUTSIDE this module:
``getbnd`` skips the SST update on its very first call (cold start), and
the boundary update must run exactly ONCE per model day (a same-day
second call in the Fortran corrupts the brackets on mid-month days -
bndry1's guard restores time1/time2 but var_next has already advanced).
"""

from __future__ import annotations

import os

import netCDF4
import numpy as np

from ..calendar import ModelCalendar, time_interp


class BoundaryData:
    """Boundary datasets on the model grid, with mid-month interpolation.

    Parameters
    ----------
    path:
        Directory containing the converted netCDF registry
        (``sst_reynolds_clim.nc``, ``sst_reynolds_1949_2001.nc``,
        ``albedo_darnell.nc``, ``surface.nc``, ...).
    calendar:
        The run's :class:`~qtcm1.calendar.ModelCalendar` (anchor table and
        month lengths).
    """

    def __init__(self, path: str, calendar: ModelCalendar | None = None,
                 surface=None, albedo_mode: str = 'auto',
                 sst_mode: str | None = None):
        self.path = path
        self.calendar = calendar or ModelCalendar()
        self.albedo_mode = albedo_mode
        self.sst_mode = sst_mode
        # Interpolation anchors are julian(yyyy mm 15), i.e. cumulative
        # month lengths + 15 -- NOT calendar.F90's ``midmonth`` table.
        # The Fortran carries both conventions and they disagree for
        # February (midmonth says 45, julian(0215) = 31+15 = 46); the
        # boundary readers (readsst/bndry1) use julian(), so we must too.
        cum = np.concatenate([[0], np.cumsum(self.calendar.monlen)])
        mid = cum[:12] + 15                       # Jan..Dec mid-month julian
        self._anchors = np.concatenate([[mid[11] - cum[12]], mid,
                                        [mid[0] + cum[12]]])   # 0..13

        def _load(fname, varname):
            with netCDF4.Dataset(os.path.join(path, fname)) as ds:
                return np.array(ds[varname][:])

        self.sst_clim = _load('sst_reynolds_clim.nc', 'sst')      # (12,ny,nx)
        self.albedo_clim = _load('albedo_darnell.nc', 'albedo')   # (12,ny,nx)
        self.albedo_annual = _load('albedo_darnell.nc', 'albedo_annual')
        self.stype = _load('surface.nc', 'stype').astype(np.int16)
        self.top = _load('surface.nc', 'top')
        self.lat = _load('surface.nc', 'lat')
        self.lon = _load('surface.nc', 'lon')

        # optional custom surface (qtcm1.surface): replaces stype/top and
        # switches albedo handling for the changed points
        self.stype_ref = self.stype
        self.surface_sha256 = None
        self._changed = None
        self._alb_static = None
        if surface is not None:
            from .. import surface as _surf
            stype_new, top_new = _surf.coerce(surface)
            _surf.validate(stype_new, top_new, self.stype.shape)
            self.surface_sha256 = _surf.sha256(stype_new, top_new)
            self._changed = stype_new != self.stype_ref
            new_ocean = (stype_new == 0) & (self.stype_ref != 0)
            if new_ocean.any() and sst_mode != 'zonal':
                import warnings
                warnings.warn(
                    f'{int(new_ocean.sum())} ocean points created where the '
                    'registry has land: prescribed SST there is the '
                    "dataset's under-land fill, not observations "
                    "(sst_mode='zonal' avoids this)",
                    stacklevel=3)
            self.stype = stype_new
            self.top = top_new
            # static per-type albedo, diagnosed from the packaged annual
            # albedo over the *original* mask (used at changed points)
            bytype = np.array([
                float(self.albedo_annual[self.stype_ref == t].mean())
                for t in range(4)])
            self._alb_static = bytype[self.stype]

        # dated SST series indexed by (year, month)
        fn = os.path.join(path, 'sst_reynolds_1949_2001.nc')
        if os.path.exists(fn):
            with netCDF4.Dataset(fn) as ds:
                t = np.array(ds['time'][:])            # days since 1949-01-01
                self.sst_dated = np.array(ds['sst'][:])
            months = np.rint((t % 365) + 0.5)          # decode via table below
            # robust decode: month from anchor day-of-year
            mid = self.calendar.midmonth[1:13]
            doy = (t % 365).astype(int)
            month = np.array([int(np.argmin(np.abs(mid - d))) + 1 for d in doy])
            year = (t // 365).astype(int) + 1949
            self._dated_index = {(int(y), int(m)): i
                                 for i, (y, m) in enumerate(zip(year, month))}
        else:                                          # registry without obs SST
            self.sst_dated = None
            self._dated_index = {}

        fn = os.path.join(path, 'sst_perpetual.nc')
        self.sst_perpetual = None
        if os.path.exists(fn):
            with netCDF4.Dataset(fn) as ds:
                self.sst_perpetual = np.array(ds['sst'][:])

    # ------------------------------------------------------------------
    def bracket(self, dayofyear: int) -> tuple[int, int, int, int]:
        """Mid-month julian anchors surrounding ``dayofyear``.

        Returns ``(m1, m2, t1, t2)`` where ``m1``/``m2`` are anchor month
        slots 0..13 (0 = December of the previous year, 13 = January of
        the next) and ``t1``/``t2`` their julian anchor times (t1 may be
        negative, t2 may exceed the year length). ``side='right'``
        reproduces the Fortran advance rule ``dateofmodel >= date2``: on a
        mid-month day the bracket starting at that day is active.
        """
        mid = self._anchors                            # index 0..13
        m1 = int(np.searchsorted(mid, dayofyear, side='right') - 1)
        return m1, m1 + 1, int(mid[m1]), int(mid[m1 + 1])

    @staticmethod
    def _slot_to_month(m: int) -> tuple[int, int]:
        """Anchor slot 0..13 -> (calendar month 1..12, year offset)."""
        if m == 0:
            return 12, -1
        if m == 13:
            return 1, +1
        return m, 0

    # ------------------------------------------------------------------
    def _interp_monthly(self, clim: np.ndarray, dayofyear: int,
                        half_interval: float = 0.5) -> np.ndarray:
        m1, m2, t1, t2 = self.bracket(dayofyear)
        mo1, _ = self._slot_to_month(m1)
        mo2, _ = self._slot_to_month(m2)
        return time_interp(clim[mo1 - 1], clim[mo2 - 1], t1, t2,
                           dayofyear + half_interval)

    def sst(self, year: int, dayofyear: int, mode: str = 'seasonal',
            interval_days: int = 1) -> np.ndarray:
        """SST [K] for the day, valid at ``dayofyear + interval_days/2``.

        ``seasonal``: climatological monthly files; ``real_time``: dated
        observed series (adjacent-year wrap at anchors); ``perpetual``:
        the fixed 00000000.sst field; ``zonal``: the seasonal
        climatology zonally averaged over the *registry's* ocean points
        per latitude (under-land fill values never enter; a latitude
        with no registry ocean falls back to the full-longitude mean) —
        a zonally symmetric ocean with the observed meridional and
        seasonal structure, for idealized-geography runs.
        ``interval_days`` is the ocean coupling interval (``ndays`` of
        readsst; standard runs use 1).
        """
        half = interval_days / 2.0
        if mode == 'zonal':
            field = self._interp_monthly(self.sst_clim, dayofyear, half)
            ocean = self.stype_ref == 0
            out = np.empty_like(field)
            for j in range(field.shape[0]):
                sel = ocean[j]
                out[j] = field[j, sel].mean() if sel.any() else field[j].mean()
            return out
        if mode == 'perpetual':
            if self.sst_perpetual is None:
                raise FileNotFoundError('sst_perpetual.nc not in registry')
            return self.sst_perpetual.copy()
        if mode == 'seasonal':
            return self._interp_monthly(self.sst_clim, dayofyear, half)
        if mode == 'real_time':
            if self.sst_dated is None:
                raise FileNotFoundError(
                    'sst_reynolds_1949_2001.nc not in registry')
            m1, m2, t1, t2 = self.bracket(dayofyear)
            mo1, dy1 = self._slot_to_month(m1)
            mo2, dy2 = self._slot_to_month(m2)
            f1 = self.sst_dated[self._dated_index[(year + dy1, mo1)]]
            f2 = self.sst_dated[self._dated_index[(year + dy2, mo2)]]
            return time_interp(f1, f2, t1, t2, dayofyear + half)
        raise ValueError(f'unknown SSTmode: {mode!r}')

    def albedo(self, dayofyear: int) -> np.ndarray:
        """Surface albedo [-] for the day (getbnd/bndry1; ndays=1).

        With a custom surface, points whose ``stype`` changed use the
        static per-type albedo (mode ``'auto'``, default); mode
        ``'by_stype'`` uses it everywhere; ``'darnell'`` keeps the
        Earth-locked climatology everywhere (usually wrong over moved
        coastlines -- explicit opt-in only).
        """
        clim = self._interp_monthly(self.albedo_clim, dayofyear, 0.5)
        if self._alb_static is None or self.albedo_mode == 'darnell':
            return clim
        if self.albedo_mode == 'by_stype':
            return self._alb_static.copy()
        return np.where(self._changed, self._alb_static, clim)
