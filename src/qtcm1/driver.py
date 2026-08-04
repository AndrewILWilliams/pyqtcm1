"""Free-running drivers: cold start and control integrations.

Replicates the standard Fortran run sequence exactly:

* init (``qtcminit``): parinit constants, ``bndinit`` (STYPE, annual
  albedo, CDN from surface type), ``varinit`` cold start, TimeManager(1).
  The init-time ``physics1`` call is a no-op for a cold start (all winds,
  gradients and the ABL warm start are zero, and physics1 writes nothing
  else persistent), so it is not repeated here. The ABL first-call CDN
  mutation (land doubled, ocean pinned to 0.0011) is applied to the
  bndinit CDN directly.
* per day (``driver``/``atm_oc_step``): TimeManager -> ocean/getbnd ->
  72 atmospheric steps. The boundary update runs exactly ONCE per day
  (getbnd is not idempotent on mid-month bracket days), and the very
  first getbnd of a cold start skips the SST application (the Fortran
  ``ncall`` guard: day 1 integrates with the varinit 295-K ocean).
* the first barotropic group skips ``gradphis`` (Fortran first-call
  early return), carried as ``ModelState.gradphis_virgin`` (set by
  ``cold_start``; warm starts from captured oracle states leave it off).

Monthly means are accumulated every time step (``varmean`` semantics),
not from daily snapshots.
"""

from __future__ import annotations

import os

import numpy as np

from .calendar import ModelCalendar
from .config import RunConfig, provenance
from .io.bnddata import BoundaryData
from .io.output import DEFAULT_OUTPUT, OutputManager
from .io.restart import load_restart, save_restart
from .model import Model
from .physics.land import Z0

HPBL = 2000.0          #: PBL depth [m] used by the bndinit CDN formula
VONKAR = 0.4           #: Von Karman constant

#: state fields and step diagnostics accumulated into monthly means
STATE_FIELDS = ['u1', 'v1', 'T1', 'q1', 'u0', 'v0', 'Ts', 'WD']
DIAG_FIELDS = ['Qc', 'Evap', 'FTs', 'OLR', 'cl1', 'S0', 'taux']


def bndinit_cdn(stype) -> np.ndarray:
    """Drag coefficient from surface type (``bndinit`` + ABL first call).

    CDN = 1/(ln(0.025*hPBL/Z0)/k + 8.4)^2 (BATS/CCM2 form), then the ABL
    first-call mutation: land (STYPE > 0.5) doubled to compensate mountain
    drag, ocean pinned to 0.0011.
    """
    stype = np.asarray(stype)
    z0 = Z0[stype.astype(int)]
    cdn = 1.0 / (np.log(0.025 * HPBL / z0) / VONKAR + 8.4) ** 2
    return np.where(stype > 0.5, cdn * 2.0, 0.0011)


class ControlRun:
    """Cold-started free-running control integration."""

    def __init__(self, data_path: str | None = None, year0: int = 1,
                 month0: int = 1, day0: int = 1, sst_mode: str = 'seasonal',
                 params=None, init_dtype=np.float64,
                 config: RunConfig | None = None, surface=None):
        if config is None:
            config = RunConfig(
                data_path=data_path,
                build='f32' if np.dtype(init_dtype) == np.float32 else 'f64',
                year0=year0, month0=month0, day0=day0, sst_mode=sst_mode,
                params=dict(params or {}))
        self.config = config
        data_path = config.data_path
        year0, month0, day0 = config.year0, config.month0, config.day0
        sst_mode, params = config.sst_mode, (config.params or None)
        init_dtype = config.init_dtype
        # custom surface: in-memory argument wins over the config path
        surface = surface if surface is not None else config.surface
        if surface is not None and sst_mode in ('mixed_layer', 'blend'):
            raise ValueError(
                'slab-ocean modes need a Q-flux diagnosed for the custom '
                'geography: run a fixed-SST control on the new surface, '
                'diagnose with tools/make_qflux.py, and point data_path at '
                'a registry containing that qflux.nc')
        self.bd = BoundaryData(data_path, surface=surface,
                               albedo_mode=config.albedo_mode,
                               sst_mode=sst_mode)
        self.calendar = ModelCalendar(year0=year0, month0=month0, day0=day0)
        self.sst_mode = sst_mode
        self.ocean = None
        if sst_mode in ('mixed_layer', 'blend'):
            from .physics.ocean import MixedLayerOcean, QFlux
            qf = QFlux.from_netcdf(os.path.join(data_path, 'qflux.nc'),
                                   self.bd._anchors)
            mask = None
            if sst_mode == 'blend':
                import netCDF4
                with netCDF4.Dataset(os.path.join(data_path,
                                                  'masks.nc')) as dsm:
                    mask = np.array(dsm['ensopac'][:])
            self.ocean = MixedLayerOcean(qf, self.bd.stype, mask=mask)
            # OceanInit: Tnow starts as the bracket-left monthly SST field
            doy0 = self.calendar.timemanager(1).dayofyear
            m1 = int(np.searchsorted(self.bd._anchors, doy0, 'right') - 1)
            mo1 = 12 if m1 == 0 else (1 if m1 == 13 else m1)
            self.ocean.Tnow = self.bd.sst_clim[mo1 - 1].copy()
        stype = self.bd.stype.astype(np.float64)
        # TOPO: bndinit threshold (top below 0.1, i.e. 1 km, is zeroed)
        topo_top = None
        if config.topo:
            topo_top = np.where(self.bd.top < 0.1, 0.0, self.bd.top)
        self.model = Model(stype, bndinit_cdn(stype), params=params,
                           init_dtype=init_dtype, topo_top=topo_top)
        self.state = self.model.cold_start()      # carries gradphis_virgin
        self.dayofmodel = 0
        self._getbnd_virgin = True
        # configurable output (xarray): see qtcm1.io.output
        import json
        prov = provenance(config)
        if self.bd.surface_sha256 is not None:
            prov['surface_sha256'] = self.bd.surface_sha256
        attrs = {k: (v if isinstance(v, str) else json.dumps(v))
                 for k, v in prov.items()}
        nastep = int(round(86400.0 / self.model.params['dt']))
        self.output = OutputManager(config.output or DEFAULT_OUTPUT,
                                    self.bd.lat, self.bd.lon, nastep,
                                    attrs=attrs)
        # monthly accumulators
        self._acc = None
        self._acc_n = 0
        self._acc_key = None                      # (year, month)
        self.monthly = []                         # list of (year, month, dict)

    # ------------------------------------------------------------------
    def _flush_month(self):
        if self._acc is not None and self._acc_n:
            mean = {k: v / self._acc_n for k, v in self._acc.items()}
            self.monthly.append((*self._acc_key, mean))
        self._acc, self._acc_n = None, 0

    def _accumulate(self, s, diags, key):
        if key != self._acc_key:
            self._flush_month()
            self._acc_key = key
        if self._acc is None:                     # native grids throughout
            self._acc = {k: np.zeros_like(getattr(s, k))
                         for k in STATE_FIELDS}
            self._acc.update({k: np.zeros_like(
                np.asarray(diags[k], dtype=np.float64))
                for k in DIAG_FIELDS})
        for k in STATE_FIELDS:
            self._acc[k] += getattr(s, k)
        for k in DIAG_FIELDS:
            self._acc[k] += diags[k]
        self._acc_n += 1

    # ------------------------------------------------------------------
    def advance_day(self):
        """One coupling day: calendar, boundary (once), 72 steps."""
        self.dayofmodel += 1
        date = self.calendar.timemanager(self.dayofmodel)
        doy = date.dayofyear
        alb = self.bd.albedo(doy)
        if self.ocean is not None:               # slab: ocean before getbnd
            data_sst = (self.bd.sst(date.yearofmodel, doy)
                        if self.ocean.mask is not None else None)
            sst = self.ocean.step_day(doy, date.monthofyear,
                                      date.dayofmonth, data_sst)
        if self._getbnd_virgin:                  # getbnd ncall==0: skip SST
            self._getbnd_virgin = False
        else:
            if self.ocean is None:
                sst = self.bd.sst(date.yearofmodel, doy, self.sst_mode)
            self.state, _ = self.model.apply_boundary(self.state, sst, alb)
        nastep = int(round(86400.0 / self.model.params['dt']))
        for it in range(1, nastep + 1):
            self.state, diags = self.model.step(self.state, alb, doy, it)
            if self.ocean is not None:           # cplmean accumulation
                self.ocean.accumulate(diags)
            self.output.record(self.state, diags, date, it)
            self._accumulate(self.state, diags,
                             (date.yearofmodel, date.monthofyear))
        return date

    # ------------------------------------------------------------------
    def to_datasets(self) -> dict:
        """Requested output so far as {frequency: xarray.Dataset}."""
        return self.output.to_datasets()

    def save_output(self, prefix: str) -> list:
        """Write one netCDF per requested frequency; returns the paths."""
        return self.output.to_netcdf(prefix)

    def run_years(self, nyears: int, progress=None):
        ndays = nyears * self.calendar.days_per_year
        for _ in range(ndays):
            date = self.advance_day()
            if progress and self.dayofmodel % 365 == 0:
                progress(self.dayofmodel, date)
        self._flush_month()

    # ------------------------------------------------------------------
    def save_monthly(self, path: str):
        import json
        out = {}
        for i, (year, month, mean) in enumerate(self.monthly):
            for k, v in mean.items():
                out[f'm{i:04d}/{k}'] = v.astype(np.float32)
        out['years'] = np.array([y for y, _, _ in self.monthly])
        out['months'] = np.array([m for _, m, _ in self.monthly])
        out['provenance'] = np.frombuffer(
            json.dumps(provenance(self.config)).encode(), dtype=np.uint8)
        np.savez_compressed(path, **out)

    # ------------------------------------------------------------------
    def save_restart(self, path: str):
        """Bit-exact restart: complete ModelState + driver position.

        Monthly-mean accumulators are NOT included -- a resumed run
        restarts its diagnostics accumulation fresh; the model trajectory
        itself continues bit-identically (tests/test_restart.py).
        """
        extra = {}
        if self.ocean is not None:               # slab state + pending means
            extra['ocean_Tnow'] = self.ocean.Tnow
            extra['ocean_n'] = np.int64(self.ocean._n)
            if self.ocean._acc is not None:
                for k, v in self.ocean._acc.items():
                    extra[f'ocean_acc_{k}'] = v
        save_restart(path, self.state, dayofmodel=self.dayofmodel,
                     getbnd_virgin=self._getbnd_virgin,
                     header=provenance(self.config), extra=extra)

    @classmethod
    def from_restart(cls, path: str,
                     config: RunConfig | None = None) -> 'ControlRun':
        """Resume a run from a restart file (config from its header if
        not supplied)."""
        state, dayofmodel, virgin, header, extra = load_restart(path)
        cfg = config or RunConfig.from_dict(header['config'])
        run = cls(config=cfg)
        run.state = state
        run.dayofmodel = dayofmodel
        run._getbnd_virgin = virgin
        if run.ocean is not None and 'ocean_Tnow' in extra:
            run.ocean.Tnow = extra['ocean_Tnow']
            run.ocean._n = int(extra['ocean_n'])
            acc = {k[len('ocean_acc_'):]: v for k, v in extra.items()
                   if k.startswith('ocean_acc_')}
            run.ocean._acc = acc or None
        return run
