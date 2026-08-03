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

import numpy as np

from .calendar import ModelCalendar
from .io.bnddata import BoundaryData
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

    def __init__(self, data_path: str, year0: int = 1, month0: int = 1,
                 day0: int = 1, sst_mode: str = 'seasonal', params=None,
                 init_dtype=np.float64):
        self.bd = BoundaryData(data_path)
        self.calendar = ModelCalendar(year0=year0, month0=month0, day0=day0)
        self.sst_mode = sst_mode
        stype = self.bd.stype.astype(np.float64)
        self.model = Model(stype, bndinit_cdn(stype), params=params,
                           init_dtype=init_dtype)
        self.state = self.model.cold_start()      # carries gradphis_virgin
        self.dayofmodel = 0
        self._getbnd_virgin = True
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
        if self._getbnd_virgin:                  # getbnd ncall==0: skip SST
            self._getbnd_virgin = False
        else:
            sst = self.bd.sst(date.yearofmodel, doy, self.sst_mode)
            self.state, _ = self.model.apply_boundary(self.state, sst, alb)
        nastep = int(round(86400.0 / self.model.params['dt']))
        for it in range(1, nastep + 1):
            self.state, diags = self.model.step(self.state, alb, doy, it)
            self._accumulate(self.state, diags,
                             (date.yearofmodel, date.monthofyear))
        return date

    def run_years(self, nyears: int, progress=None):
        ndays = nyears * self.calendar.days_per_year
        for _ in range(ndays):
            date = self.advance_day()
            if progress and self.dayofmodel % 365 == 0:
                progress(self.dayofmodel, date)
        self._flush_month()

    # ------------------------------------------------------------------
    def save_monthly(self, path: str):
        out = {}
        for i, (year, month, mean) in enumerate(self.monthly):
            for k, v in mean.items():
                out[f'm{i:04d}/{k}'] = v.astype(np.float32)
        out['years'] = np.array([y for y, _, _ in self.monthly])
        out['months'] = np.array([m for _, m, _ in self.monthly])
        np.savez_compressed(path, **out)
