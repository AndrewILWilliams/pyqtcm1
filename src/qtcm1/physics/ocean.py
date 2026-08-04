"""Mixed-layer ("slab") ocean: port of the ``MXL_OCEAN``/``BLEND_SST``
option (ocean.F90: ``mxstep``/``getQflux``/``blendsst``; cplmean.F90).

The slab integrates, once per coupling day and only over ocean points,

.. math::

   C_{mx}\,\frac{dT}{dt} = \overline{F}_{s,net} - Q_{flux},

with :math:`C_{mx} = 4.18\times10^6 \cdot D_{mx}` J K\ :sup:`-1` m\
:sup:`-2` (:math:`D_{mx}=50` m), :math:`\overline{F}_{s,net}` the
surface energy flux **averaged over the previous coupling day** (the
Fortran ``cplmean`` accumulation; on the very first day, before any
accumulation exists, the Fortran effectively uses zero fluxes -
reproduced here), and the Q-flux the ocean heat-transport correction

.. math::

   Q_{flux}(day) = f_{sn}(day) - d_{ts}(day)

diagnosed from a fixed-SST control run (``aveflux``): ``fsn`` is the
control's monthly-climatological net surface heat flux, read with the
usual mid-month linear interpolation (``bndry1``), and ``dts`` is
:math:`C_{mx}\,\partial T_s/\partial t` of the control SST, read
piecewise-constant per month with the month-01-centered switch at day
15 (``bndry2``). With this Q-flux the slab reproduces the control's
seasonal SST climatology by construction; perturbation experiments
(e.g. a greenhouse forcing) then let the ocean respond.

``BLEND_SST``: an optional 0/1 mask keeps prescribed (observed) SST in
masked regions and slab SST elsewhere.

Validation status: formula-level (unit tests + closure against the
control climatology). Bit-level golden validation against a Fortran
build compiled with ``-DMXL_OCEAN -DCPLMEAN`` is on the roadmap.
"""

from __future__ import annotations

import numpy as np

DMX = 50.0                     #: mixed-layer depth [m]
CMX = 4.18e6 * DMX             #: heat capacity [J K-1 m-2]

#: daily-mean fluxes the slab needs (accumulated cplmean-style)
CPL_FIELDS = ['FSWds', 'FSWus', 'FLWds', 'FLWus', 'Evap', 'FTs']


class QFlux:
    """Q-flux from the (12, ny, nx) fsn/dts climatologies.

    ``anchors`` is the julian mid-month anchor table of
    :class:`~qtcm1.io.bnddata.BoundaryData` (slots 0..13), used for the
    fsn interpolation exactly as bndry1 does.
    """

    def __init__(self, fsn: np.ndarray, dts: np.ndarray,
                 anchors: np.ndarray):
        self.fsn = np.asarray(fsn, dtype=np.float64)
        self.dts = np.asarray(dts, dtype=np.float64)
        self.anchors = np.asarray(anchors)

    @classmethod
    def from_netcdf(cls, path: str, anchors) -> 'QFlux':
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            return cls(np.array(ds['fsn'][:]), np.array(ds['dts'][:]),
                       anchors)

    def __call__(self, dayofyear: int, month: int, dayofmonth: int):
        """qfx = fsn(interp at day+0.5) - dts(bndry2 month rule)."""
        mid = self.anchors
        m1 = int(np.searchsorted(mid, dayofyear, side='right') - 1)
        t1, t2 = int(mid[m1]), int(mid[m1 + 1])
        mo1 = 12 if m1 == 0 else (1 if m1 == 13 else m1)
        mo2 = 1 if m1 + 1 == 13 else (12 if m1 + 1 == 0 else m1 + 1)
        frac = (dayofyear + 0.5 - t1) / (t2 - t1)
        f1, f2 = self.fsn[mo1 - 1], self.fsn[mo2 - 1]
        fsn = f1 + frac * (f2 - f1)
        month_read = month if dayofmonth < 15 else month + 1
        if month_read == 13:
            month_read = 1
        return fsn - self.dts[month_read - 1]


class MixedLayerOcean:
    """Slab-ocean state and daily update (``mxstep`` + ``cplmean``)."""

    def __init__(self, qflux: QFlux, stype, mask=None, depth: float = DMX,
                 landon: int = 1):
        self.qflux = qflux
        self.cmx = 4.18e6 * depth
        stype = np.asarray(stype)
        active = (stype == 0) if landon == 1 else np.ones_like(stype, bool)
        if mask is not None:                   # BLEND_SST: mask=1 keeps data
            active = active & (np.asarray(mask) == 0.0)
        self.active = active
        self.mask = None if mask is None else np.asarray(mask, np.float64)
        self.Tnow = None                       # set at init from data SST
        self._acc = None
        self._n = 0

    # -- cplmean ---------------------------------------------------------
    def accumulate(self, diags: dict):
        """Per-time-step flux accumulation (call every atmospheric step)."""
        if self._acc is None:
            self._acc = {k: np.zeros_like(diags[k]) for k in CPL_FIELDS}
        for k in CPL_FIELDS:
            self._acc[k] += diags[k]
        self._n += 1

    def _daily_means(self):
        if self._n == 0:                       # first day: Fortran uses 0
            return {k: 0.0 for k in CPL_FIELDS}
        return {k: v / self._n for k, v in self._acc.items()}

    # -- mxstep + blendsst ----------------------------------------------
    def step_day(self, dayofyear: int, month: int, dayofmonth: int,
                 data_sst=None, intcpl: int = 1) -> np.ndarray:
        """Advance the slab one coupling day; returns the SST to apply.

        Call at the START of each day, before the boundary update (the
        Fortran order: ocean -> getbnd -> atmosphere), passing the data
        SST when a blend mask is in use.
        """
        m = self._daily_means()
        qfx = self.qflux(dayofyear, month, dayofmonth)
        Rsnet = m['FSWds'] - m['FSWus'] + m['FLWds'] - m['FLWus']
        Fsnet = Rsnet - m['Evap'] - m['FTs']
        dT = float(intcpl) * 86400.0 * (Fsnet - qfx) / self.cmx
        self.Tnow = np.where(self.active, self.Tnow + dT, self.Tnow)
        if self.mask is not None and data_sst is not None:
            self.Tnow = (np.asarray(data_sst) * self.mask
                         + self.Tnow * (1.0 - self.mask))
        self._acc, self._n = None, 0           # reset for the new day
        return self.Tnow
