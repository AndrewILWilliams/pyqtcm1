"""Simple-LAND version 1B: port of ``sland1`` (land.F90; Zeng & Neelin).

One soil layer per land point (STYPE 1 forest, 2 grass, 3 desert):

* Interception loss ``Evapi``: available energy times a stochastic-rainfall
  interception function (storm intensity/duration constants from ARME),
  capped at half the precipitation.
* Evapotranspiration: the potential ("swamp") evaporation from ``sflux`` is
  scaled by eta = ra/(rs+ra) with stomatal resistance rs = rsmin/wet^(1/4).
* Runoff: BATS surface runoff (Qc - Evapi)*wet^4 plus power-law subsurface
  drainage Rung0*wet^11.
* Prognostics: bucket soil moisture d(WD)/dt = (P - E - R)/L (clipped at 0)
  and ground temperature d(Ts)/dt = Fsnet/soilC with a 0.1-m water-like
  soil heat capacity.

Ocean points are untouched. The alternative ``bucket`` scheme is ported for
completeness.
"""

from __future__ import annotations

import numpy as np

from ..constants import HLATENT

#: per-type parameters, index 0..3 = ocean, forest, grass, desert (BATS/SIB2)
RSMIN = np.array([0.0, 150.0, 200.0, 200.0])   #: min stomatal resistance
ALBDVEG = np.array([0.07, 0.12, 0.19, 0.30])   #: vegetation albedo
Z0 = np.array([0.0024, 2.0, 0.1, 0.05])        #: roughness length [m]
XLA = np.array([0.0, 6.0, 3.0, 1.0])           #: leaf area index
WD0 = np.array([0.0, 500.0, 400.0, 300.0])     #: field capacity [kg/m2]

_WIMAX = 0.1          # max intercepted water per leaf area [mm]
_RINTS = 1.06e-3      # storm intensity [mm/s]
_TAU_R = 4320.0       # storm duration [s]
_RUNG0 = 4.0e-4       # subsurface runoff at saturation [mm/s]
_TINY = 1.0e-10
_SOILC = 4.18e3 * 1.0e3 * 0.1    # soil heat capacity [J K-1 m-2]


def sland1(Ts, WD, stype, Qc, Evap, FTs, FSWds, FSWus, FLWds, FLWus, CV,
           dt) -> dict:
    """Land-surface update (port of ``sland1``).

    All fields on T rows (ny, nx); ``Evap`` is the potential evaporation
    from ``sflux``. Returns updated Ts, WD, Evap and diagnostics Evapi,
    wet, Runs, Runf (ocean points pass through unchanged; diagnostic
    arrays are zero over ocean, as their Fortran module arrays start).
    """
    iS = stype.astype(int)
    land = iS != 0
    with np.errstate(divide='ignore', invalid='ignore'):
        Rsnet = FSWds - FSWus + FLWds - FLWus
        Evapi0 = np.maximum(_TINY, Rsnet - FTs)
        tau_0 = _WIMAX * XLA[iS] / Evapi0 * HLATENT
        Fitc = (_TAU_R + 0.8 * tau_0) * Qc / (HLATENT * _RINTS * _TAU_R)
        Evapi = np.minimum(Evapi0 * Fitc, 0.5 * Qc)

        wet = np.where(land, WD / np.where(land, WD0[iS], 1.0), 0.0)
        wra = np.sqrt(np.sqrt(wet)) / CV
        eta = wra / (RSMIN[iS] + wra)
        ET = eta * Evap
        Evap_land = ET + Evapi

        wet4 = wet ** 4
        Runs = (Qc - Evapi) * wet4
        Rung = HLATENT * _RUNG0 * (wet4 ** 2) * wet ** 3
        Runf = Runs + Rung

        FWnet = (Qc - Evap_land - Runf) / HLATENT
        WD_new = np.maximum(WD + dt * FWnet, 0.0)

        Fsnet = Rsnet - Evap_land - FTs
        Ts_new = Ts + dt * Fsnet / _SOILC

    zero = np.zeros_like(Ts)
    return dict(
        Ts=np.where(land, Ts_new, Ts),
        WD=np.where(land, WD_new, WD),
        Evap=np.where(land, Evap_land, Evap),
        Evapi=np.where(land, Evapi, zero),
        wet=np.where(land, wet, zero),
        Runs=np.where(land, Runs, zero),
        Runf=np.where(land, Runf, zero),
    )


def bucket(Ts, WD, stype, Qc, Evap, FTs, FSWds, FSWus, FLWds, FLWus,
           dt) -> dict:
    """Manabe-style bucket alternative (port of ``bucket``)."""
    land = stype.astype(int) != 0
    WD00 = 150.0
    wet = np.where(land, WD / WD00, 0.0)
    Runf = wet ** 4 * Qc
    Evap_land = wet * Evap
    WD_new = WD + dt * (Qc - Evap_land - Runf) / HLATENT
    Ts_new = Ts + dt * (FSWds - FSWus + FLWds - FLWus - Evap_land
                        - FTs) / _SOILC
    return dict(Ts=np.where(land, Ts_new, Ts),
                WD=np.where(land, WD_new, WD),
                Evap=np.where(land, Evap_land, Evap),
                wet=wet, Runf=np.where(land, Runf, 0.0))
