"""Radiation package: ports of ``radsw`` and ``radlw`` (clrad.F90).

Shortwave (``radsw``): simplified Fu-Liou (1993)-derived scheme. Daily-mean
insolation from analytic orbit formulas (declination via Zeng 1994; solar
"constant" 1370 W/m2 with Dickinson 1983 ellipticity correction); per cloud
type, column absorption and surface irradiance are quadratics in the
daily-mean cosine zenith angle with a first-order surface-albedo reflection
correction:

    F = SolarC * sum_k cld_k * (a_k cosZ2 + b_k cosZ1) * (1 + c_k * albedo)

Note the Fortran's own latitude for insolation is ``(j - (ny+1)/2) * dy/Re``
with *integer* (ny+1)/2 - shifted half a row from the T-grid latitude;
reproduced verbatim.

Longwave (``radlw``): the weakly nonlinear Chou-Neelin (1996) scheme -
fluxes linearized about a tropical reference in T1, q1, (Ts - Tsref) and
cloud-cover deviations, with cloud-type-dependent Green's-function
coefficients (NZ eq. 4.35). ``delta_co2`` implements the DELTA_CO2 option
(ppm field or scalar; reference 330 ppm).
"""

from __future__ import annotations

import numpy as np

from ..constants import REARTH, TSREF
from .clouds import CLDREF

_CLT = 4

# ----------------------------------------------------- radsw coefficients
_A1 = np.array([-0.7223241e-01, 0.2799124e-01, -0.3336635e-01,
                -0.3025872e-01, -0.5196485e-01])
_B1 = np.array([0.2511654e+00, 0.1754200e+00, 0.2181227e+00,
                0.2593093e+00, 0.2454770e+00])
_A2 = np.array([0.1867886e+00, 0.1466294e+00, 0.3125147e+00,
                0.2819863e+00, 0.3791754e+00])
_B2 = np.array([0.5848927e+00, 0.8151811e-01, 0.3584743e+00,
                0.1479825e+00, 0.2931454e+00])
_C1 = np.array([0.1662798e+00, 0.6341881e-01, 0.1743809e+00,
                0.6970352e-01, 0.6061328e+00])
_C2 = np.array([0.1160093e+00, 0.7485661e+00, 0.2741095e+00,
                0.5867739e+00, 0.1347113e+01])

_THETA0MAX = 23.447        # max solar declination [deg]
_DAYSPRING = 81.0          # vernal equinox [day of year]
_DAYPERIGEE = 3.0          # perihelion [day of year]
_ECCE = 0.034              # orbit ellipticity factor


def radsw(cld: np.ndarray, albedo: np.ndarray, dayofyear: int, grid,
          days_per_year: float = 365.0) -> dict:
    """Daily-mean shortwave fluxes [W m-2] (port of ``radsw``)."""
    ny, nx = albedo.shape
    pi = np.pi
    sindelta = (np.sin(np.deg2rad(_THETA0MAX))
                * np.sin(2.0 * pi * (dayofyear - _DAYSPRING) / days_per_year))
    delta = np.arcsin(sindelta)
    cosdelta = np.cos(delta)
    solarc = 1370.0 * (1.0 + _ECCE * np.cos(2.0 * pi * (dayofyear - _DAYPERIGEE)
                                            / days_per_year))
    dtheta = grid.dy / REARTH
    j = np.arange(1, ny + 1)
    lamda = (j - (ny + 1) // 2) * dtheta          # integer division, verbatim
    sinl = np.sin(lamda)
    cosl = np.sqrt(1.0 - sinl ** 2)
    cosH = np.clip(-sinl * sindelta / (cosl * cosdelta), -1.0, 1.0)
    H = np.arccos(cosH)
    sinH = np.sqrt(1.0 - cosH ** 2)
    sin2H = 2.0 * sinH * cosH
    cosZ1 = (sinl * sindelta * H + cosl * cosdelta * sinH) / pi
    cosZ2 = ((sinl * sindelta) ** 2 * H
             + 2.0 * sinl * cosl * sindelta * cosdelta * sinH
             + (cosl * cosdelta) ** 2 * (H / 2.0 - sin2H / 4.0)) / pi

    # per (type, row): Ca; per (type, row, col): albedo factor
    Ca1 = _A1[:, None] * cosZ2 + _B1[:, None] * cosZ1        # (5, ny)
    Ca2 = _A2[:, None] * cosZ2 + _B2[:, None] * cosZ1
    bs1 = 1.0 + _C1[:, None, None] * albedo                  # (5, ny, nx)
    bs2 = 1.0 + _C2[:, None, None] * albedo
    FSW = solarc * (cld * Ca1[:, :, None] * bs1).sum(axis=0)
    FSWds = solarc * (cld * Ca2[:, :, None] * bs2).sum(axis=0)
    S0 = np.broadcast_to((solarc * cosZ1)[:, None], (ny, nx)).copy()
    FSWus = FSWds * albedo
    FSWut = S0 - FSWds - FSW + FSWus
    return dict(FSW=FSW, FSWds=FSWds, FSWus=FSWus, FSWut=FSWut, S0=S0)


# ----------------------------------------------------- radlw coefficients
_EPS_RCT = np.array([-0.100751e+03, -0.616890e+02, -0.122923e+02,
                     -0.276311e+02])
_EPS_RCS = np.array([0.230918e+02, 0.843160e+01, 0.325455e+02,
                     0.205806e+02])
_EPS_RTT = np.array([0.133549e+01, 0.152938e+01, 0.145419e+01,
                     0.141225e+01, 0.140892e+01])
_EPS_RQT = np.array([-0.806434e+00, -0.644083e-01, -0.352098e+00,
                     -0.471530e+00, -0.421301e+00])
_EPS_RST = np.array([0.535400e+00, 0.109650e-02, 0.208200e+00,
                     0.900000e-03, 0.115160e+00])
_EPS_RTS = np.array([0.129186e+01, 0.174937e+01, 0.151779e+01,
                     0.183933e+01, 0.168531e+01])
_EPS_RQS = np.array([0.259074e+01, 0.656097e+00, 0.173063e+01,
                     0.118402e+00, 0.862426e+00])
_EPS_RSS = 0.628300e+01
_FLWDSREF, _FLWUSREF, _FLWUTREF = 443.4288, 475.3227, 240.3406
_EPS_RCO2T = np.array([-0.564960e+01, -0.326653e+01, -0.419050e+01,
                       -0.559220e+01, -0.522538e+01])
_EPS_RCO2S = np.array([0.188300e+00, 0.131395e+00, 0.165100e+00,
                       0.112100e+00, 0.139727e+00])
_CO2M = 330.0


def radlw(T1, q1, Ts, cld, *, delta_co2=None) -> dict:
    """Longwave fluxes [W m-2] (port of ``radlw``; CN96 weakly nonlinear).

    ``delta_co2``: CO2 concentration [ppm] (scalar or field) for the
    DELTA_CO2 option; None reproduces the standard build.
    """
    # cloud-cover-weighted coefficients (sums over type 0..4)
    eps_rTst = (_EPS_RTS[:, None, None] * cld).sum(axis=0)
    eps_rqst = (_EPS_RQS[:, None, None] * cld).sum(axis=0)
    eps_rTtt = (_EPS_RTT[:, None, None] * cld).sum(axis=0)
    eps_rqtt = (_EPS_RQT[:, None, None] * cld).sum(axis=0)
    eps_rstt = (_EPS_RST[:, None, None] * cld).sum(axis=0)

    FLWds = _FLWDSREF + eps_rTst * T1 + eps_rqst * q1
    FLWut = _FLWUTREF + eps_rTtt * T1 + eps_rqtt * q1 + eps_rstt * (Ts - TSREF)
    if delta_co2 is not None:
        co2 = np.asarray(delta_co2, dtype=np.float64)
        eps_rco2st = (_EPS_RCO2S[:, None, None] * cld).sum(axis=0)
        eps_rco2tt = (_EPS_RCO2T[:, None, None] * cld).sum(axis=0)
        FLWds = FLWds + eps_rco2st * (co2 - _CO2M) / 330.0
        FLWut = FLWut + eps_rco2tt * (co2 - _CO2M) / 330.0
    for n in range(1, _CLT + 1):           # cloud-deviation contributions
        FLWds = FLWds + _EPS_RCS[n - 1] * (cld[n] - CLDREF[n - 1])
        FLWut = FLWut + _EPS_RCT[n - 1] * (cld[n] - CLDREF[n - 1])
    FLWus = np.maximum(_FLWUSREF + _EPS_RSS * (Ts - TSREF), 0.0)
    FLW = FLWus - FLWds - FLWut
    return dict(FLWds=FLWds, FLWus=FLWus, FLWut=FLWut, FLW=FLW)
