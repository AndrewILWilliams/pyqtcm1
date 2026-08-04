"""Surface fluxes and ABL surface winds: ports of ``Sflux`` and ``abl.F90``.

Surface winds: the default scheme (``SfcWindABL``) solves the steady mixed-
layer momentum balance at every T point with a 2-D Newton iteration,

    we/zi * (u_b - u_s) + f k x u_s - grad(phi_s) - (CD |V_s|/zi) u_s = 0,

warm-started from the *previous call's* solution (the Fortran keeps
``us, vs, VVs`` across calls and notes this makes QTCM not strictly
restartable). The wind speed entering the drag is
``|V_s| = sqrt(VVsmin^2 + us^2 + vs^2)``, so the persistent state is fully
recoverable from (us, vs). Iteration acceptance replicates the Fortran
loop exactly: a point must satisfy ``|f|+|g| < 1e-9`` at one of the first nine
residual checks, otherwise it reverts to its start-of-call winds (the
Fortran's ``iterate > 9`` branch also catches convergence at the 10th
check).

The top-of-ABL wind uses the mode-1 projection at z = ziml,
V1b = V1interpol(ziml) from the V1(z) table (-0.2282 at 500 m).

Fluxes (``Sflux``): bulk formulas NZ (5.16)-(5.17) with ``CV = CDN*|V_s|``;
evaporation uses saturation humidity at Ts through the model's hsat table;
ocean sensible heat flux is floored at -5 W/m2. Stress: taux/tauy from
CV averaged to the staggered points times the T-point surface wind
(verbatim, including the row-ny tauy being left unwritten).

The ``NO_ABL`` variant (v2.2 surface winds) is ``sfcwind_noabl``.
"""

from __future__ import annotations

import numpy as np

from ..constants import (CP, HLATENT, QREFS, RHOAIR, TREFS, V1Z_TABLE,
                         A1S, B1S, V1S)
from .convection import ConvectionTables, get_tables

try:                                       # optional speedup; bit-identical
    from numba import njit
    _NUMBA = True
except ImportError:                        # pragma: no cover
    _NUMBA = False

    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)


@njit(cache=True)
def _abl_newton_k(u1, v1, u0, v0, us_prev, vs_prev, dphisdx, dphisdy,
                  cdn, fu, v1b, wezi, zii, vvsminsq, n_iter, tol):
    """Per-point Newton iteration of SfcWindABL.

    Each point's iteration path is independent of the others, so the
    sequential per-point loop with early exit is bit-identical to the
    vectorized lockstep-with-masks form (same op order per element).
    """
    ny, nx = u1.shape
    us = np.empty_like(u1)
    vs = np.empty_like(u1)
    vvs = np.empty_like(u1)
    ub = np.empty_like(u1)
    vb = np.empty_like(u1)
    for p in range(ny):
        fuj = fu[p]
        for i in range(nx):
            iw = (i - 1) % nx
            ubi = 0.5 * ((u1[p, i] + u1[p, iw]) * v1b
                         + u0[p, i] + u0[p, iw])
            vbi = 0.5 * ((v1[p + 1, i] + v1[p, i]) * v1b
                         + v0[p + 1, i] + v0[p, i])
            dphx = 0.5 * (dphisdx[p, i] + dphisdx[p, iw])
            dphy = 0.5 * (dphisdy[p + 1, i] + dphisdy[p, i])
            cdzi = cdn[p, i] * zii
            u = us_prev[p, i]
            v = vs_prev[p, i]
            sv = np.sqrt(vvsminsq + u * u + v * v)
            usav, vsav, svsav = u, v, sv
            accepted = False
            for _k in range(1, n_iter):
                cdziwind = cdzi * sv
                cdziwindi = cdzi / sv
                f = ubi * wezi + fuj * v - dphx - u * (cdziwind + wezi)
                g = vbi * wezi - fuj * u - dphy - v * (cdziwind + wezi)
                if abs(f) + abs(g) < tol:
                    accepted = True
                    break
                dfdu = -wezi - cdziwind - u * u * cdziwindi
                dfdv = fuj - u * v * cdziwindi
                dgdv = -wezi - cdziwind - v * v * cdziwindi
                dgdu = -fuj - u * v * cdziwindi
                det = dfdu * dgdv - dfdv * dgdu
                if det != 0.0:
                    du = (dgdv * f - dfdv * g) / det
                    dv = (dfdu * g - dgdu * f) / det
                    u = u - du
                    v = v - dv
                    sv = np.sqrt(vvsminsq + u * u + v * v)
            if not accepted:
                u, v, sv = usav, vsav, svsav
            us[p, i] = u
            vs[p, i] = v
            vvs[p, i] = sv
            ub[p, i] = ubi
            vb[p, i] = vbi
    return us, vs, vvs, ub, vb


def v1interpol(zi: float) -> float:
    """Mode-1 velocity projection at height zi [m] (port of ``V1interpol``)."""
    z, v = V1Z_TABLE[:, 0], V1Z_TABLE[:, 1]
    if zi <= z[0] or zi > z[-1]:
        raise ValueError(f'ABL height out of V1(z) table range: {zi}')
    i = int(np.searchsorted(z, zi))
    we = (z[i] - zi) / (z[i] - z[i - 1])
    return float(v[i] * (1.0 - we) + v[i - 1] * we)


def _to_T_grid(u1, v1, u0, v0, coeff):
    """Mode-projected wind at T points: 0.5*(x_i + x_west) etc."""
    uT = 0.5 * ((u1 + np.roll(u1, 1, axis=1)) * coeff
                + u0 + np.roll(u0, 1, axis=1))
    vT = 0.5 * ((v1[1:] + v1[:-1]) * coeff + v0[1:] + v0[:-1])
    return uT, vT


def sfcwind_abl(u1, v1, u0, v0, us_prev, vs_prev, dphisdx, dphisdy,
                cdn, fu, *, weml, ziml, vvsmin, v1b=None, n_iter=10,
                tol=1e-9):
    """ABL surface winds (port of ``SfcWindABL``); returns us, vs, VVs, ub, vb."""
    if v1b is None:
        v1b = v1interpol(ziml)
    wezi = weml / ziml
    zii = 1.0 / ziml
    vvsminsq = vvsmin ** 2

    if _NUMBA:
        us, vs, vvs, ub, vb = _abl_newton_k(
            np.ascontiguousarray(u1), np.ascontiguousarray(v1),
            np.ascontiguousarray(u0), np.ascontiguousarray(v0),
            np.ascontiguousarray(us_prev), np.ascontiguousarray(vs_prev),
            np.ascontiguousarray(dphisdx), np.ascontiguousarray(dphisdy),
            np.ascontiguousarray(cdn), np.ascontiguousarray(fu),
            float(v1b), float(wezi), float(zii), float(vvsminsq),
            int(n_iter), float(tol))
        return dict(us=us, vs=vs, VVs=vvs, ub=ub, vb=vb)

    ub, vb = _to_T_grid(u1, v1, u0, v0, v1b)

    # geopotential gradients at T points
    dphisdxT = 0.5 * (dphisdx + np.roll(dphisdx, 1, axis=1))
    dphisdyT = 0.5 * (dphisdy[1:] + dphisdy[:-1])

    u = np.array(us_prev, dtype=np.float64, copy=True)
    v = np.array(vs_prev, dtype=np.float64, copy=True)
    sv = np.sqrt(vvsminsq + u * u + v * v)
    usav, vsav, svsav = u.copy(), v.copy(), sv.copy()

    cdzi = cdn * zii
    fuj = np.broadcast_to(fu[:, None], u.shape)
    accepted = np.zeros(u.shape, dtype=bool)
    for k in range(1, n_iter):                 # checks 1..9 can accept
        cdziwind = cdzi * sv
        cdziwindi = cdzi / sv
        f = ub * wezi + fuj * v - dphisdxT - u * (cdziwind + wezi)
        g = vb * wezi - fuj * u - dphisdyT - v * (cdziwind + wezi)
        newly = ~accepted & (np.abs(f) + np.abs(g) < tol)
        accepted |= newly
        active = ~accepted
        if not active.any():
            break
        dfdu = -wezi - cdziwind - u * u * cdziwindi
        dfdv = fuj - u * v * cdziwindi
        dgdv = -wezi - cdziwind - v * v * cdziwindi
        dgdu = -fuj - u * v * cdziwindi
        determ = dfdu * dgdv - dfdv * dgdu
        upd = active & (determ != 0.0)
        du = (dgdv * f - dfdv * g) / np.where(determ == 0.0, 1.0, determ)
        dv = (dfdu * g - dgdu * f) / np.where(determ == 0.0, 1.0, determ)
        u = np.where(upd, u - du, u)
        v = np.where(upd, v - dv, v)
        sv = np.where(upd, np.sqrt(vvsminsq + u * u + v * v), sv)

    # points never accepted revert to their start-of-call state
    u = np.where(accepted, u, usav)
    v = np.where(accepted, v, vsav)
    sv = np.where(accepted, sv, svsav)
    return dict(us=u, vs=v, VVs=sv, ub=ub, vb=vb)


def sfcwind_noabl(u1, v1, u0, v0, *, eta=0.6, vvse_min=5.0, vvs_min=4.0,
                  v1se=-0.17):
    """v2.2 surface winds without ABL (port of ``SfcWindNoABL``)."""
    us, vs = _to_T_grid(u1, v1, u0, v0, V1S)
    uTe, vTe = _to_T_grid(u1, v1, u0, v0, v1se)
    VVsE = np.sqrt(vvse_min ** 2 + eta ** 2 * (uTe ** 2 + vTe ** 2))
    VVs = np.sqrt(us ** 2 + vs ** 2 + vvs_min ** 2)
    return dict(us=us, vs=vs, VVs=VVs, VVsE=VVsE)


def sflux(T1, q1, Ts, stype, cdn, wind,
          tables: ConvectionTables | None = None) -> dict:
    """Bulk surface fluxes (port of the flux section of ``Sflux``).

    ``wind`` is the dict from :func:`sfcwind_abl` (or ``sfcwind_noabl``;
    for NO_ABL pass its VVsE via ``wind['VVsE']`` and stress uses VVs).
    ``tables`` selects the hsat table build for evaporation (float64 if
    omitted). Returns CV, taux, tauy, Evap, FTs (tauy row ny-1 is unused).
    """
    tables = tables or get_tables()
    us, vs, VVs = wind['us'], wind['vs'], wind['VVs']
    CV = cdn * VVs

    taux = RHOAIR * 0.5 * (CV + np.roll(CV, -1, axis=1)) * us
    tauy = np.zeros_like(CV)
    tauy[:-1] = RHOAIR * 0.5 * (CV[:-1] + CV[1:]) * vs[:-1]

    CVE = cdn * wind['VVsE'] if 'VVsE' in wind else CV
    rhoCp = RHOAIR * CP
    Evap = rhoCp * CVE * (tables.hsat(Ts) * (HLATENT / CP)
                          - QREFS - B1S * q1)
    FTs = rhoCp * CVE * (Ts - TREFS - A1S * T1)
    FTs = np.where(stype == 0, np.maximum(FTs, -5.0), FTs)
    return dict(CV=CVE, taux=taux, tauy=tauy, Evap=Evap, FTs=FTs)
