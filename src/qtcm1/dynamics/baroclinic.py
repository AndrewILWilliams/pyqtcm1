"""Baroclinic-mode update: port of ``barcl`` (qtcm.F90).

Advances u1, v1, T1, q1 one time step (NZ eqs. 5.1, 5.3-5.4) with the
Fortran's exact sequential ("Euler-backward in a sense") structure:

1. u1 explicit update from old v1/T1 (stress damping -gpTi*V1s/V1sqhat*taux,
   vertical diffusion eps_i1, Coriolis, mode-1 pressure gradient
   -R*d(T1)/dx, advection + diffusion tendencies), then polar-filtered;
2. v1 update *using the new, filtered u1* in its Coriolis term; wall rows
   set to zero; polar-filtered (Fortran filters rows 1..ny);
3. mode-1 divergence recomputed from the new winds;
4. T1 update: adiabatic cooling -Ms*div1 with the cloud-top-corrected gross
   dry stability Ms = Msr + Mqp*max(q1, q1m), plus (Qc+FSW+FLW+FTs)/Cpg,
   all divided by a1hat; polar-filtered;
5. q1 update: moisture convergence +Mq1*div1 with Mq1 = Mqr + Mqp*q1, plus
   (Evap-Qc)/Cpg, divided by b1hat; polar-filtered.
"""

from __future__ import annotations

import numpy as np

from ..constants import (A1HAT, B1HAT, B1S, CP, CPG, EPS_I1, GMQP, GMQR,
                         GMSR, GPTI, HLATENT, QREFS, RAIR, V1S, V1SQHAT)
from .advection import _roll_e, _roll_w, divergence_mode1

try:                                       # optional speedup; bit-identical
    from numba import njit
    _NUMBA = True
except ImportError:                        # pragma: no cover
    _NUMBA = False

    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

#: threshold surface moisture (19 g/kg) for the cloud-top Ms correction
_Q1M = (19.0 * 0.001 * HLATENT / CP - QREFS) / B1S


@njit(cache=True)
def _barcl_u_k(u1, v1, T1, taux, advu1, dfsu1, fu, rdxui, epstau,
               epsi1, dt):
    """Pre-filter u1 update (op order matches the vectorized form)."""
    ny, nx = u1.shape
    out = np.empty_like(u1)
    for p in range(ny):
        for i in range(nx):
            ie = (i + 1) % nx
            vatu = 0.25 * (v1[p + 1, i] + v1[p + 1, ie]
                           + v1[p, i] + v1[p, ie])
            rhs = (-epstau * taux[p, i] - epsi1 * u1[p, i]
                   + fu[p] * vatu
                   - rdxui[p] * (T1[p, ie] - T1[p, i])
                   + advu1[p, i] + dfsu1[p, i])
            out[p, i] = u1[p, i] + dt * rhs
    return out


@njit(cache=True)
def _barcl_v_k(u1n, v1, T1, tauy, advv1, dfsv1, fv, rdyi, epstau,
               epsi1, dt):
    """Pre-filter v1 update from the new filtered u1 (walls zero)."""
    nyp1, nx = v1.shape
    out = np.zeros_like(v1)
    for k in range(nyp1 - 2):              # v rows j = k+1
        for i in range(nx):
            iw = (i - 1) % nx
            uatv = 0.25 * (u1n[k, i] + u1n[k, iw]
                           + u1n[k + 1, i] + u1n[k + 1, iw])
            rhs = (-epstau * tauy[k, i] - epsi1 * v1[k + 1, i]
                   - fv[k + 1] * uatv
                   - rdyi * (T1[k + 1, i] - T1[k, i])
                   + advv1[k + 1, i] + dfsv1[k + 1, i])
            out[k + 1, i] = v1[k + 1, i] + dt * rhs
    return out


@njit(cache=True)
def _barcl_tq_k(u1n, v1n, T1, q1, Qc, FSW, FLW, FTs, Evap, advT1, dfsT1,
                advq1, dfsq1, dxui, dyui, cosv, q1m, dt):
    """div1 from the new winds + pre-filter T1/q1 updates."""
    ny, nx = T1.shape
    div1 = np.empty_like(T1)
    GMs1 = np.empty_like(T1)
    GMq1 = np.empty_like(T1)
    T1pre = np.empty_like(T1)
    q1pre = np.empty_like(T1)
    for p in range(ny):
        for i in range(nx):
            iw = (i - 1) % nx
            div1[p, i] = ((u1n[p, i] - u1n[p, iw]) * dxui[p]
                          + (v1n[p + 1, i] * cosv[p + 1]
                             - v1n[p, i] * cosv[p]) * dyui[p])
            gq = q1[p, i]
            gs = GMSR + GMQP * (gq if gq > q1m else q1m)
            GMs1[p, i] = gs
            GMq1[p, i] = GMQR + GMQP * gq
            rhs_T = ((-gs * div1[p, i]
                      + (Qc[p, i] + FSW[p, i] + FLW[p, i] + FTs[p, i])
                      / CPG) / A1HAT + advT1[p, i] + dfsT1[p, i])
            rhs_q = ((GMq1[p, i] * div1[p, i]
                      + (-Qc[p, i] + Evap[p, i]) / CPG) / B1HAT
                     + advq1[p, i] + dfsq1[p, i])
            T1pre[p, i] = T1[p, i] + dt * rhs_T
            q1pre[p, i] = q1[p, i] + dt * rhs_q
    return div1, GMs1, GMq1, T1pre, q1pre


def barcl(u1, v1, T1, q1, *, taux, tauy, advu1, advv1, advT1, advq1,
          dfsu1, dfsv1, dfsT1, dfsq1, Qc, FSW, FLW, FTs, Evap,
          grid, polar_filter, dt) -> dict:
    """One baroclinic step; returns new u1, v1, T1, q1 (+ div1, GMs1, GMq1)."""
    g = grid
    epstau = GPTI * V1S / V1SQHAT
    rdyi = RAIR * g.dyi
    C = np.ascontiguousarray

    if _NUMBA:
        u1pre = _barcl_u_k(C(u1), C(v1), C(T1), C(taux), C(advu1),
                           C(dfsu1), C(g.fu), C(RAIR * g.dxui),
                           float(epstau), float(EPS_I1), float(dt))
        u1n = polar_filter(u1pre)
        v1pre = _barcl_v_k(C(u1n), C(v1), C(T1), C(tauy), C(advv1),
                           C(dfsv1), C(g.fv), float(rdyi),
                           float(epstau), float(EPS_I1), float(dt))
        v1n = v1pre
        v1n[1:] = polar_filter(v1n[1:])        # Fortran filters rows 1..ny
        div1, GMs1, GMq1, T1pre, q1pre = _barcl_tq_k(
            C(u1n), C(v1n), C(T1), C(q1), C(Qc), C(FSW), C(FLW), C(FTs),
            C(Evap), C(advT1), C(dfsT1), C(advq1), C(dfsq1),
            C(g.dxui), C(g.dyui), C(g.cosv), float(_Q1M), float(dt))
        T1n = polar_filter(T1pre)
        q1n = polar_filter(q1pre)
        return dict(u1=u1n, v1=v1n, T1=T1n, q1=q1n, div1=div1,
                    GMs1=GMs1, GMq1=GMq1)

    rdxui = (RAIR * g.dxui)[:, None]
    # -- x momentum (u rows j=1..ny) ------------------------------------
    vatu = 0.25 * (v1[1:] + _roll_e(v1[1:]) + v1[:-1] + _roll_e(v1[:-1]))
    rhs_u = (-epstau * taux - EPS_I1 * u1 + g.fu[:, None] * vatu
             - rdxui * (_roll_e(T1) - T1) + advu1 + dfsu1)
    u1n = polar_filter(u1 + dt * rhs_u)

    # -- y momentum (v rows j=1..ny-1), using the new filtered u1 -------
    uatv = 0.25 * (u1n[:-1] + _roll_w(u1n[:-1]) + u1n[1:] + _roll_w(u1n[1:]))
    rhs_v = (-epstau * tauy[:-1] - EPS_I1 * v1[1:-1]
             - g.fv[1:-1, None] * uatv - rdyi * (T1[1:] - T1[:-1])
             + advv1[1:-1] + dfsv1[1:-1])
    v1n = np.zeros_like(v1)
    v1n[1:-1] = v1[1:-1] + dt * rhs_v          # wall rows stay zero
    v1n[1:] = polar_filter(v1n[1:])            # Fortran filters rows 1..ny

    # -- diagnostic divergence from the updated winds -------------------
    div1 = divergence_mode1(u1n, v1n, g)

    # -- temperature ----------------------------------------------------
    GMs1 = GMSR + GMQP * np.maximum(q1, _Q1M)
    rhs_T = ((-GMs1 * div1 + (Qc + FSW + FLW + FTs) / CPG) / A1HAT
             + advT1 + dfsT1)
    T1n = polar_filter(T1 + dt * rhs_T)

    # -- moisture -------------------------------------------------------
    GMq1 = GMQR + GMQP * q1
    rhs_q = ((GMq1 * div1 + (-Qc + Evap) / CPG) / B1HAT + advq1 + dfsq1)
    q1n = polar_filter(q1 + dt * rhs_q)

    return dict(u1=u1n, v1=v1n, T1=T1n, q1=q1n, div1=div1,
                GMs1=GMs1, GMq1=GMq1)
