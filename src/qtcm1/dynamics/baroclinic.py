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

#: threshold surface moisture (19 g/kg) for the cloud-top Ms correction
_Q1M = (19.0 * 0.001 * HLATENT / CP - QREFS) / B1S


def barcl(u1, v1, T1, q1, *, taux, tauy, advu1, advv1, advT1, advq1,
          dfsu1, dfsv1, dfsT1, dfsq1, Qc, FSW, FLW, FTs, Evap,
          grid, polar_filter, dt) -> dict:
    """One baroclinic step; returns new u1, v1, T1, q1 (+ div1, GMs1, GMq1)."""
    g = grid
    epstau = GPTI * V1S / V1SQHAT
    rdxui = (RAIR * g.dxui)[:, None]
    rdyi = RAIR * g.dyi

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
