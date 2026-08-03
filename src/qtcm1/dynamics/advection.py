"""Momentum and thermodynamic advection: ports of ``advctuv``, ``advctTq``.

Row conventions as in :mod:`qtcm1.dynamics.diffusion`: u/T-grid fields are
``(ny, nx)`` (Fortran rows 1..ny), v-grid fields ``(ny+1, nx)`` (rows 0..ny).

The nonlinear advection projects products of the two velocity modes onto
each mode with the precomputed inner products ``Vijk``/``Vwijk``
(NZ eqs. 4.7-4.9), plus the vertical ("omega") transport terms proportional
to the mode-1 divergence. Former compile-time options are keyword flags:
``no_wadv`` (drop w-advection from the tendencies), ``v1v0advh`` /
``v1v0advvh`` (halve the V1-squared terms in the barotropic x/y equations),
and ``div0`` may be supplied for the TOPO option (defaults to zero).
"""

from __future__ import annotations

import numpy as np

from ..constants import DQIJKT, DTIJKT, VIJKT, VWIJKT


def _roll_e(f):
    return np.roll(f, -1, axis=1)      # Fortran ip1


def _roll_w(f):
    return np.roll(f, 1, axis=1)       # Fortran im1


def _ddy_bounded(f, dyi):
    """d(f)/dy on the same rows as f: centered inside, one-sided at walls.

    Implements the Fortran ``ajp1/aj/ajm1`` coefficient triple used for
    u-grid and T-grid meridional derivatives.
    """
    out = np.empty_like(f)
    out[1:-1] = 0.5 * (f[2:] - f[:-2]) * dyi
    out[0] = (f[1] - f[0]) * dyi
    out[-1] = (f[-1] - f[-2]) * dyi
    return out


def divergence_mode1(u1, v1, grid):
    """Mode-1 divergence on T rows (``div1`` as computed in advctuv)."""
    g = grid
    return ((u1 - _roll_w(u1)) * g.dxui[:, None]
            + (v1[1:] * g.cosv[1:, None] - v1[:-1] * g.cosv[:-1, None])
            * g.dyui[:, None])


def advctuv(u1, v1, u0, v0, grid, *, div0=None,
            no_wadv: bool = False, v1v0advh: bool = False,
            v1v0advvh: bool = False):
    """Momentum advection tendencies (port of ``advctuv``).

    Returns dict with advu0, advu1 (ny, nx); advv0, advv1 (ny+1, nx; wall
    rows 0 and ny left zero as in the Fortran, which never writes them);
    advwu0, advwu1 (ny, nx); and div1 (ny, nx), which the Fortran computes
    here and reuses downstream.
    """
    g = grid
    dyi, ny = g.dyi, g.ny
    div1 = divergence_mode1(u1, v1, grid)
    d0 = np.zeros_like(div1) if div0 is None else div0

    # ---------------- x-momentum (u rows j=1..ny) ----------------------
    dxih = (0.5 * g.dxui)[:, None]
    du1x = (_roll_e(u1) - _roll_w(u1)) * dxih
    du0x = (_roll_e(u0) - _roll_w(u0)) * dxih
    du1y = _ddy_bounded(u1, dyi)
    du0y = _ddy_bounded(u0, dyi)
    v1atu = 0.25 * (v1[1:] + _roll_e(v1[1:]) + v1[:-1] + _roll_e(v1[:-1]))
    v0atu = 0.25 * (v0[1:] + _roll_e(v0[1:]) + v0[:-1] + _roll_e(v0[:-1]))

    vdv1 = u0 * du0x + v0atu * du0y
    vdv2 = u1 * du0x + v1atu * du0y
    vdv3 = u0 * du1x + v0atu * du1y
    vdv4 = u1 * du1x + v1atu * du1y
    divv3 = d0 * u1                      # TOPO only (div0 = 0 otherwise)
    divv4 = div1 * u1

    half = 0.5 if v1v0advh else 1.0
    advu0 = -vdv1 - VIJKT[3] * vdv4 * half
    advwu0 = -VWIJKT[2] * divv3 - VWIJKT[3] * divv4
    if not no_wadv:
        # note: the Fortran *0.5 continuation binds only to the divv4 term
        advu0 = advu0 - VWIJKT[2] * divv3 - VWIJKT[3] * divv4 * half
    advu1 = -(vdv2 + vdv3 + VIJKT[7] * vdv4)
    advwu1 = -VWIJKT[6] * divv3 - VWIJKT[7] * divv4
    if not no_wadv:
        advu1 = advu1 - VWIJKT[6] * divv3 - VWIJKT[7] * divv4

    # ---------------- y-momentum (v rows j=1..ny-1) --------------------
    rows = slice(1, ny)                  # v-grid interior rows
    dxih_v = (0.5 * g.dxvi[rows])[:, None]
    v1_in, v0_in = v1[rows], v0[rows]
    dv1x = (_roll_e(v1_in) - _roll_w(v1_in)) * dxih_v
    dv0x = (_roll_e(v0_in) - _roll_w(v0_in)) * dxih_v
    dv1y = 0.5 * (v1[2:] - v1[:-2]) * dyi
    dv0y = 0.5 * (v0[2:] - v0[:-2]) * dyi
    # u at v rows j: mean of u rows j, j+1 and their west neighbors
    u1atv = 0.25 * (u1[:-1] + _roll_w(u1[:-1]) + u1[1:] + _roll_w(u1[1:]))
    u0atv = 0.25 * (u0[:-1] + _roll_w(u0[:-1]) + u0[1:] + _roll_w(u0[1:]))

    vdv1 = u0atv * dv0x + v0_in * dv0y
    vdv2 = u1atv * dv0x + v1_in * dv0y
    vdv3 = u0atv * dv1x + v0_in * dv1y
    vdv4 = u1atv * dv1x + v1_in * dv1y
    div1_v = div1[: ny - 1]              # Fortran uses div1(:, j) at v row j
    d0_v = d0[: ny - 1]
    divv3 = d0_v * v1_in
    divv4 = div1_v * v1_in

    half_v = 0.5 if v1v0advvh else 1.0
    advv0 = np.zeros_like(v0)
    advv1 = np.zeros_like(v1)
    advv0[rows] = -vdv1 - VIJKT[3] * vdv4 * half_v
    if not no_wadv:
        advv0[rows] += -VWIJKT[2] * divv3 - VWIJKT[3] * divv4 * half_v
    advv1[rows] = -(vdv2 + vdv3 + VIJKT[7] * vdv4)
    if not no_wadv:
        advv1[rows] += -VWIJKT[6] * divv3 - VWIJKT[7] * divv4

    return dict(advu0=advu0, advu1=advu1, advv0=advv0, advv1=advv1,
                advwu0=advwu0, advwu1=advwu1, div1=div1)


def advctTq(T1, q1, u1, v1, u0, v0, grid):
    """Advection of T1 and q1 on T rows (port of ``advctTq``).

    Note the Fortran folds ``dxui`` into the x-derivative but leaves the
    0.5 factors explicit; reproduced term for term. Returns advT1, advq1.
    """
    g = grid
    dyi = g.dyi
    u1atC = 0.5 * (u1 + _roll_w(u1))
    u0atC = 0.5 * (u0 + _roll_w(u0))
    v1atC = 0.5 * (v1[1:] + v1[:-1])
    v0atC = 0.5 * (v0[1:] + v0[:-1])

    dT1x = 0.5 * (_roll_e(T1) - _roll_w(T1))
    dq1x = 0.5 * (_roll_e(q1) - _roll_w(q1))
    dT1y = _ddy_bounded(T1, 1.0)         # coefficients only; dyi applied below
    dq1y = _ddy_bounded(q1, 1.0)

    dxui = g.dxui[:, None]
    advT1 = (-DTIJKT[0] * (u0atC * dT1x * dxui + v0atC * dT1y * dyi)
             - DTIJKT[1] * (u1atC * dT1x * dxui + v1atC * dT1y * dyi))
    advq1 = (-DQIJKT[0] * (u0atC * dq1x * dxui + v0atC * dq1y * dyi)
             - DQIJKT[1] * (u1atC * dq1x * dxui + v1atC * dq1y * dyi))
    return dict(advT1=advT1, advq1=advq1)
