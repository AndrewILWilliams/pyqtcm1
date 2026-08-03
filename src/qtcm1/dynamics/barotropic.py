"""Barotropic mode: ports of ``bartr``, ``Savebartr``, ``gradphis``,
``dphiint`` (qtcm.F90).

The barotropic prognostics are the vorticity ``vort0`` (at v rows 1..ny-1)
and the domain-average zonal wind ``u0bar``, both advanced with third-order
Adams-Bashforth (the Fortran keeps three RHS slices and cycles indices; the
port carries the two previous RHS explicitly). Each step:

1. RHS = curl_z of (-gpTi*stress + advection + diffusion tendencies) plus
   the beta term (NZ eq. 5.2, spherical form);
2. AB3 update of vort0, polar-filtered; AB3 update of u0bar from
   cos-weighted domain sums (the Fortran ``hbar`` divides by nx*ny, not by
   the weight sum - reproduced verbatim);
3. streamfunction from nabla^2 psi0 = vort0 by the FATD Dirichlet solve,
   with psi0 = 0 at the south wall and psi0 = -ny*dy*u0bar at the north
   wall (carrying the mean zonal wind);
4. winds diagnosed: u0 = -d(psi0)/dy, v0 = d(psi0)/dx, v0 = 0 at walls.

``gradphis`` recovers the surface geopotential gradient from the mode-0
momentum balance (NZ eqs. 4.6/4.10/4.11, appendix A): the wind tendency
over the barotropic step (vs the pre-step winds saved by ``savebartr``)
plus advection, diffusion, Coriolis, stress, and the baroclinic
R*a1phat*grad(T1) contribution. ``dphiint`` line-integrates the gradient
(along the Fortran's row nyh=ny/2 first, then meridionally, reproducing
the half-row sequence-association quirk of passing the (0:ny) v-grid array
to a (1:ny) dummy) and normalizes to a 1013.25-hPa mean surface pressure.
"""

from __future__ import annotations

import numpy as np

from ..constants import A1PHAT, GPTI, RAIR
from .advection import _roll_e, _roll_w

_ADAMS1, _ADAMS2, _ADAMS3 = 23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0
_PREF = 101325.0          # reference surface pressure [Pa]
_RHOAIR_PS = 1.2          # air density used in dphiint [kg m-3]


def hbar(a, wgh) -> float:
    """Cos-weighted domain sum / (nx*ny) - verbatim ``hbar`` semantics."""
    return float((a * wgh[:, None]).sum() / a.size)


def savebartr(u0, v0) -> dict:
    """Snapshot winds entering the barotropic step (port of ``Savebartr``).

    The Fortran copies v0 rows 1..ny into an (nx, ny) array, dropping the
    south-wall row; ``v0sav`` is returned in that convention as (ny, nx).
    """
    return dict(u0sav=u0.copy(), v0sav=v0[1:].copy())


def bartr(vort0, u0bar, v0, rhs_hist, rhsbar_hist, *, taux, tauy,
          advu0, advv0, dfsu0, dfsv0, grid, polar_filter, poisson,
          dt, mt0=1) -> dict:
    """One barotropic step (port of ``bartr``).

    Layout: ``vort0`` is (ny, nx) with active v rows 1..ny-1 stored in rows
    0..ny-2 (the Fortran (nx, ny) layout transposed); row ny-1 is inert.
    ``rhs_hist = [rhs_{n-1}, rhs_{n-2}]`` (same shape, zeros at cold start);
    ``rhsbar_hist`` the scalar analogue for u0bar. ``poisson`` is a
    :class:`~qtcm1.dynamics.elliptic.PoissonDirichlet` for this grid.

    Returns new vort0, u0bar, psi0 ((ny+1, nx) incl. walls), u0 ((ny, nx)),
    v0 ((ny+1, nx)) and updated histories.
    """
    g = grid
    ny = g.ny
    rows = slice(1, ny)                        # v rows j=1..ny-1

    # -- RHS of the vorticity equation on v rows 1..ny-1 ----------------
    tau_v = tauy[:-1]                          # tauy(i,j), j=1..ny-1
    curl_tau = -GPTI * (
        (_roll_e(tau_v) - tau_v) * g.dxvi[rows, None]
        - (taux[1:] * g.cosu[1:, None] - taux[:-1] * g.cosu[:-1, None])
        * g.dyvi[rows, None])
    beta = (-(g.fu[1:] - g.fu[:-1])[:, None] * g.dyi
            * 0.5 * (v0[rows] + _roll_e(v0[rows])))
    adv_v = advv0[rows]
    curl_adv = ((_roll_e(adv_v) - adv_v) * g.dxvi[rows, None]
                - (advu0[1:] * g.cosu[1:, None]
                   - advu0[:-1] * g.cosu[:-1, None]) * g.dyvi[rows, None])
    dfs_v = dfsv0[rows]
    curl_dfs = ((_roll_e(dfs_v) - dfs_v) * g.dxvi[rows, None]
                - (dfsu0[1:] * g.cosu[1:, None]
                   - dfsu0[:-1] * g.cosu[:-1, None]) * g.dyvi[rows, None])
    rhs = curl_tau + beta + curl_adv + curl_dfs        # (ny-1, nx)

    # -- AB3 updates -----------------------------------------------------
    vort_new = vort0.copy()
    vort_new[: ny - 1] = vort0[: ny - 1] + dt * mt0 * (
        _ADAMS1 * rhs + _ADAMS2 * rhs_hist[0] + _ADAMS3 * rhs_hist[1])
    vort_new = polar_filter(vort_new)

    rhsbar = (-GPTI * hbar(taux, g.cosu) + hbar(advu0, g.cosu)
              + hbar(dfsu0, g.cosu))
    u0bar_new = u0bar + dt * mt0 * (_ADAMS1 * rhsbar
                                    + _ADAMS2 * rhsbar_hist[0]
                                    + _ADAMS3 * rhsbar_hist[1])

    # -- streamfunction and winds ---------------------------------------
    psi_north = -ny * g.dy * u0bar_new         # carries the mean zonal wind
    psi0 = poisson.solve(vort_new[: ny - 1], psi_south=0.0,
                         psi_north=np.full(g.nx, psi_north))
    u0 = (psi0[:-1] - psi0[1:]) * g.dyi        # u rows 1..ny
    v0n = np.zeros_like(v0)
    v0n[rows] = (psi0[rows] - _roll_w(psi0[rows])) * g.dxvi[rows, None]

    return dict(vort0=vort_new, u0bar=u0bar_new, psi0=psi0, u0=u0, v0=v0n,
                rhs_hist=[rhs, rhs_hist[0]],
                rhsbar_hist=[rhsbar, rhsbar_hist[0]])


def gradphis(u0, v0, u0sav, v0sav, T1, *, taux, tauy, advu0, advv0,
             dfsu0, dfsv0, grid, dt, mt0=1) -> dict:
    """Surface geopotential gradient + surface pressure (port of
    ``gradphis``; NZ 4.6/4.10/4.11 & appendix A).

    ``u0sav``/``v0sav`` from :func:`savebartr` (pre-bartr winds; v0sav in
    the rows-1..ny convention). Returns dphisdx (ny, nx), dphisdy
    ((ny+1, nx), wall rows extrapolated) and ps (ny, nx).
    """
    g = grid
    ny = g.ny
    dti = 1.0 / (dt * mt0)
    radxi = (RAIR * A1PHAT / g.dx / g.cosu)[:, None]
    radyi = RAIR * A1PHAT / g.dy

    # zonal gradient on u/T rows: Coriolis uses the (i, i+1) v average
    vatu = 0.25 * (v0[1:] + _roll_e(v0[1:]) + v0[:-1] + _roll_e(v0[:-1]))
    dphisdx = (-(u0 - u0sav) * dti + advu0 + dfsu0
               + g.fu[:, None] * vatu - GPTI * taux
               - (_roll_e(T1) - T1) * radxi)

    # meridional gradient on v rows 1..ny-1
    v0_in = v0[1:ny]
    v0sav_in = v0sav[: ny - 1]                 # v0sav row 0 = v row 1
    uatv = np.empty_like(v0_in)
    uatv[0] = 0.5 * (u0[0] + np.roll(u0[0], -1))   # row j=1: only u row 1
    uatv[1:] = 0.25 * (u0[1:ny - 1] + _roll_e(u0[1:ny - 1])
                       + u0[: ny - 2] + _roll_e(u0[: ny - 2]))
    dphisdy = np.zeros_like(v0)
    dphisdy[1:ny] = (-(v0_in - v0sav_in) * dti + advv0[1:ny] + dfsv0[1:ny]
                     - g.fv[1:ny, None] * uatv - GPTI * tauy[:-1]
                     - (T1[1:] - T1[:-1]) * radyi)
    dphisdy[0] = 2.0 * dphisdy[1] - dphisdy[2]
    dphisdy[ny] = 2.0 * dphisdy[ny - 1] - dphisdy[ny - 2]

    ps = dphiint(dphisdx, dphisdy, grid)
    return dict(dphisdx=dphisdx, dphisdy=dphisdy, ps=ps)


def dphiint(phix, phiy_v, grid) -> np.ndarray:
    """Recover surface pressure from its gradient (port of ``dphiint``).

    ``phiy_v`` is the (ny+1, nx) v-grid gradient; the Fortran passes it to
    an (nx, ny) dummy, so T row j reads v rows j-1 and j-2 - the half-row
    quirk is reproduced by indexing phiy_v[r] for Fortran phiy(:, r+1).
    """
    g = grid
    ny, nx = g.ny, g.nx
    nyh = ny // 2                              # Fortran row 21 -> index 20
    r0 = nyh - 1
    dxuh = 0.5 * g.cosu * g.dx
    dyh = 0.5 * g.dy

    ps = np.zeros((ny, nx))
    # along the reference row: cumulative trapezoid in x, ps(0, r0) = 0
    incr = (phix[r0] + np.roll(phix[r0], 1)) * dxuh[r0]
    ps[r0] = np.cumsum(incr) - incr[0]
    # northward: T row r uses phiy_v rows r and r-1 (the quirk)
    for r in range(nyh, ny):
        ps[r] = ps[r - 1] + (phiy_v[r] + phiy_v[r - 1]) * dyh
    # southward
    for r in range(r0 - 1, -1, -1):
        ps[r] = ps[r + 1] - (phiy_v[r] + phiy_v[r + 1]) * dyh

    return _RHOAIR_PS * (ps - ps.mean()) + _PREF
