"""Horizontal diffusion: ports of ``nabla2``, ``nabla4mm5``, ``dffus``.

Grid-row alignment convention used throughout the dynamics modules:

* u/T-grid fields ("full" rows, Fortran ``j = 1..ny``) are ``(ny, nx)``
  arrays with Python row ``p`` = Fortran row ``j = p + 1``.
* v-grid fields ("half" rows, Fortran ``j = 0..ny``) are ``(ny+1, nx)``
  arrays with Python row ``p`` = Fortran row ``j = p``.

The generic operators take the field, its per-row ``1/(cos*dx)`` array, and
a row-aligned weight table; ``dffus`` wires them exactly as the Fortran does
(including which viscosities pair with which operator). Former compile-time
options are keyword flags: ``vi2u0``/``vi2u1`` (Laplacian instead of
del-4 for the winds), ``spherical`` (NO_SPHERDFS inverse).
"""

from __future__ import annotations

import numpy as np

try:                                       # optional speedup; bit-identical
    from numba import njit
    _NUMBA = True
except ImportError:                        # pragma: no cover
    _NUMBA = False

    def njit(*a, **k):                     # no-op decorator
        return a[0] if a and callable(a[0]) else (lambda f: f)


def _roll_e(f):
    """f at east neighbor (Fortran ip1)."""
    return np.roll(f, -1, axis=1)


def _roll_w(f):
    """f at west neighbor (Fortran im1)."""
    return np.roll(f, 1, axis=1)


@njit(cache=True)
def _nabla2_sph(viscx, viscy, fld, dxi, w2, dyi):
    """Loop form of the spherical nabla2; per-element ops in the exact
    order of the vectorized expressions (bit-identical, no fastmath)."""
    n, nx = fld.shape
    out = np.empty_like(fld)
    vydy2i = viscy * dyi ** 2
    for p in range(n):
        vx = viscx * dxi[p] ** 2
        for i in range(nx):
            e = fld[p, (i + 1) % nx]
            w = fld[p, (i - 1) % nx]
            d2x = vx * (e - 2.0 * fld[p, i] + w)
            if p == 0:
                out[p, i] = d2x + vydy2i * (fld[1, i] - fld[0, i])
            elif p == n - 1:
                out[p, i] = d2x + vydy2i * (-fld[n - 1, i] + fld[n - 2, i])
            else:
                out[p, i] = d2x + vydy2i * (
                    fld[p + 1, i] * w2[p, 0] + fld[p, i] * w2[p, 1]
                    + fld[p - 1, i] * w2[p, 2])
    return out


@njit(cache=True)
def _nabla4_sph(viscx, viscy, fld, dxi, w4, dyi):
    """Loop form of the spherical nabla4mm5 (bit-identical op order)."""
    n, nx = fld.shape
    out = np.empty_like(fld)
    vydy2i = viscy * dyi ** 2
    for p in range(n):
        vx = viscx * dxi[p] ** 2
        for i in range(nx):
            e = fld[p, (i + 1) % nx]
            w = fld[p, (i - 1) % nx]
            if 2 <= p <= n - 3:
                ee = fld[p, (i + 2) % nx]
                ww = fld[p, (i - 2) % nx]
                d4x = -vx * (ee + ww - 4.0 * (e + w) + 6.0 * fld[p, i])
                d4y = -vydy2i * (
                    fld[p + 2, i] * w4[p, 0] + fld[p + 1, i] * w4[p, 1]
                    + fld[p, i] * w4[p, 2] + fld[p - 1, i] * w4[p, 3]
                    + fld[p - 2, i] * w4[p, 4])
                out[p, i] = d4x + d4y
            else:
                lap = vx * (e - 2.0 * fld[p, i] + w)
                if p == 0:
                    out[p, i] = lap + vydy2i * (fld[1, i] - fld[0, i])
                elif p == 1:
                    out[p, i] = lap + vydy2i * (fld[2, i] + fld[0, i]
                                                - 2.0 * fld[1, i])
                elif p == n - 2:
                    out[p, i] = lap + vydy2i * (fld[n - 1, i] + fld[n - 3, i]
                                                - 2.0 * fld[n - 2, i])
                else:
                    out[p, i] = lap + vydy2i * (-fld[n - 1, i]
                                                + fld[n - 2, i])
    return out


def nabla2(viscx: float, viscy: float, fld: np.ndarray, dxi: np.ndarray,
           weight2: np.ndarray, dyi: float,
           spherical: bool = True) -> np.ndarray:
    """Second-order diffusion operator (port of ``nabla2``).

    ``fld``: (nrows, nx) with rows spanning Fortran ``js..ny``; ``dxi`` and
    ``weight2`` row-aligned with ``fld``. Boundary rows use one-sided,
    flux-zero forms exactly as the Fortran.
    """
    if spherical and _NUMBA:
        return _nabla2_sph(float(viscx), float(viscy),
                           np.ascontiguousarray(fld),
                           np.ascontiguousarray(dxi),
                           np.ascontiguousarray(weight2), float(dyi))
    vydy2i = viscy * dyi ** 2
    vx = (viscx * dxi ** 2)[:, None]
    d2x = vx * (_roll_e(fld) - 2.0 * fld + _roll_w(fld))
    out = np.empty_like(fld)
    if spherical:
        out[1:-1] = d2x[1:-1] + vydy2i * (
            fld[2:] * weight2[1:-1, 0:1]
            + fld[1:-1] * weight2[1:-1, 1:2]
            + fld[:-2] * weight2[1:-1, 2:3])
    else:
        out[1:-1] = d2x[1:-1] + vydy2i * (fld[2:] - 2.0 * fld[1:-1]
                                          + fld[:-2])
    out[0] = d2x[0] + vydy2i * (fld[1] - fld[0])
    out[-1] = d2x[-1] + vydy2i * (-fld[-1] + fld[-2])
    return out


def nabla4mm5(viscx: float, viscy: float, fld: np.ndarray, dxi: np.ndarray,
              weight4: np.ndarray, dyi: float,
              spherical: bool = True) -> np.ndarray:
    """Fourth-order (MM5-style) diffusion operator (port of ``nabla4mm5``).

    Interior rows get the del-4 stencil (spherical weights in y); the two
    rows at each wall fall back to the one-sided Laplacian forms coded in
    the Fortran.
    """
    if spherical and _NUMBA:
        return _nabla4_sph(float(viscx), float(viscy),
                           np.ascontiguousarray(fld),
                           np.ascontiguousarray(dxi),
                           np.ascontiguousarray(weight4), float(dyi))
    vydy2i = viscy * dyi ** 2
    vx = (viscx * dxi ** 2)[:, None]
    fE, fW = _roll_e(fld), _roll_w(fld)
    fEE, fWW = np.roll(fld, -2, axis=1), np.roll(fld, 2, axis=1)
    out = np.empty_like(fld)

    inner = slice(2, -2)
    d4x = -vx[inner] * (fEE[inner] + fWW[inner]
                        - 4.0 * (fE[inner] + fW[inner]) + 6.0 * fld[inner])
    if spherical:
        d4y = -vydy2i * (fld[4:] * weight4[inner, 0:1]
                         + fld[3:-1] * weight4[inner, 1:2]
                         + fld[inner] * weight4[inner, 2:3]
                         + fld[1:-3] * weight4[inner, 3:4]
                         + fld[:-4] * weight4[inner, 4:5])
    else:
        d4y = -vydy2i * (fld[4:] + fld[:-4]
                         - 4.0 * (fld[3:-1] + fld[1:-3]) + 6.0 * fld[inner])
    out[inner] = d4x + d4y

    lap_x = vx * (fE - 2.0 * fld + fW)
    out[0] = lap_x[0] + vydy2i * (fld[1] - fld[0])
    out[1] = lap_x[1] + vydy2i * (fld[2] + fld[0] - 2.0 * fld[1])
    out[-2] = lap_x[-2] + vydy2i * (fld[-1] + fld[-3] - 2.0 * fld[-2])
    out[-1] = lap_x[-1] + vydy2i * (-fld[-1] + fld[-2])
    return out


def dffus(u1, v1, u0, v0, T1, q1, grid, *,
          viscxu1, viscyu1, visc4x, visc4y, viscxT, viscyT,
          viscxq, viscyq, viscxu0=None, viscyu0=None,
          vi2u0: bool = False, vi2u1: bool = False,
          spherical: bool = True) -> dict[str, np.ndarray]:
    """Diffusion tendencies for all prognostics (port of ``dffus``).

    u/T-grid inputs (u1, u0, T1, q1): (ny, nx); v-grid (v1, v0): (ny+1, nx).
    Returns dfsu1, dfsv1, dfsu0, dfsv0, dfsT1, dfsq1 on the same rows as
    their fields. Default build: del-4 for winds (u1/v1 with viscxu1/viscyu1;
    u0/v0 with visc4x/visc4y), Laplacian for T1/q1.
    """
    g = grid
    dyi = g.dyi
    w2u, w4u = g.weight2u[1:], g.weight4u[1:]     # align to u-grid rows
    w2v, w4v = g.weight2v, g.weight4v[: g.ny + 1]  # align to v-grid rows

    if vi2u1:
        dfsu1 = nabla2(viscxu1, viscyu1, u1, g.dxui, w2u, dyi, spherical)
        dfsv1 = nabla2(viscxu1, viscyu1, v1, g.dxvi, w2v, dyi, spherical)
    else:
        dfsu1 = nabla4mm5(viscxu1, viscyu1, u1, g.dxui, w4u, dyi, spherical)
        dfsv1 = nabla4mm5(viscxu1, viscyu1, v1, g.dxvi, w4v, dyi, spherical)

    if vi2u0:
        dfsu0 = nabla2(viscxu0, viscyu0, u0, g.dxui, w2u, dyi, spherical)
        dfsv0 = nabla2(viscxu0, viscyu0, v0, g.dxvi, w2v, dyi, spherical)
    else:
        dfsu0 = nabla4mm5(visc4x, visc4y, u0, g.dxui, w4u, dyi, spherical)
        dfsv0 = nabla4mm5(visc4x, visc4y, v0, g.dxvi, w4v, dyi, spherical)

    dfsT1 = nabla2(viscxT, viscyT, T1, g.dxui, w2u, dyi, spherical)
    dfsq1 = nabla2(viscxq, viscyq, q1, g.dxui, w2u, dyi, spherical)

    return dict(dfsu1=dfsu1, dfsv1=dfsv1, dfsu0=dfsu0, dfsv0=dfsv0,
                dfsT1=dfsT1, dfsq1=dfsq1)
