"""C-grid geometry, metric factors, and diffusion weights.

Port of ``Module Grid`` (qtcmmod.F90) and the grid section of ``parinit``
(qtcm.F90). QTCM1 uses an Arakawa C-grid, periodic in longitude, walls at
``YB = 78.75`` deg latitude, with u/T ("full") points centered per row and
v ("half") points staggered between rows.

Index conventions in this port: Python arrays are 0-based; a Fortran
u-point array ``x(j), j=1..ny`` maps to ``x[j-1]``, and a v-point array
``x(j), j=0..ny`` maps to ``x[j]`` with the same length ``ny+1``. All arrays
here are float64 by default (``dtype`` parameter); the Fortran is float32,
which only matters for Tier-1/2 comparisons, not for grid geometry.
"""

from __future__ import annotations

import numpy as np

from .constants import REARTH, OMEGA


class Grid:
    """QTCM1 horizontal grid (default r64x42: 5.625 x 3.75 deg, abs(lat) < 78.75).

    Attributes mirror ``Module Grid``; see the Fortran name in each comment.
    """

    YB = 78.75          #: domain spans YB S .. YB N [degrees]

    def __init__(self, nx: int = 64, ny: int = 42, dtype=np.float64):
        self.nx, self.ny = nx, ny
        self.nxy = nx * ny
        f = lambda v: np.asarray(v, dtype=dtype)

        pi = np.pi
        self.dx = 2.0 * pi * REARTH / nx            # dx (equator) [m]
        self.dxi = 1.0 / self.dx
        self.dy = self.YB / 90.0 * pi * REARTH / ny  # dy [m]
        self.dyi = 1.0 / self.dy

        # -- v-point ("half", staggered) rows: Fortran j = 0..ny ---------
        j = np.arange(0, ny + 1)
        thetav = self.dy / REARTH * (j + 0.5 - (ny + 1.0) / 2.0)
        self.fv = f(2.0 * OMEGA * np.sin(thetav))    # Coriolis at (j+1/2)
        self.cosv = f(np.cos(thetav))
        self.cosvi = f(1.0 / self.cosv)
        self.dxvi = f(self.cosvi * self.dxi)         # 1/(cosv*dx)
        self.dyvi = f(self.cosvi * self.dyi)         # 1/(cosv*dy)
        self.latv = f(np.degrees(thetav))            # convenience (not in F90)

        # -- u/T-point ("full", centered) rows: Fortran j = 1..ny --------
        j = np.arange(1, ny + 1)
        thetau = self.dy / REARTH * (j - (ny + 1.0) / 2.0)
        self.fu = f(2.0 * OMEGA * np.sin(thetau))    # Coriolis at full rows
        self.cosu = f(np.cos(thetau))
        self.cosui = f(1.0 / self.cosu)
        self.dxui = f(self.cosui * self.dxi)
        self.dyui = f(self.cosui * self.dyi)

        # -- T-point coordinates (parinit: latt, lont) -------------------
        jj = np.arange(1, ny + 1)
        self.latt = f((self.YB / ny) * (2 * jj - ny - 1))
        self.lont = f((360.0 / nx) * np.arange(nx))

        # -- spherical diffusion weights (parinit) -----------------------
        # weight2*(j, 1:3), weight4*(j, 1:5); rows outside the loops stay 0
        # exactly as in Fortran (they are never used there). The stated
        # conservation property (tested in tests/test_grid.py):
        #   sum_j cosu(j) * sum_k weight2u(j,k) T(j+k-2) = 0 (interior)
        cosu_g = np.concatenate([[np.nan], self.cosu, [np.nan]])  # 1-based+ghosts
        cosv_g = self.cosv                                        # index j = 0..ny
        w2u = np.zeros((ny + 1, 3), dtype=dtype)   # index by Fortran j=1..ny
        w4u = np.zeros((ny + 1, 5), dtype=dtype)
        for jf in range(2, ny):                    # Fortran j=2..ny-1
            w2u[jf, 0] = cosv_g[jf] * self.cosui[jf - 1]
            w2u[jf, 1] = -(cosv_g[jf] + cosv_g[jf - 1]) * self.cosui[jf - 1]
            w2u[jf, 2] = cosv_g[jf - 1] * self.cosui[jf - 1]
            w4u[jf, 0] = cosu_g[jf + 1] * self.cosui[jf - 1]
            w4u[jf, 1] = -2.0 * (cosu_g[jf + 1] + cosu_g[jf]) * self.cosui[jf - 1]
            w4u[jf, 2] = ((2.0 * (cosu_g[jf + 1] + 2.0 * cosu_g[jf]
                                  + cosu_g[jf - 1])
                           - cosu_g[jf + 1] - cosu_g[jf - 1])
                          * self.cosui[jf - 1])
            w4u[jf, 3] = -2.0 * (cosu_g[jf] + cosu_g[jf - 1]) * self.cosui[jf - 1]
            w4u[jf, 4] = cosu_g[jf - 1] * self.cosui[jf - 1]
        w2v = np.zeros((ny + 1, 3), dtype=dtype)   # index by Fortran j=1..ny-1
        w4v = np.zeros((ny + 1, 5), dtype=dtype)
        for jf in range(1, ny):                    # Fortran j=1..ny-1
            w2v[jf, 0] = cosu_g[jf + 1] * self.cosvi[jf]
            w2v[jf, 1] = -(cosu_g[jf + 1] + cosu_g[jf]) * self.cosvi[jf]
            w2v[jf, 2] = cosu_g[jf] * self.cosvi[jf]
            w4v[jf, 0] = cosv_g[jf + 1] * self.cosvi[jf]
            w4v[jf, 1] = -2.0 * (cosv_g[jf + 1] + cosv_g[jf]) * self.cosvi[jf]
            w4v[jf, 2] = ((2.0 * (cosv_g[jf + 1] + 2.0 * cosv_g[jf]
                                  + cosv_g[jf - 1])
                           - cosv_g[jf + 1] - cosv_g[jf - 1])
                          * self.cosvi[jf])
            w4v[jf, 3] = -2.0 * (cosv_g[jf] + cosv_g[jf - 1]) * self.cosvi[jf]
            w4v[jf, 4] = cosv_g[jf - 1] * self.cosvi[jf]
        self.weight2u, self.weight4u = w2u, w4u    # Fortran-j indexed rows
        self.weight2v, self.weight4v = w2v, w4v

        # -- periodic-x neighbor index arrays (parinit im1/ip1/im2/ip2) --
        i = np.arange(nx)
        self.im1 = np.roll(i, 1)     # index of point i-1 (periodic)
        self.ip1 = np.roll(i, -1)
        self.im2 = np.roll(i, 2)
        self.ip2 = np.roll(i, -2)

    def __repr__(self) -> str:  # pragma: no cover
        return (f'Grid(nx={self.nx}, ny={self.ny}, '
                f'dlon={360.0 / self.nx}, dlat={2 * self.YB / self.ny})')
