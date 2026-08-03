"""FATD elliptic solver: Fourier analysis in x + tridiagonal solve in y.

Port of ``fatdpkg.F90`` (Adcroft, UCLA 1997) and its QTCM front ends
``fatdfe_di`` / ``fatdfe_neu`` (qtcm.F90, spherical geometry, H. Su 2002).

Solves, for periodic x and coefficients constant in x,

    d(j)*[u(i-1,j) + u(i+1,j)] + a(j)*u(i,j-1) + b(j)*u(i,j) + c(j)*u(i,j+1)
        = r(i,j).

An x-FFT diagonalizes the i-coupling: each zonal wavenumber k satisfies an
independent tridiagonal system with modified diagonal
b(j) + 2 d(j) cos(2 pi k / nx); the Thomas factorization of every wavenumber
is precomputed once (``inifatd``) and each solve is one rfft, one vectorized
forward/back substitution, and one irfft (``fatd``). The bundled FFTPACK in
the Fortran was modified to normalize ``rfftb``, which is exactly NumPy's
``rfft``/``irfft`` convention, so no scaling adjustments are needed.

Array convention: fields are ``(ny, nx)`` = (lat, lon), C-order.
"""

from __future__ import annotations

import numpy as np


class FATDSolver:
    """Generic periodic-x / tridiagonal-y solver (``inifatd`` + ``fatd``).

    Parameters are the y-diagonals ``a, b, c`` (sub/main/super) and the
    x-coupling coefficient ``d``, each of shape ``(nyy,)`` where ``nyy`` is
    the number of solved rows. Rows are indexed 0-based here; callers map
    from Fortran 1-based rows.

    Note: with pure Neumann boundary rows the k = 0 (zonal-mean) subsystem is
    singular up to a constant; as in the Fortran, no pinning is applied - the
    arbitrary constant is harmless for fields used only through gradients
    (velocity potential, surface geopotential).
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                 d: np.ndarray, nx: int):
        a, b, c, d = (np.asarray(x, dtype=np.float64) for x in (a, b, c, d))
        nyy = a.shape[0]
        self.nx, self.nyy = nx, nyy
        nk = nx // 2 + 1
        k = np.arange(nk)
        # modified main diagonal per wavenumber: (nyy, nk)
        bmod = b[:, None] + 2.0 * d[:, None] * np.cos(2.0 * np.pi * k / nx)
        # Thomas prefactorization (initri2d), vectorized over wavenumbers
        bet = np.empty_like(bmod)
        gam = np.zeros_like(bmod)
        bet[0] = bmod[0]
        for j in range(1, nyy):
            gam[j] = c[j - 1] / bet[j - 1]
            bet[j] = bmod[j] - a[j] * gam[j]
        self._a, self._bet, self._gam = a, bet, gam

    def solve(self, r: np.ndarray) -> np.ndarray:
        """Solve for ``u`` given right-hand side ``r`` of shape (nyy, nx)."""
        rhat = np.fft.rfft(r, axis=1)                  # (nyy, nk) complex
        a, bet, gam = self._a, self._bet, self._gam
        y = np.empty_like(rhat)
        y[0] = rhat[0] / bet[0]
        for j in range(1, self.nyy):                   # forward substitution
            y[j] = (rhat[j] - a[j] * y[j - 1]) / bet[j]
        for j in range(self.nyy - 2, -1, -1):          # back substitution
            y[j] = y[j] - gam[j + 1] * y[j + 1]
        return np.fft.irfft(y, n=self.nx, axis=1)


class PoissonDirichlet:
    """Streamfunction solver: port of ``fatdfe_di`` (qtcm.F90).

    Solves, on v-point rows ``j = 1..ny-1`` (Fortran numbering) with
    prescribed boundary rows ``psi(:, j=0)`` and ``psi(:, j=ny)``,

        [u(i-1,j) - 2 u(i,j) + u(i+1,j)] / (dx cosv_j)^2
        + [u(i,j-1) - 2 u(i,j) + u(i,j+1)] / dy^2  =  rhs(i,j),

    i.e. the discrete Laplacian in the model's spherical-metric
    approximation. Used by ``bartr`` for nabla^2 psi0 = zeta0.
    """

    def __init__(self, grid):
        self.grid = grid
        ny, nx = grid.ny, grid.nx
        dx2 = grid.dx ** 2
        self.dy2i = 1.0 / grid.dy ** 2
        cosv = grid.cosv                       # index j = 0..ny
        j = np.arange(1, ny)                   # Fortran rows 1..ny-1
        self._cosv_in = cosv[j]
        a = self.dy2i * self._cosv_in ** 2 * dx2
        b = -2.0 - 2.0 * self.dy2i * self._cosv_in ** 2 * dx2
        c = a.copy()
        d = np.ones_like(a)
        a[0] = 0.0                             # Dirichlet: BC rows to RHS
        c[-1] = 0.0
        self._dx2 = dx2
        self._solver = FATDSolver(a, b, c, d, nx)

    def solve(self, rhs: np.ndarray, psi_south: np.ndarray | float = 0.0,
              psi_north: np.ndarray | float = 0.0) -> np.ndarray:
        """Return psi of shape (ny+1, nx) given rhs on rows 1..ny-1.

        ``rhs`` may be (ny, nx) (Fortran layout, row ny unused) or
        (ny-1, nx); boundary rows of the result are the prescribed BCs.
        """
        g = self.grid
        r_in = rhs[:g.ny - 1] if rhs.shape[0] >= g.ny - 1 else rhs
        r = r_in * (self._dx2 * self._cosv_in ** 2)[:, None]
        bc_fac = self._dx2 * self.dy2i
        r = r.copy()
        r[0] -= self._cosv_in[0] ** 2 * bc_fac * np.asarray(psi_south)
        r[-1] -= self._cosv_in[-1] ** 2 * bc_fac * np.asarray(psi_north)
        uu = self._solver.solve(r)
        psi = np.empty((g.ny + 1, g.nx), dtype=uu.dtype)
        psi[0] = psi_south
        psi[1:g.ny] = uu
        psi[g.ny] = psi_north
        return psi


class PoissonNeumann:
    """Velocity-potential solver: port of ``fatdfe_neu`` (qtcm.F90).

    Solves the analogous system on u-point rows ``j = 1..ny`` with Neumann
    conditions (v = 0) at both walls; solution defined up to a constant
    (see FATDSolver note). Used for chi / surface-geopotential diagnostics.
    """

    def __init__(self, grid):
        self.grid = grid
        ny, nx = grid.ny, grid.nx
        dx2 = grid.dx ** 2
        rdy2 = 1.0 / grid.dy ** 2
        cosu = grid.cosu                       # Fortran rows 1..ny
        lam = rdy2 * (cosu * grid.dx) ** 2     # rdy2*(cosu(j)*dx)**2
        a = lam.copy()
        b = -2.0 - 2.0 * lam
        c = lam.copy()
        d = np.ones(ny)
        # Neumann rows exactly as fatdfe_neu:
        a[0], b[0], c[0] = 0.0, -2.0 - lam[0], lam[0]
        a[-1], b[-1], c[-1] = lam[-1], -2.0 - lam[-1], 0.0
        self._lam_scale = (cosu * grid.dx) ** 2
        self._solver = FATDSolver(a, b, c, d, nx)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Return chi of shape (ny, nx) for divergence-like rhs (ny, nx)."""
        r = rhs * self._lam_scale[:, None]
        return self._solver.solve(r)
