"""FATD solver tests: exact inversion of the manufactured discrete operator."""

import numpy as np
import pytest

from qtcm1.grid import Grid
from qtcm1.dynamics.elliptic import (FATDSolver, PoissonDirichlet,
                                     PoissonNeumann)

RNG = np.random.default_rng(42)


def apply_stencil(a, b, c, d, u):
    """Apply the FATD 5-point operator (periodic x) to u of shape (nyy, nx)."""
    r = (d[:, None] * (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1))
         + b[:, None] * u)
    r[1:] += a[1:, None] * u[:-1]
    r[:-1] += c[:-1, None] * u[1:]
    return r


def test_fatdsolver_roundtrip():
    nyy, nx = 41, 64
    a = 1.0 + 0.1 * RNG.random(nyy)
    c = 1.0 + 0.1 * RNG.random(nyy)
    d = np.ones(nyy)
    b = -(a + c + 2 * d) - 1.0 - RNG.random(nyy)   # diagonally dominant
    a[0] = 0.0
    c[-1] = 0.0
    u_true = RNG.standard_normal((nyy, nx))
    r = apply_stencil(a, b, c, d, u_true)
    u = FATDSolver(a, b, c, d, nx).solve(r)
    np.testing.assert_allclose(u, u_true, rtol=0, atol=1e-10)


def test_poisson_dirichlet_recovers_manufactured_psi():
    g = Grid()
    ny, nx = g.ny, g.nx
    # smooth manufactured streamfunction on v-points (rows 0..ny)
    lam = 2 * np.pi * np.arange(nx) / nx
    phi = np.linspace(0, np.pi, ny + 1)
    psi_true = (np.sin(phi)[:, None] ** 2
                * (np.cos(2 * lam) + 0.3 * np.sin(3 * lam))) * 1e6
    # rhs from the documented discrete operator on interior rows 1..ny-1
    dy2i = 1.0 / g.dy ** 2
    cosv_in = g.cosv[1:ny]
    interior = psi_true[1:ny]
    d2x = (np.roll(interior, 1, 1) - 2 * interior + np.roll(interior, -1, 1))
    d2y = psi_true[0:ny - 1] - 2 * interior + psi_true[2:ny + 1]
    rhs = d2x / (g.dx * cosv_in[:, None]) ** 2 + d2y * dy2i
    psi = PoissonDirichlet(g).solve(rhs, psi_south=psi_true[0],
                                    psi_north=psi_true[ny])
    np.testing.assert_allclose(psi, psi_true, rtol=0,
                               atol=1e-7 * np.abs(psi_true).max())


def test_poisson_dirichlet_zero_bc_zonal_mean():
    """Zonal-mean (k=0) component solved correctly with psi=0 walls."""
    g = Grid()
    rhs = np.ones((g.ny - 1, g.nx)) * 1e-10       # uniform vorticity
    psi = PoissonDirichlet(g).solve(rhs)
    assert psi.shape == (g.ny + 1, g.nx)
    np.testing.assert_allclose(psi[0], 0.0)
    np.testing.assert_allclose(psi[-1], 0.0)
    # solution is zonally uniform and negative-definite inside
    assert np.ptp(psi[1:-1], axis=1).max() < 1e-6 * np.abs(psi).max()
    assert psi[1:-1].max() < 0


def test_poisson_neumann_recovers_chi_up_to_constant():
    g = Grid()
    ny, nx = g.ny, g.nx
    solver = PoissonNeumann(g)
    lam = 2 * np.pi * np.arange(nx) / nx
    chi_true = (np.cos(np.linspace(0.2, 2.9, ny))[:, None]
                * np.cos(3 * lam)) * 1e5
    # apply the exact Neumann matrix the solver was built from
    lamsc = solver._lam_scale
    a = np.empty(ny); b = np.empty(ny); c = np.empty(ny)
    rdy2 = 1.0 / g.dy ** 2
    lamv = rdy2 * (g.cosu * g.dx) ** 2
    a[:], b[:], c[:] = lamv, -2.0 - 2.0 * lamv, lamv
    a[0], b[0], c[0] = 0.0, -2.0 - lamv[0], lamv[0]
    a[-1], b[-1], c[-1] = lamv[-1], -2.0 - lamv[-1], 0.0
    d = np.ones(ny)
    r_scaled = np.zeros((ny, nx))
    r_scaled += (d[:, None] * (np.roll(chi_true, 1, 1)
                               + np.roll(chi_true, -1, 1))
                 + b[:, None] * chi_true)
    r_scaled[1:] += a[1:, None] * chi_true[:-1]
    r_scaled[:-1] += c[:-1, None] * chi_true[1:]
    rhs = r_scaled / lamsc[:, None]
    chi = solver.solve(rhs)
    # defined up to a constant (k=0 nullspace): compare demeaned fields
    diff = (chi - chi.mean()) - (chi_true - chi_true.mean())
    assert np.abs(diff).max() < 1e-6 * np.abs(chi_true).max()


def test_dirichlet_matches_dense_solve():
    """Cross-check the FATD path against a dense linear solve."""
    g = Grid(nx=16, ny=10)
    ny, nx = g.ny, g.nx
    solver = PoissonDirichlet(g)
    rhs = RNG.standard_normal((ny - 1, nx)) * 1e-9
    psi = solver.solve(rhs)
    # dense operator
    n = (ny - 1) * nx
    A = np.zeros((n, n))
    dy2i = 1.0 / g.dy ** 2
    cosv_in = g.cosv[1:ny]
    def idx(j, i):
        return j * nx + i % nx
    for j in range(ny - 1):
        cf = 1.0 / (g.dx * cosv_in[j]) ** 2
        for i in range(nx):
            A[idx(j, i), idx(j, i)] += -2 * cf - 2 * dy2i
            A[idx(j, i), idx(j, i - 1)] += cf
            A[idx(j, i), idx(j, i + 1)] += cf
            if j > 0:
                A[idx(j, i), idx(j - 1, i)] += dy2i
            if j < ny - 2:
                A[idx(j, i), idx(j + 1, i)] += dy2i
    x = np.linalg.solve(A, rhs.ravel())
    np.testing.assert_allclose(psi[1:ny].ravel(), x, rtol=0,
                               atol=1e-9 * np.abs(x).max())
