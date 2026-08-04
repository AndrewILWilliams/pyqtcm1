"""Validation of the TOPO option port.

Three independent pins, in increasing strength:

1. the V1st terrain interpolation and div0 stencil against a literal
   loop-for-loop transcription of the Fortran (independent of the
   vectorized implementation);
2. flat-topography TOPO == TOPO-off, bitwise, over a multi-day run
   (the option's machinery is exactly inert when grad(TOP) = 0);
3. the div0 stencil and the -fv*div0 vorticity term against the *actual
   Fortran expression* compiled by gfortran in double precision on
   random fields (bit-level; skipped when gfortran is absent).

Full-model goldens against a -DTOPO oracle build remain on the roadmap
(they need the f2py py3-port from the reference-build session).
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from qtcm1.config import PACKAGED_DATA, RunConfig
from qtcm1.constants import V1S, V1Z_TABLE
from qtcm1.dynamics.barotropic import topo_div0
from qtcm1.grid import Grid
from qtcm1.model import Model

_DATA = os.environ.get('QTCM1_BNDDATA', PACKAGED_DATA)
_needs_data = pytest.mark.skipif(not os.path.isdir(_DATA),
                                 reason='boundary registry not found')


def _v1st_map(top):
    """Model's precomputed V1st (via a throwaway Model would need stype;
    replicate the arithmetic here from the same table)."""
    nz = V1Z_TABLE.shape[0]
    we = top * nz + 1.0
    kw = we.astype(np.int64)
    frac = we - kw
    v1z = V1Z_TABLE[:, 1]
    return (1.0 - frac) * v1z[kw - 1] + frac * v1z[kw]


def _div0_fortran_loops(u1, v1, u0, v0, top, grid):
    """Literal transcription of the qtcm.F90 TOPO block (1-based logic)."""
    g = grid
    ny, nx = g.ny, g.nx
    v1st = _v1st_map(top)
    out = np.zeros((ny, nx))
    for jf in range(1, ny):                     # Fortran j = 1..ny-1
        p = jf - 1                              # u/T-row python index
        for i in range(nx):
            ie = (i + 1) % nx                   # ip1
            vs = v1st[p, i]
            out[p, i] = (
                ((u0[p, i] + vs * u1[p, i]) * (top[p, ie] - top[p, i])
                 + (u0[p + 1, i] + vs * u1[p + 1, i])
                 * (top[p + 1, ie] - top[p + 1, i])) * 0.5 * g.dxvi[jf]
                + ((v0[jf, i] + vs * v1[jf, i]) * (top[p + 1, i] - top[p, i])
                   + (v0[jf, ie] + vs * v1[jf, ie])
                   * (top[p + 1, ie] - top[p, ie])) * 0.5 * g.dyi)
    return out


def _random_fields(seed=7):
    g = Grid()
    rng = np.random.default_rng(seed)
    u1 = rng.normal(0, 5, (g.ny, g.nx))
    u0 = rng.normal(0, 5, (g.ny, g.nx))
    v1 = rng.normal(0, 3, (g.ny + 1, g.nx))
    v0 = rng.normal(0, 3, (g.ny + 1, g.nx))
    top = np.clip(rng.gamma(0.4, 0.15, (g.ny, g.nx)), 0, 0.6)
    top[top < 0.1] = 0.0                        # bndinit threshold
    return g, u1, v1, u0, v0, top


def test_v1st_interpolation():
    # top = 0 -> surface value V1s exactly
    assert _v1st_map(np.array([[0.0]]))[0, 0] == V1S
    # top = 0.35 -> we = 5.9: rows 5/6 of the table (1-based), frac 0.9
    got = _v1st_map(np.array([[0.35]]))[0, 0]
    v1z = V1Z_TABLE[:, 1]
    we = 0.35 * 14 + 1.0
    frac = we - 5.0
    assert got == (1.0 - frac) * v1z[4] + frac * v1z[5]


def test_div0_matches_fortran_loop_transcription():
    g, u1, v1, u0, v0, top = _random_fields()
    got = topo_div0(u1, v1, u0, v0, top, _v1st_map(top), g)
    ref = _div0_fortran_loops(u1, v1, u0, v0, top, g)
    np.testing.assert_array_equal(got[-1], 0.0)     # Fortran row ny inert
    np.testing.assert_allclose(got, ref, rtol=0, atol=0)   # bitwise


@_needs_data
def test_flat_topo_is_bitwise_inert():
    """TOPO on with zero topography == TOPO off, 2 days, every field."""
    from qtcm1.driver import ControlRun
    from qtcm1.io.bnddata import BoundaryData

    ref = BoundaryData(_DATA)
    flat = {'stype': ref.stype, 'top': np.zeros_like(ref.top)}

    ra = ControlRun(config=RunConfig(data_path=_DATA))
    rb = ControlRun(config=RunConfig(data_path=_DATA, topo=True),
                    surface=flat)
    for _ in range(2):
        ra.advance_day()
        rb.advance_day()
    for f in ['u1', 'v1', 'T1', 'q1', 'u0', 'v0', 'vort0', 'Ts', 'WD']:
        np.testing.assert_array_equal(getattr(ra.state, f),
                                      getattr(rb.state, f), err_msg=f)
    assert (rb.state.div0 == 0.0).all()


@_needs_data
def test_topo_run_is_active_and_stable():
    """Real topography: div0 lights up over the mountains, run stays sane."""
    from qtcm1.driver import ControlRun

    ra = ControlRun(config=RunConfig(data_path=_DATA))
    rb = ControlRun(config=RunConfig(data_path=_DATA, topo=True))
    for _ in range(3):
        ra.advance_day()
        rb.advance_day()
    d0 = rb.state.div0
    assert np.isfinite(d0).all()
    assert 1e-9 < np.abs(d0).max() < 1e-3          # ~ v.grad(TOP) scale
    # the effect reaches the barotropic flow
    assert np.abs(rb.state.vort0 - ra.state.vort0).max() > 0
    for f in ['u1', 'T1', 'q1', 'u0']:
        assert np.isfinite(getattr(rb.state, f)).all(), f


@_needs_data
def test_topo_restart_roundtrip(tmp_path):
    """TOPO runs restart bit-exactly (div0 carried in the file)."""
    from qtcm1.driver import ControlRun

    cfg = RunConfig(data_path=_DATA, topo=True)
    ra = ControlRun(config=cfg)
    for _ in range(3):
        ra.advance_day()

    rb = ControlRun(config=cfg)
    rb.advance_day()
    p = str(tmp_path / 'topo.restart.npz')
    rb.save_restart(p)
    rc = ControlRun.from_restart(p)
    assert rc.state.div0 is not None
    rc.advance_day()
    rc.advance_day()
    for f in ['u1', 'v1', 'T1', 'q1', 'u0', 'v0', 'vort0', 'div0']:
        np.testing.assert_array_equal(getattr(ra.state, f),
                                      getattr(rc.state, f), err_msg=f)


# ---------------------------------------------------------------------------
# gfortran kernel oracle: the actual Fortran expressions, double precision
# ---------------------------------------------------------------------------

_F90 = r"""
program topo_kernel
  implicit none
  integer, parameter :: nx=64, ny=42, nz=14
  real(8) :: u1(nx,ny), u0(nx,ny), v1(nx,0:ny), v0(nx,0:ny)
  real(8) :: top(nx,ny), div0(nx,ny), rhstopo(nx,ny)
  real(8) :: v1z(nz), fv(0:ny), dxvi(0:ny), dyi
  real(8) :: we, v1st
  integer :: i, j, kw, ip1(nx)
  open(10, file='in.bin', form='unformatted', access='stream')
  read(10) u1, u0, v1, v0, top, v1z, fv, dxvi, dyi
  close(10)
  do i=1,nx-1
     ip1(i)=i+1
  end do
  ip1(nx)=1
  div0=0d0
  rhstopo=0d0
  do j=1,ny-1
     do i=1,nx
        we=top(i,j)*nz+1d0
        kw=we
        we=we-kw
        v1st=(1d0-we)*v1z(kw)+we*v1z(kw+1)
        div0(i,j)= &
             ((u0(i,j)+v1st*u1(i,j))*(top(ip1(i),j)-top(i,j)) &
             +(u0(i,j+1)+v1st*u1(i,j+1))*(top(ip1(i),j+1)-top(i,j+1))) &
             *0.5d0*dxvi(j) &
             +((v0(i,j)+v1st*v1(i,j))*(top(i,j+1)-top(i,j)) &
             +(v0(ip1(i),j)+v1st*v1(ip1(i),j))*(top(ip1(i),j+1) &
             -top(ip1(i),j)))*0.5d0*dyi
        rhstopo(i,j)=-fv(j)*div0(i,j)
     end do
  end do
  open(11, file='out.bin', form='unformatted', access='stream')
  write(11) div0, rhstopo
  close(11)
end program topo_kernel
"""


@pytest.mark.skipif(shutil.which('gfortran') is None,
                    reason='gfortran not available')
def test_div0_matches_gfortran_kernel(tmp_path):
    g, u1, v1, u0, v0, top = _random_fields(seed=11)
    src = tmp_path / 'topo_kernel.f90'
    src.write_text(_F90)
    exe = tmp_path / 'topo_kernel'
    subprocess.run(['gfortran', '-O0', '-ffp-contract=off',
                    str(src), '-o', str(exe)], check=True)

    fv = np.asarray(g.fv, dtype=np.float64)
    dxvi = np.asarray(g.dxvi, dtype=np.float64)
    with open(tmp_path / 'in.bin', 'wb') as f:
        # Fortran (nx, ny) column-major == Python (ny, nx) C-order bytes;
        # v arrays are (nx, 0:ny) == Python (ny+1, nx)
        for arr in (u1, u0, v1, v0, top):
            f.write(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(V1Z_TABLE[:, 1]).tobytes())
        f.write(fv.tobytes())
        f.write(dxvi.tobytes())
        f.write(np.float64(g.dyi).tobytes())
    subprocess.run([str(exe)], cwd=tmp_path, check=True)
    out = np.fromfile(tmp_path / 'out.bin', dtype=np.float64)
    div0_f = out[: 42 * 64].reshape(42, 64)
    rhstopo_f = out[42 * 64:].reshape(42, 64)

    div0_py = topo_div0(u1, v1, u0, v0, top, _v1st_map(top), g)
    np.testing.assert_array_equal(div0_py, div0_f)
    # the vorticity-equation term as the model applies it
    rhs_py = np.zeros_like(div0_py)
    rhs_py[:-1] = -fv[1:g.ny, None] * div0_py[:-1]
    np.testing.assert_array_equal(rhs_py, rhstopo_f)
