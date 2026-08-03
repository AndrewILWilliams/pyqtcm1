"""Model driver: assembles the ported routines into QTCM1 time steps.

Reproduces the Fortran ``atm_step`` call order exactly (qtcm.F90 ``qtcm``):

    physics1 (mconvct -> cloud -> radsw -> radlw -> sflux) -> sland1 ->
    advctuv -> advctTq -> dffus -> barcl -> [savebartr -> bartr -> gradphis]

with the barotropic group every ``mt0`` steps. The daily coupling update
(``getbnd``/``ocean``) is :meth:`Model.apply_boundary`: prescribed SST onto
ocean points and the interpolated albedo.

State is a plain dataclass of arrays (float64); everything the Fortran
carries across steps is explicit here: the ABL warm-start winds (us, vs),
the geopotential gradients used by the *next* step's surface fluxes, and
the AB3 histories of the barotropic mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .dynamics.advection import advctTq, advctuv
from .dynamics.baroclinic import barcl
from .dynamics.barotropic import bartr, gradphis, savebartr
from .dynamics.diffusion import dffus
from .dynamics.elliptic import PoissonDirichlet
from .dynamics.filters import PolarFilter
from .grid import Grid
from .physics.clouds import cloud
from .physics.convection import get_tables, mconvct
from .physics.land import WD0, sland1
from .physics.radiation import radlw, radsw
from .physics.sfcflux import sfcwind_abl, sflux, v1interpol

#: default runtime parameters (defaults.py of the wrapped model / Input mod).
#: eps_c is the package's decimal literal (nominal 1/7200 s-1 = 2-h tau_c);
#: writing 1.0/7200.0 instead differs by 8e-9 relative and shows up as a
#: coherent 1e-9 offset in cold-start races against the oracle.
DEFAULT_PARAMS = dict(
    dt=1200.0, mt0=1, eps_c=0.00013888889,
    viscxu0=7.0e5, viscyu0=7.0e5, viscxu1=7.0e5, viscyu1=7.0e5,
    visc4x=7.0e5, visc4y=7.0e5, viscxT=12.0e5, viscyT=12.0e5,
    viscxq=12.0e5, viscyq=12.0e5,
    weml=0.01, ziml=500.0, vvsmin=4.5,
)


@dataclass
class ModelState:
    """Everything QTCM1 carries from one step to the next."""

    u1: np.ndarray; v1: np.ndarray; T1: np.ndarray; q1: np.ndarray
    u0: np.ndarray; v0: np.ndarray
    vort0: np.ndarray; u0bar: float
    rhs_hist: list                       # [rhs_{n-1}, rhs_{n-2}] (ny-1, nx)
    rhsbar_hist: list                    # [float, float]
    Ts: np.ndarray; WD: np.ndarray
    us: np.ndarray; vs: np.ndarray       # ABL warm start
    dphisdx: np.ndarray; dphisdy: np.ndarray
    psi0: np.ndarray | None = None


def _q32(a):
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def _quantize_state(s: ModelState) -> ModelState:
    return replace(
        s, u1=_q32(s.u1), v1=_q32(s.v1), T1=_q32(s.T1), q1=_q32(s.q1),
        u0=_q32(s.u0), v0=_q32(s.v0), vort0=_q32(s.vort0),
        u0bar=float(np.float32(s.u0bar)),
        rhs_hist=[_q32(r) for r in s.rhs_hist],
        rhsbar_hist=[float(np.float32(r)) for r in s.rhsbar_hist],
        Ts=_q32(s.Ts), WD=_q32(s.WD), us=_q32(s.us), vs=_q32(s.vs),
        dphisdx=_q32(s.dphisdx), dphisdy=_q32(s.dphisdy),
        psi0=None if s.psi0 is None else _q32(s.psi0))


class Model:
    """QTCM1 stepping engine (standard configuration)."""

    def __init__(self, stype, cdn, grid: Grid | None = None, params=None,
                 quantize32: bool = False, init_dtype=np.float64,
                 first_gradphis_skip: bool = False):
        #: quantize32 emulates the Fortran's float32 state storage: the
        #: state is rounded to float32 after every step (arithmetic stays
        #: float64). Used for Tier-2 trajectory comparisons against the
        #: single-precision oracle; leave False for production runs.
        self.quantize32 = quantize32
        #: init_dtype is the precision at which the mirrored Fortran build
        #: computes its *init-time constants*: the polar-filter row extent
        #: and factors, and the hsat/T1c lookup tables. float64 (default)
        #: matches a double-precision build and is the production choice;
        #: float32 mirrors the standard REAL*4 build (note it filters one
        #: polar row fewer -- see dynamics/filters.py).
        self.init_dtype = np.dtype(init_dtype)
        self.grid = grid or Grid()
        self.params = dict(DEFAULT_PARAMS, **(params or {}))
        self.stype = np.asarray(stype)
        self.cdn = np.asarray(cdn, dtype=np.float64)
        self.pfilter = PolarFilter(self.grid, dtype=self.init_dtype)
        self.tables = get_tables(self.init_dtype)
        self.poisson = PoissonDirichlet(self.grid)
        self.v1b = v1interpol(self.params['ziml'])
        self.fu = np.asarray(self.grid.fu, dtype=np.float64)
        #: Fortran gradphis returns early on its very first call (its
        #: firstcall block only sets constants), so a cold-started run
        #: keeps dphisdx/dphisdy = 0 through the first barotropic group.
        #: Leave False when warm-starting from a captured oracle state.
        self._skip_gradphis = bool(first_gradphis_skip)

    # ------------------------------------------------------------------
    def cold_start(self) -> ModelState:
        """Fortran ``varinit`` cold start (day0 pinned to 1 upstream)."""
        g = self.grid
        z = lambda: np.zeros((g.ny, g.nx))
        zv = lambda: np.zeros((g.ny + 1, g.nx))
        return ModelState(
            u1=z(), v1=zv(), T1=np.full((g.ny, g.nx), -100.0),
            q1=np.full((g.ny, g.nx), -50.0),
            u0=z(), v0=zv(), vort0=z(), u0bar=0.0,
            rhs_hist=[np.zeros((g.ny - 1, g.nx)), np.zeros((g.ny - 1, g.nx))],
            rhsbar_hist=[0.0, 0.0],
            Ts=np.full((g.ny, g.nx), 295.0),
            WD=0.7 * WD0[self.stype.astype(int)],
            us=z(), vs=z(), dphisdx=z(), dphisdy=zv())

    def apply_boundary(self, state: ModelState, sst, albedo) -> np.ndarray:
        """``getbnd``: prescribed SST onto ocean points; returns albedo."""
        ocean = self.stype == 0
        state.Ts = np.where(ocean, sst, state.Ts)
        return np.asarray(albedo, dtype=np.float64)

    # ------------------------------------------------------------------
    def step(self, s: ModelState, albedo, dayofyear: int,
             it: int = 1) -> tuple[ModelState, dict]:
        """One atmospheric time step; returns (new state, diagnostics)."""
        p = self.params
        dt = p['dt']

        # -- physics1 ---------------------------------------------------
        conv = mconvct(s.T1, s.q1, p['eps_c'], self.pfilter,
                       tables=self.tables)
        Qc = conv['Qc']
        cl = cloud(Qc)
        sw = radsw(cl['cld'], albedo, dayofyear, self.grid)
        lw = radlw(s.T1, s.q1, s.Ts, cl['cld'])
        wind = sfcwind_abl(s.u1, s.v1, s.u0, s.v0, s.us, s.vs,
                           s.dphisdx, s.dphisdy, self.cdn, self.fu,
                           weml=p['weml'], ziml=p['ziml'],
                           vvsmin=p['vvsmin'], v1b=self.v1b)
        fx = sflux(s.T1, s.q1, s.Ts, self.stype, self.cdn, wind,
                   tables=self.tables)

        # -- land -------------------------------------------------------
        land = sland1(s.Ts, s.WD, self.stype, Qc, fx['Evap'], fx['FTs'],
                      sw['FSWds'], sw['FSWus'], lw['FLWds'], lw['FLWus'],
                      fx['CV'], dt)

        # -- tendencies -------------------------------------------------
        adv = advctuv(s.u1, s.v1, s.u0, s.v0, self.grid)
        tq = advctTq(s.T1, s.q1, s.u1, s.v1, s.u0, s.v0, self.grid)
        df = dffus(s.u1, s.v1, s.u0, s.v0, s.T1, s.q1, self.grid,
                   viscxu1=p['viscxu1'], viscyu1=p['viscyu1'],
                   visc4x=p['visc4x'], visc4y=p['visc4y'],
                   viscxT=p['viscxT'], viscyT=p['viscyT'],
                   viscxq=p['viscxq'], viscyq=p['viscyq'],
                   viscxu0=p['viscxu0'], viscyu0=p['viscyu0'])

        # -- baroclinic mode -------------------------------------------
        bc = barcl(s.u1, s.v1, s.T1, s.q1,
                   taux=fx['taux'], tauy=fx['tauy'],
                   advu1=adv['advu1'], advv1=adv['advv1'],
                   advT1=tq['advT1'], advq1=tq['advq1'],
                   dfsu1=df['dfsu1'], dfsv1=df['dfsv1'],
                   dfsT1=df['dfsT1'], dfsq1=df['dfsq1'],
                   Qc=Qc, FSW=sw['FSW'], FLW=lw['FLW'], FTs=fx['FTs'],
                   Evap=land['Evap'], grid=self.grid,
                   polar_filter=self.pfilter, dt=dt)

        # -- barotropic mode (every mt0 steps) --------------------------
        new = replace(s, u1=bc['u1'], v1=bc['v1'], T1=bc['T1'], q1=bc['q1'],
                      Ts=land['Ts'], WD=land['WD'],
                      us=wind['us'], vs=wind['vs'])
        diags = dict(Qc=Qc, cl1=cl['cl1'], Evap=land['Evap'], FTs=fx['FTs'],
                     taux=fx['taux'], tauy=fx['tauy'], div1=bc['div1'],
                     S0=sw['S0'], FSWds=sw['FSWds'], OLR=lw['FLWut'],
                     Runf=land['Runf'], wet=land['wet'])
        if it % p['mt0'] == 0:
            sav = savebartr(s.u0, s.v0)
            bt = bartr(s.vort0, s.u0bar, s.v0, s.rhs_hist, s.rhsbar_hist,
                       taux=fx['taux'], tauy=fx['tauy'],
                       advu0=adv['advu0'], advv0=adv['advv0'],
                       dfsu0=df['dfsu0'], dfsv0=df['dfsv0'],
                       grid=self.grid, polar_filter=self.pfilter,
                       poisson=self.poisson, dt=dt, mt0=p['mt0'])
            new = replace(new, u0=bt['u0'], v0=bt['v0'], vort0=bt['vort0'],
                          u0bar=bt['u0bar'], psi0=bt['psi0'],
                          rhs_hist=bt['rhs_hist'],
                          rhsbar_hist=bt['rhsbar_hist'])
            if self._skip_gradphis:                # Fortran first-call return
                self._skip_gradphis = False
            else:
                gp = gradphis(bt['u0'], bt['v0'], sav['u0sav'],
                              sav['v0sav'], bc['T1'],
                              taux=fx['taux'], tauy=fx['tauy'],
                              advu0=adv['advu0'], advv0=adv['advv0'],
                              dfsu0=df['dfsu0'], dfsv0=df['dfsv0'],
                              grid=self.grid, dt=dt, mt0=p['mt0'])
                new = replace(new, dphisdx=gp['dphisdx'],
                              dphisdy=gp['dphisdy'])
                diags['ps'] = gp['ps']
        if self.quantize32:
            new = _quantize_state(new)
        return new, diags

    def step_day(self, s: ModelState, sst, albedo,
                 dayofyear: int) -> tuple[ModelState, dict]:
        """One coupling day: boundary update + 86400/dt atmospheric steps."""
        alb = self.apply_boundary(s, sst, albedo)
        nastep = int(round(86400.0 / self.params['dt']))
        for it in range(1, nastep + 1):
            s, diags = self.step(s, alb, dayofyear, it)
        return s, diags
