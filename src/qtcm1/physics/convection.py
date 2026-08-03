"""Betts-Miller convection projected on mode 1: port of ``mconvct``.

Includes the two lookup tables the Fortran builds at init (utilities.F90):

* ``humtable``/``hsat``: saturation specific humidity at 1000 mb every 0.1 K
  over 200-400 K, from the Shuttleworth/Tetens formula ``hsat0``; queries
  interpolate linearly in the table (the table discretization is part of the
  model's numerics, so ``hsat`` here goes through the same table, never the
  closed-form formula directly).
* ``t1ctable``/``nlt1c``: the nonlinear convective closure. For each
  candidate ``T1c`` in [-300, 300] K (1-K steps), the column-projected
  moisture ``qcphat(T1c)`` is a trapezoid integral of
  ``alpha_k * hsat(Tcref_k + a1_k*T1c) * L/Cp * 1e3/p_k`` over the T1c-table
  pressure levels, and ``rhsx = a1hat*T1c + qcphat``; ``nlt1c`` inverts this
  monotone relation by linear interpolation (NZ eqs. 2.20-2.23 context).

``mconvct`` then computes projected CAPE and the convective heating
Qc = eps_c * max(CAPE1, 0) * Cpg  [W m-2], applies the polar filter, and
clips negatives introduced by the filter. The ``LINEAR_T1C`` compile option
is the ``linear_closure`` flag.

Table precision
---------------
The stored table entries carry the ``Real`` kind of the Fortran build being
mirrored: :class:`ConvectionTables` built with ``dtype=float32`` matches the
standard single-precision build, ``float64`` (default) a double-precision
build and production. The distinction is measurable: a 1-ulp float32 table
difference (~3e-5 K in T1c) is amplified by ``eps_c*Cpg`` (~1.2e3 W m-2 K-1)
into ~0.04 W m-2 of coherent Qc offset. Query arithmetic is float64 in both
cases and replicates the Fortran indexing (``w = 1 + (T-hTmin)*hdTi``).
"""

from __future__ import annotations

import numpy as np

from ..constants import (A1HAT, B1HAT, BB1HAT, CP, CPG, HLATENT,
                         QCREFHAT, QREFHAT, T1C_TABLE, TCREFHAT, TREFHAT)

_HTMIN, _HTMAX, _HDT = 200.0, 400.0, 0.1
_NT = int((_HTMAX - _HTMIN) / _HDT) + 1            # 2001 entries


def _hsat0(T):
    """Saturation specific humidity at 1000 mb [kg/kg] (``hsat0``)."""
    Tc = T - 273.15
    esat = 0.6108 * np.exp(17.270 * Tc / (237.3 + Tc))   # kPa
    return 0.622 * esat * 0.01


class ConvectionBlowup(RuntimeError):
    """T1c left the [-300, 300] K table range (the Fortran stops here)."""


class ConvectionTables:
    """hsat + T1c lookup tables at a given build precision (see module doc)."""

    def __init__(self, dtype=np.float64):
        self.dtype = np.dtype(dtype)
        ft = self.dtype.type
        # humtable: T = hTmin + hdT*(k-1) and hsat0 evaluated at the build's
        # kind; stored float64 for the (always-float64) query arithmetic.
        T = ft(_HTMIN) + ft(_HDT) * np.arange(_NT, dtype=ft)
        self.hums = _hsat0(T).astype(ft).astype(np.float64)
        self.t1cs, self.rhsx = self._build_t1c_table()

    # -- hsat --------------------------------------------------------------
    def hsat(self, T):
        """Table-interpolated saturation humidity (port of ``hsat``).

        Replicates the Fortran clamps (T<200 -> 200; T>400 -> 400-hdT) and
        index arithmetic ``w = 1 + (T-hTmin)*hdTi`` with ``hdTi = 1/hdT``.
        """
        hdti = 1.0 / np.float64(_HDT)              # exactly 10.0
        T = np.asarray(T, dtype=np.float64)
        T = np.where(T < _HTMIN, _HTMIN, T)
        T = np.where(T > _HTMAX, _HTMAX - _HDT, T)
        w = 1.0 + (T - _HTMIN) * hdti
        k = w.astype(np.int64)                     # Fortran k=w truncation
        k = np.minimum(k, _NT - 1)                 # safety at T ~ hTmax
        w = w - k
        return (1.0 - w) * self.hums[k - 1] + w * self.hums[k]

    # -- t1ctable ----------------------------------------------------------
    def _build_t1c_table(self):
        T1cs = -300.0 + np.arange(601, dtype=np.float64)
        prs = T1C_TABLE[:, 0]
        alpha = T1C_TABLE[:, 1]
        Tcref = T1C_TABLE[:, 2]
        a1 = T1C_TABLE[:, 3]
        cpi = 1.0 / CP                             # Fortran Cpi=1./Cp
        # qcp: (601, np) through the hsat table, Fortran operand order
        qcp = (alpha[None, :]
               * self.hsat(Tcref[None, :] + a1[None, :] * T1cs[:, None])
               * HLATENT * cpi * 1.0e3 / prs[None, :])
        # trapezoid accumulated in the Fortran's sequential k order
        qcphat = np.zeros(601, dtype=np.float64)
        for k in range(len(prs) - 1):
            qcphat = qcphat + (qcp[:, k] + qcp[:, k + 1]) * 0.5 * (prs[k]
                                                                   - prs[k + 1])
        qcphat = qcphat / (prs[0] - prs[-1])
        return T1cs, A1HAT * T1cs + qcphat

    def nlt1c(self, x):
        """Invert the nonlinear closure: T1c(x) (port of ``nlt1c``)."""
        x = np.asarray(x, dtype=np.float64)
        if (x < self.rhsx[0]).any() or (x > self.rhsx[-1]).any():
            raise ConvectionBlowup(
                f'T1c out of table range: x in [{x.min():.2f}, {x.max():.2f}]'
                f', table rhs in [{self.rhsx[0]:.2f}, {self.rhsx[-1]:.2f}]')
        return np.interp(x, self.rhsx, self.t1cs)


_TABLES: dict = {}


def get_tables(dtype=np.float64) -> ConvectionTables:
    """Shared :class:`ConvectionTables` instance for ``dtype`` (cached)."""
    key = np.dtype(dtype)
    if key not in _TABLES:
        _TABLES[key] = ConvectionTables(key)
    return _TABLES[key]


def hsat(T, tables: ConvectionTables | None = None):
    """Module-level convenience wrapper (float64 tables by default)."""
    return (tables or get_tables()).hsat(T)


def nlt1c(x, tables: ConvectionTables | None = None):
    """Module-level convenience wrapper (float64 tables by default)."""
    return (tables or get_tables()).nlt1c(x)


# -------------------------------------------------------------- mconvct
def mconvct(T1, q1, eps_c, polar_filter, *, linear_closure: bool = False,
            tables: ConvectionTables | None = None):
    """Convective heating / precipitation Qc [W m-2] (port of ``mconvct``).

    Parameters: T1, q1 on T rows (ny, nx); ``eps_c`` = 1/tau_c [s-1];
    ``polar_filter`` a :class:`~qtcm1.dynamics.filters.PolarFilter`;
    ``tables`` a :class:`ConvectionTables` (float64 build if omitted).
    Returns dict with Qc and the diagnostic T1c.
    """
    tables = tables or get_tables()
    dTrefhat = TCREFHAT - TREFHAT
    dqrefhat = QCREFHAT - QREFHAT

    if linear_closure:                      # LINEAR_T1C (v2.1 closure)
        T1c = ((A1HAT * T1 + B1HAT * q1 - dTrefhat - dqrefhat)
               / (A1HAT + BB1HAT))
    else:                                   # default nonlinear closure
        x = TREFHAT - TCREFHAT + A1HAT * T1 + QREFHAT + B1HAT * q1
        T1c = tables.nlt1c(x)

    CAPE1 = np.maximum(A1HAT * (T1c - T1) + dTrefhat, 0.0)
    Qc = eps_c * CAPE1 * CPG
    Qc = polar_filter(Qc)
    Qc = np.maximum(Qc, 0.0)
    return dict(Qc=Qc, T1c=T1c)
