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

Tables are built in float32 to track the single-precision Fortran tables;
interpolation is float64.
"""

from __future__ import annotations

import numpy as np

from ..constants import (A1HAT, B1HAT, BB1HAT, CP, CPG, HLATENT,
                         QCREFHAT, QREFHAT, T1C_TABLE, TCREFHAT, TREFHAT)

# ------------------------------------------------------------- humtable
_HTMIN, _HTMAX, _HDT = 200.0, 400.0, 0.1
_NT = int((_HTMAX - _HTMIN) / _HDT) + 1            # 2001 entries


def _hsat0(T):
    """Saturation specific humidity at 1000 mb [kg/kg] (``hsat0``)."""
    Tc = T - 273.15
    esat = 0.6108 * np.exp(17.270 * Tc / (237.3 + Tc))   # kPa
    return 0.622 * esat * 0.01


_HUMS = _hsat0(np.float32(_HTMIN)
               + np.float32(_HDT) * np.arange(_NT, dtype=np.float32)
               ).astype(np.float32)


def hsat(T):
    """Table-interpolated saturation humidity (port of ``hsat``).

    Clamps below 200 K and above 400 K exactly as the Fortran (which uses
    the [400-dT, 400] segment for high T).
    """
    T = np.asarray(T, dtype=np.float64)
    T = np.clip(T, _HTMIN, _HTMAX - _HDT * 1e-6)
    w = (T - _HTMIN) / _HDT
    k = np.floor(w).astype(int)
    k = np.minimum(k, _NT - 2)
    w = w - k
    hums = _HUMS.astype(np.float64)
    return (1.0 - w) * hums[k] + w * hums[k + 1]


# ------------------------------------------------------------- t1ctable
def _build_t1c_table():
    T1cs = -300.0 + np.arange(601, dtype=np.float64)
    prs = T1C_TABLE[:, 0]
    alpha = T1C_TABLE[:, 1]
    Tcref = T1C_TABLE[:, 2]
    a1 = T1C_TABLE[:, 3]
    # qcp: (601, np) evaluated through the hsat table
    qcp = (alpha[None, :] * hsat(Tcref[None, :] + a1[None, :] * T1cs[:, None])
           * HLATENT / CP * 1.0e3 / prs[None, :])
    dp = prs[:-1] - prs[1:]
    qcphat = 0.5 * ((qcp[:, :-1] + qcp[:, 1:]) * dp).sum(axis=1)
    qcphat /= prs[0] - prs[-1]
    rhsx = A1HAT * T1cs + qcphat
    return T1cs, rhsx


_T1CS, _RHSX = _build_t1c_table()


class ConvectionBlowup(RuntimeError):
    """T1c left the [-300, 300] K table range (the Fortran stops here)."""


def nlt1c(x):
    """Invert the nonlinear closure: T1c(x) (port of ``nlt1c``)."""
    x = np.asarray(x, dtype=np.float64)
    if (x < _RHSX[0]).any() or (x > _RHSX[-1]).any():
        raise ConvectionBlowup(
            f'T1c out of table range: x in [{x.min():.2f}, {x.max():.2f}], '
            f'table rhs in [{_RHSX[0]:.2f}, {_RHSX[-1]:.2f}]')
    return np.interp(x, _RHSX, _T1CS)


# -------------------------------------------------------------- mconvct
def mconvct(T1, q1, eps_c, polar_filter, *, linear_closure: bool = False):
    """Convective heating / precipitation Qc [W m-2] (port of ``mconvct``).

    Parameters: T1, q1 on T rows (ny, nx); ``eps_c`` = 1/tau_c [s-1];
    ``polar_filter`` a :class:`~qtcm1.dynamics.filters.PolarFilter`.
    Returns dict with Qc and the diagnostic T1c.
    """
    dTrefhat = TCREFHAT - TREFHAT
    dqrefhat = QCREFHAT - QREFHAT

    if linear_closure:                      # LINEAR_T1C (v2.1 closure)
        T1c = ((A1HAT * T1 + B1HAT * q1 - dTrefhat - dqrefhat)
               / (A1HAT + BB1HAT))
    else:                                   # default nonlinear closure
        x = TREFHAT - TCREFHAT + A1HAT * T1 + QREFHAT + B1HAT * q1
        T1c = nlt1c(x)

    CAPE1 = np.maximum(A1HAT * (T1c - T1) + dTrefhat, 0.0)
    Qc = eps_c * CAPE1 * CPG
    Qc = polar_filter(Qc)
    Qc = np.maximum(Qc, 0.0)
    return dict(Qc=Qc, T1c=T1c)
