"""Vertical basis functions of the QTCM1 Galerkin expansion.

QTCM1's three-dimensional fields are single-mode Galerkin expansions in
pressure (Neelin & Zeng 2000, hereafter NZ):

.. math::

   \\mathbf{v}(x,y,p,t) = \\mathbf{v}_0(x,y,t)
       + V_1(p)\\,\\mathbf{v}_1(x,y,t), \\qquad
   T = T_r(p) + a_1(p)\\,T_1, \\qquad
   q = q_r(p) + b_1(p)\\,q_1,

with the velocity basis derived from the temperature basis (NZ 3.9-3.10):

.. math::

   V_1(p) = a_1^+(p) - \\widehat{a_1^+}, \\qquad
   a_1^+(p) = \\int_p^{p_{rs}} a_1(p')\\,d\\ln p' ,

so that full profiles are reconstructions from the prognostic mode
amplitudes: e.g. zonal wind ``u(p) = u0 + V1(p) * u1``.

:func:`load_basis` returns the profiles as an :class:`xarray.Dataset`
built from the tables ported verbatim from ``qtcmpar.F90``:

* :data:`qtcm1.constants.T1C_TABLE` - :math:`a_1(p)` (with the closure
  quantities ``alpha`` and ``Tsat``) on 11 pressure levels, 1000-200 hPa;
* :data:`qtcm1.constants.V1Z_TABLE` - :math:`V_1(z)` on 14 height levels
  (0-9 km), the table the ABL surface-wind scheme interpolates.

:math:`a_1^+` is computed as the *exact* integral of the tabulated
:math:`a_1`, treated (like the closure interpolation) as piecewise linear
in :math:`\\ln p`; requesting more ``levels`` refines the sampling, not
the quadrature, so values at the table nodes are level-set independent.

Two footnotes worth knowing before leaning on this module:

* the moisture basis :math:`b_1(p)` is **not available**: v2.3 tabulates
  only its projections (``b1hat``, ``bb1hat``, ``b1s``), which is all the
  discrete equations need, so moisture *profiles* cannot be reconstructed
  from package data;
* the subtraction constant :math:`\\widehat{a_1^+}` is the model's own
  ``a1phat`` = 0.24527773 (NZ's reference-profile projection), not a
  re-quadrature of the 11-level table - this makes ``V1`` at 1000 hPa
  equal the model's ``V1s`` exactly. The table's own trapezoidal
  pressure-means reproduce ``a1phat`` to 2.4% and ``a1hat`` to 3.8%
  (the 11-level discretization error), a useful scale for how literally
  to read reconstructed profiles between table nodes.
"""

from __future__ import annotations

import numpy as np

from .constants import (A1HAT, A1PHAT, A1S, B1HAT, B1S, BB1HAT, T1C_TABLE,
                        V1S, V1SQHAT, V1Z_TABLE)

__all__ = ['load_basis']

#: reference surface pressure of the expansion, p_rs [hPa] (NZ 3.9)
PRS = 1000.0
#: nominal tropopause of the expansion, p_rt [hPa] (p_rs - delp)
PRT = 200.0


def _a1_plus(p: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """Cumulative :math:`\\int_p^{p_{rs}} a_1 d\\ln p'` on descending ``p``.

    Trapezoidal cumulative sum in :math:`\\ln p` - exact for the
    piecewise-linear-in-:math:`\\ln p` tabulated ``a1``.
    """
    lp = np.log(p)
    out = np.zeros_like(a1)
    incr = 0.5 * (a1[:-1] + a1[1:]) * (lp[:-1] - lp[1:])
    out[1:] = np.cumsum(incr)
    return out


def load_basis(levels=None):
    """The vertical basis functions as an :class:`xarray.Dataset`.

    Parameters
    ----------
    levels:
        Pressure levels [hPa] for the ``p``-space profiles.

        * ``None`` (default) - the 11 native table levels, 1000-200 hPa;
        * an ``int`` ``n`` - ``n`` levels evenly spaced in :math:`\\ln p`
          from 1000 to 200 hPa (handy for smooth plots);
        * an array of pressures inside [200, 1000] hPa (no extrapolation:
          the ``qtcmpar.F90`` tables end at those bounds).

    Returns
    -------
    xarray.Dataset
        ``a1(p)``, ``a1p(p)`` (:math:`a_1^+`), ``V1(p)``, the closure
        table columns ``alpha(p)`` and ``Tsat(p)``, and ``V1z(z)`` (the
        ABL height-level table). The scalar projections (``a1hat``,
        ``a1phat``, ``V1s``, ``b1hat``, ...) ride along as attributes.

    Examples
    --------
    Reconstruct zonal-mean zonal wind on 41 levels (NZ 3.10)::

        basis = load_basis(levels=41)
        u = ds['u0'].mean('lon') + basis['V1'] * ds['u1'].mean('lon')

    ``u`` broadcasts to dims ``(p, lat)`` - see
    ``examples/06_zonal_mean_winds.py``.
    """
    import xarray as xr                      # deferred, like io.output

    p_tab = T1C_TABLE[:, 0]                  # 1000 .. 200, descending
    if levels is None:
        p_req = p_tab.copy()
    elif np.ndim(levels) == 0:
        n = int(levels)
        if n < 2:
            raise ValueError('levels as an int must be >= 2')
        p_req = np.exp(np.linspace(np.log(PRS), np.log(PRT), n))
        p_req[0], p_req[-1] = PRS, PRT       # exact endpoints
    else:
        p_req = np.asarray(levels, dtype=np.float64)
        if p_req.ndim != 1 or p_req.size == 0:
            raise ValueError('levels must be a 1-D, non-empty array')
        if p_req.min() < PRT or p_req.max() > PRS:
            raise ValueError(
                f'levels must lie inside [{PRT:.0f}, {PRS:.0f}] hPa: the '
                f'qtcmpar.F90 tables do not extend beyond the troposphere')
        p_req = np.sort(p_req)[::-1]         # descending, like the table

    # Integrate once on the native nodes; evaluate off-node points by the
    # closed-form partial-segment integral of the piecewise-linear a1.
    # Node values are then bitwise identical for every requested level set.
    a1_tab = T1C_TABLE[:, 3]
    lp_tab = np.log(p_tab)
    a1p_tab = _a1_plus(p_tab, a1_tab)

    lp_req = np.log(p_req)
    # segment index i: p_req in [p_tab[i+1], p_tab[i]] (descending table)
    i = np.clip(np.searchsorted(-p_tab, -p_req, side='right') - 1,
                0, p_tab.size - 2)
    frac = (lp_tab[i] - lp_req) / (lp_tab[i] - lp_tab[i + 1])
    at_right = frac == 1.0                   # only the p = prt endpoint
    a1_req = np.where(at_right, a1_tab[i + 1],
                      a1_tab[i] + (a1_tab[i + 1] - a1_tab[i]) * frac)
    dx = lp_tab[i] - lp_req                  # 0 exactly at nodes
    a1p_req = np.where(at_right, a1p_tab[i + 1],
                       a1p_tab[i] + 0.5 * (a1_tab[i] + a1_req) * dx)

    interp = lambda col: np.where(at_right, col[i + 1],
                                  col[i] + (col[i + 1] - col[i]) * frac)

    def var(values, units, long_name, ref):
        return xr.DataArray(values, dims='p', attrs=dict(
            units=units, long_name=long_name, reference=ref))

    ds = xr.Dataset(
        {
            'a1': var(a1_req, '1', 'temperature basis function',
                      'NZ (3.2); qtcmpar.F90 T1cTableIn col 4'),
            'a1p': var(a1p_req, '1',
                       'a1+ = integral_p^prs a1 dlnp', 'NZ (3.9)'),
            'V1': var(a1p_req - A1PHAT, '1',
                      'velocity basis function, a1+ - a1phat', 'NZ (3.10)'),
            'alpha': var(interp(T1C_TABLE[:, 1]), '1',
                         'convective-closure weighting profile',
                         'qtcmpar.F90 T1cTableIn col 2'),
            'Tsat': var(interp(T1C_TABLE[:, 2]), 'K',
                        'convective-closure saturation profile',
                        'qtcmpar.F90 T1cTableIn col 3'),
            'V1z': xr.DataArray(
                V1Z_TABLE[:, 1], dims='z', attrs=dict(
                    units='1',
                    long_name='velocity basis function on height levels',
                    reference='qtcmpar.F90 V1(z) table (ABL scheme)')),
        },
        coords={
            'p': ('p', p_req, dict(units='hPa', long_name='pressure',
                                   positive='down')),
            'z': ('z', V1Z_TABLE[:, 0], dict(units='m', long_name='height')),
        },
        attrs=dict(
            title='QTCM1 vertical basis functions (Neelin & Zeng 2000)',
            a1hat=A1HAT, a1phat=A1PHAT, a1s=A1S, V1s=V1S, V1sqhat=V1SQHAT,
            b1hat=B1HAT, bb1hat=BB1HAT, b1s=B1S, prs_hPa=PRS, prt_hPa=PRT,
            note=('b1(p) is not tabulated in QTCM1 v2.3 (only its '
                  'projections b1hat/bb1hat/b1s enter the equations), so '
                  'moisture profiles cannot be reconstructed'),
        ),
    )
    return ds
