"""Polar zonal filter: port of ``xfilter`` (qtcm.F90).

Arakawa/Lamb (1977) low-pass filter applied poleward of 60 degrees: zonal
wavenumber ``m`` of each filtered row is damped by

    min( 0.9 * sqrt(cos(lat)) / sin(m*pi/nx), 1 )

(no damping of the zonal mean). The Fortran multiplies FFTPACK's packed real
coefficients; both quadrature components of a wavenumber get the same
factor, so operating on the complex rfft coefficients is identical.

Precision of the *setup* computation matters, because the number of
filtered rows sits on a truncation knife edge.  The Fortran computes

    js = (1. - 60./YB) * ny / 2        ! rows 1..js filtered (per pole)

whose exact value for the standard grid (YB=78.75, ny=42) is exactly 5;
rounded arithmetic lands on either side of it:

* REAL*4 build:  4.99999952          -> js = 4  (under-filters one row)
* REAL*8 build:  5.000000000000001   -> js = 5  (the intended extent:
  row 5 is at 61.875 deg, poleward of the 60-deg cutoff)

``dtype`` selects which build is mirrored; float64 (default) matches a
double-precision build and is the production choice.  The damping factors
are evaluated at the same precision (``ara`` shares the build's ``Real``
kind); a 1e-7 relative factor difference applied 72x/day compounds into a
measurable coherent polar-mode drift, so like must be compared with like.
"""

from __future__ import annotations

import numpy as np


class PolarFilter:
    """Precomputed filter factors for a given grid (first-call setup).

    Parameters
    ----------
    grid : Grid
    lat_cutoff, reduction : the Fortran's hardwired 60 deg / 0.9.
    dtype : precision of the mirrored Fortran build's setup arithmetic
        (row extent ``js`` and factor table). float64 = double-precision
        build (production default); float32 = the standard REAL*4 build.
        Application arithmetic is always float64.
    """

    def __init__(self, grid, lat_cutoff: float = 60.0, reduction: float = 0.9,
                 dtype=np.float64):
        nx, ny = grid.nx, grid.ny
        ft = np.dtype(dtype).type
        self.dtype = np.dtype(dtype)
        # Fortran: js=(1.-60./YB)*ny/2, evaluated in the build's Real kind,
        # truncated on integer assignment.
        js = int(ft(ft(ft(1.0) - ft(lat_cutoff) / ft(grid.YB))
                    * ft(ny)) / ft(2.0))
        self.js = js                               # rows 0..js-1 (south)
        self.jn0 = ny - js                         # rows jn0..ny-1 (north)
        # Factor table, replicating the Fortran chain at the build's kind
        # (pi = asin(1.)*2. as in parinit).
        pi = ft(2.0) * np.arcsin(ft(1.0))
        m = np.arange(1, nx // 2 + 1, dtype=np.int64)
        coslat = np.cos(ft(ft(np.asarray(grid.latt, dtype=ft))
                           * pi / ft(180.0)))
        fac = (ft(reduction) * np.sqrt(coslat, dtype=ft)[:, None]
               / np.sin(ft(m[None, :] * pi / ft(nx)), dtype=ft))
        self.factors = np.minimum(fac.astype(np.float64), 1.0)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """Filter high zonal wavenumbers on the polar rows of ``field``.

        ``field`` is (ny, nx) on T rows; returns a filtered copy.
        """
        out = np.array(field, dtype=np.float64, copy=True)
        for rows in (slice(0, self.js), slice(self.jn0, None)):
            if out[rows].shape[0] == 0:
                continue
            c = np.fft.rfft(out[rows], axis=1)
            c[:, 1:] *= self.factors[rows]
            out[rows] = np.fft.irfft(c, n=out.shape[1], axis=1)
        return out
