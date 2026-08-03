"""Polar zonal filter: port of ``xfilter`` (qtcm.F90).

Arakawa/Lamb (1977) low-pass filter applied poleward of 60 degrees: zonal
wavenumber ``m`` of each filtered row is damped by

    min( 0.9 * sqrt(cos(lat)) / sin(m*pi/nx), 1 )

(no damping of the zonal mean). The Fortran multiplies FFTPACK's packed real
coefficients; both quadrature components of a wavenumber get the same
factor, so operating on the complex rfft coefficients is identical.

The filtered-row extent is computed with the Fortran's own single-precision
expression ``js = int((1. - 60./YB)*ny/2)`` to reproduce its truncation
behavior exactly.
"""

from __future__ import annotations

import numpy as np


class PolarFilter:
    """Precomputed filter factors for a given grid (first-call setup)."""

    def __init__(self, grid, lat_cutoff: float = 60.0, reduction: float = 0.9):
        nx, ny = grid.nx, grid.ny
        # Fortran: js=(1.-60./YB)*ny/2 evaluated in REAL*4, truncated to int
        f32 = np.float32
        js = int(f32(f32(f32(1.0) - f32(lat_cutoff) / f32(grid.YB))
                     * f32(ny)) / f32(2.0))
        self.js = js                               # rows 0..js-1 (south)
        self.jn0 = ny - js                         # rows jn0..ny-1 (north)
        m = np.arange(1, nx // 2 + 1)
        coslat = np.cos(np.deg2rad(np.asarray(grid.latt, dtype=np.float64)))
        fac = (reduction * np.sqrt(coslat)[:, None]
               / np.sin(m * np.pi / nx)[None, :])
        self.factors = np.minimum(fac, 1.0)        # (ny, nx//2), per T row

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
