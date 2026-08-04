# pyqtcm1

[![Documentation Status](https://readthedocs.org/projects/pyqtcm1/badge/?version=latest)](https://pyqtcm1.readthedocs.io/en/latest/)

Pure-Python reimplementation of the Neelin–Zeng Quasi-Equilibrium Tropical
Circulation Model, version 1 (QTCM1 v2.3), migrated from the Fortran core of
J. W.-B. Lin's `qtcm` 0.1.2 package.

**Documentation: [pyqtcm1.readthedocs.io](https://pyqtcm1.readthedocs.io)** —
quickstart, example gallery (fixed-SST control, SST anomalies, idealized
greenhouse forcing, restarts/twins), the validation methodology, and the API
reference.

The complete atmospheric model runs cold-started and self-contained (netCDF
boundary data, no Fortran anywhere in the loop), and is validated against
the original at three tiers:

- **per-routine golden tests** vs an instrumented double-precision build of
  the Fortran: ≤ 2×10⁻¹⁴ relative on every routine;
- **trajectory shadowing**: 30-day free runs track the bit-deterministic
  Fortran at accumulated float64 roundoff (RMS T₁ 9×10⁻¹³ K after 2160
  steps), from warm and cold starts;
- **climate statistics**: a 10-year control is indistinguishable from a
  Fortran control member at the model's internal-variability level on all
  archived fields.

Beyond parity, the port adds bit-exact restarts, declarative run
configuration with provenance stamping (`build='f64'` recommended /
`'f32'` heritage — the two Fortran precision builds are measurably
different models), and optional Numba kernels that are provably
bit-identical to the NumPy reference path (~6× the Fortran's runtime,
from 9× unoptimized).

Conventions: arrays are C-ordered `(lat, lon)` = Fortran `(ny, nx)`
transposed; scientific field names keep the paper notation (`u1, T1, q1`, …);
every ported function's docstring names its Fortran origin and the Neelin &
Zeng (2000) equations it implements. Boundary inputs are netCDF only,
shipped in `data/r64x42` (the repo is self-contained; `tools/convert_bnddata.py` regenerates it from the original ASCII `bnddir`).

Not yet ported from the original option set: the mixed-layer/slab ocean,
topography, and the ISCCP cloud option — see the roadmap in the docs.
