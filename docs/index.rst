pyqtcm1
=======

A pure-Python reimplementation of the Neelin–Zeng **Quasi-Equilibrium
Tropical Circulation Model** (QTCM1, v2.3): a single-baroclinic-mode
intermediate-complexity model of the tropical atmosphere with
Betts–Miller convection, simplified radiation, an interactive land
surface, and a barotropic mode — the model of Neelin & Zeng (2000) and
Zeng, Neelin & Chou (2000).

The port is validated against the original Fortran at three tiers, down
to bit level where that is meaningful (see :doc:`validation`):

* every routine matches a double-precision build of the Fortran at
  ~:math:`10^{-14}` relative on captured golden states;
* free 30-day integrations shadow the (bit-deterministic) Fortran at
  accumulated float64 roundoff — RMS :math:`T_1` difference
  :math:`9\times10^{-13}` K after 2160 time steps;
* a 10-year control climatology is statistically indistinguishable from
  a Fortran control member at the model's own internal-variability
  level, for all twelve archived fields.

Compared to the original, the port adds bit-exact restarts, a declarative
run configuration with provenance stamping, netCDF boundary data and
output, and optional Numba acceleration that is provably bit-identical
to the NumPy reference path.

.. toctree::
   :maxdepth: 2

   quickstart
   output
   examples
   validation
   api
