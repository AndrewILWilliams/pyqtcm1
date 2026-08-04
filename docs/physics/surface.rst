Surface fluxes and ABL winds
============================

Module: :mod:`qtcm1.physics.sfcflux` (Fortran ``Sflux`` + ``abl.F90``).

Bulk fluxes
-----------

Evaporation and sensible heat follow the bulk formulas NZ (5.16)–(5.17)
with the drag velocity :math:`C_V = C_{DN}\,|\mathbf{V}_s|`:

.. math::

   E &= \rho_a c_p\, C_V \left[ q_{sat}(T_s)\tfrac{L}{c_p}
        - q_{ref,s} - b_{1s}\, q_1 \right], \\
   H &= \rho_a c_p\, C_V \left[ T_s - T_{ref,s} - a_{1s}\, T_1 \right],

with :math:`q_{sat}(T_s)` evaluated through the model's own saturation
table (see :doc:`convection`) and the ocean sensible flux floored at
:math:`-5` W m\ :sup:`-2`. Surface stress is
:math:`\boldsymbol\tau = \rho_a C_V \mathbf{v}_s` with :math:`C_V`
averaged to the staggered wind points. The neutral drag coefficient
comes from surface type at init (BATS/CCM2 form),

.. math::

   C_{DN} = \left[\ln(0.025\,h_{PBL}/Z_0)/\kappa + 8.4\right]^{-2},
   \qquad h_{PBL} = 2000\ \mathrm{m},

then doubled over land ("to compensate mountain drag") and pinned to
0.0011 over ocean — the Fortran's ABL first-call mutation, reproduced.

Surface winds (ABL scheme)
--------------------------

The surface wind solves the steady mixed-layer momentum balance at
every T point,

.. math::

   \frac{w_e}{z_i}(\mathbf{v}_b - \mathbf{v}_s)
   + f\,\hat{\mathbf{k}}\times\mathbf{v}_s
   - \nabla\phi_s
   - \frac{C_{DN}|\mathbf{V}_s|}{z_i}\,\mathbf{v}_s = 0,

where :math:`\mathbf{v}_b` is the wind at the ABL top from the mode-1
projection :math:`V_1(z_i)` (−0.228 at :math:`z_i = 500` m),
:math:`w_e = 0.01` m s\ :sup:`-1` the entrainment velocity,
:math:`\nabla\phi_s` the surface geopotential gradient from the
barotropic mode (see :doc:`dynamics`), and
:math:`|\mathbf{V}_s| = (V_{s,min}^2 + u_s^2 + v_s^2)^{1/2}` with the
gustiness floor :math:`V_{s,min} = 4.5` m s\ :sup:`-1`.

The 2×2 nonlinear system is solved pointwise by Newton iteration,
warm-started from the previous call's solution (this cross-step memory
is why the original model was not exactly restartable; the port carries
:math:`u_s, v_s` in :class:`~qtcm1.model.ModelState`). Acceptance
replicates the Fortran loop verbatim: a point must reach
:math:`|f|+|g| < 10^{-9}` within nine residual checks or it reverts to
its start-of-call winds. A ``NO_ABL`` variant (v2.2 surface winds,
fixed projection coefficients) is provided as ``sfcwind_noabl``.

**References.** NZ §5; ZNC §2e; Deardorff-type mixed-layer closure
discussed therein.
