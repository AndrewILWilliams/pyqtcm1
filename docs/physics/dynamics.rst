Dynamics
========

Modules: :mod:`qtcm1.dynamics` (Fortran ``qtcm.F90``: ``barcl``,
``bartr``, ``advctuv``, ``advctTq``, ``dffus``, ``xfilter``,
``fatdpkg``).

Baroclinic mode
---------------

The mode-1 momentum, temperature and moisture equations (NZ 5.1,
5.3–5.4) are advanced with the Fortran's exact sequential structure:
:math:`u_1` first (surface-stress damping
:math:`-\epsilon_\tau \tau_x`, internal damping
:math:`\epsilon_{1}`, Coriolis from the *old* :math:`v_1`, the mode
pressure gradient :math:`-R\,\partial_x T_1`, advection and
diffusion), polar-filtered; then :math:`v_1` using the **new, filtered**
:math:`u_1` in its Coriolis term; then the mode-1 divergence
:math:`\nabla\!\cdot\!\mathbf{v}_1` from the updated winds drives

.. math::

   \hat a_1 \partial_t T_1 &= -M_s\,\nabla\!\cdot\!\mathbf{v}_1
      + \frac{Q_c + F^{SW} + F^{LW} + H}{C_{pg}} + \ldots \\
   \hat b_1 \partial_t q_1 &= +M_q\,\nabla\!\cdot\!\mathbf{v}_1
      + \frac{E - Q_c}{C_{pg}} + \ldots

with the *gross moist stability* closure central to QTCM dynamics:

.. math::

   M_s = M_{sr} + M_{qp}\,\max(q_1, q_{1m}), \qquad
   M_q = M_{qr} + M_{qp}\, q_1 ,

i.e. a dry stability corrected for cloud-top/moisture dependence and a
moisture stratification linear in :math:`q_1` (NZ §3). Advection uses
the precomputed mode inner products (:math:`V_{ijk}` etc., NZ 4.7–4.9)
including the vertical ("ω") transport terms.

Barotropic mode
---------------

The vertically averaged flow is nondivergent; its vorticity

.. math::

   \partial_t \zeta_0 = \hat{\mathbf{k}}\cdot\nabla\times\left(
      -g p_T^{-1}\boldsymbol\tau + \mathbf{A}_0 + \mathbf{D}_0\right)
      - \beta\, v_0

is advanced with third-order Adams–Bashforth, polar-filtered, and
inverted for the streamfunction, :math:`\nabla^2\psi_0 = \zeta_0`, by
the FATD direct solver (real FFT in longitude + prefactorized
tridiagonal solves in latitude) with Dirichlet walls; the north-wall
value carries the AB3-advanced domain-mean zonal wind :math:`\bar u_0`.
The surface geopotential gradient :math:`\nabla\phi_s` — needed by the
ABL wind scheme — is then recovered from the mode-0 momentum balance
(NZ appendix A), and a line integration provides the diagnostic surface
pressure.

Grid, filter, diffusion
-----------------------

Arakawa C-grid, 5.625° × 3.75°, walls at 78.75°S/N, :math:`\Delta t =
1200` s. Poleward of 60° an Arakawa–Lamb zonal filter damps wavenumber
:math:`m` by :math:`\min\!\big(0.9\sqrt{\cos\varphi}/\sin(m\pi/n_x),
1\big)` to relax the zonal CFL limit; the filtered-row count is an
init-precision knife edge documented in :doc:`../validation`.
Momentum uses fourth-order (∇⁴, MM5-style) horizontal diffusion with
one-sided Laplacian forms at the walls; :math:`T_1` and :math:`q_1` use
∇² diffusion; all with spherical-metric weight tables. Viscosities
(defaults :math:`7\times10^5` m² s\ :sup:`-1` for winds,
:math:`1.2\times10^6` for :math:`T_1, q_1`) are resolution-tuned — see
the 1° discussion in the project notes before changing resolution.

**References.** NZ §§3–5 and appendix A; Adcroft's FATD solver notes
(distributed with QTCM1); Arakawa & Lamb (1977) for the filter.
