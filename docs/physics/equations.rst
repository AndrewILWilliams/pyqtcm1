Governing equations
===================

This page sketches how the QTCM1 prognostic equations follow from a
separable (Galerkin) ansatz applied to the primitive equations in
pressure coordinates; NZ §§2–4 give the full derivation. The truncation
keeps a *single deep baroclinic mode* plus the barotropic component,
and — the essential QTCM idea — the vertical structure functions are
not arbitrary basis elements but the shapes selected by convective
quasi-equilibrium itself, so that in convecting regions the leading
term of the expansion is already close to the true solution.

The ansatz
----------

Temperature and moisture are expanded about reference profiles with a
single space–time amplitude each,

.. math::

   T(x,y,p,t) \;\approx\; T_{ref}(p) + a_1(p)\,T_1(x,y,t), \qquad
   q(x,y,p,t) \;\approx\; q_{ref}(p) + b_1(p)\,q_1(x,y,t),

i.e. the separable form :math:`T' = a_1(p)\,T_1(x,y,t)` for the
perturbation. Quasi-equilibrium fixes :math:`a_1`: deep convection
ties the free-tropospheric temperature to subcloud moist static
energy, so temperature perturbations follow perturbations of the moist
adiabat, :math:`a_1(p) = \partial T_{madiabat}(p; T_b)/\partial T_b`
(warming amplified aloft). :math:`b_1(p)` similarly concentrates
moisture variations in the lower troposphere.

Hydrostatic balance and the wind structure
------------------------------------------

Integrating hydrostatic balance :math:`\partial\phi/\partial p = -RT/p`
upward from the surface,

.. math::

   \phi = \phi_s + \phi_{ref}(p) + R\,a_1^+(p)\,T_1, \qquad
   a_1^+(p) = \int_p^{p_s} a_1(p')\, d\ln p' ,

so a single temperature structure implies a single geopotential
structure. Requiring the momentum equation to close within the
truncation, the wind must then be

.. math::

   \mathbf{v}(x,y,p,t) = \mathbf{v}_0(x,y,t) + V_1(p)\,\mathbf{v}_1(x,y,t),
   \qquad V_1 = a_1^+ - \langle a_1^+ \rangle ,

with :math:`\langle\cdot\rangle` the vertical (mass) average:
:math:`V_1` is the baroclinic wind structure driven by
:math:`\nabla T_1`, and :math:`\mathbf{v}_0` the barotropic flow driven
by :math:`\nabla\phi_s`. Continuity with rigid lids and
:math:`\langle V_1\rangle = 0` gives a nondivergent barotropic mode,
:math:`\nabla\!\cdot\!\mathbf{v}_0 = 0` (hence the streamfunction
formulation of :doc:`dynamics`), and the vertical velocity carried
entirely by mode 1,

.. math::

   \omega(p) = -\,\Omega_1(p)\, \nabla\!\cdot\!\mathbf{v}_1, \qquad
   \Omega_1(p) = \int_{p_T}^{p} V_1\, dp' .

Projected equations
-------------------

Substituting the ansatz and projecting — momentum onto :math:`V_1`,
thermodynamics onto :math:`a_1`, moisture onto :math:`b_1`, with
vertical averages :math:`\langle\cdot\rangle` — every vertical
integral becomes a constant inner product of the structure functions
(the hatted coefficients of :mod:`qtcm1.constants`: :math:`\hat a_1 =
\langle a_1 \rangle`, :math:`\langle V_1^2\rangle`, the advection
tensors :math:`V_{ijk} = \langle V_i V_j V_k\rangle`-type terms, etc.).
The result is the prognostic set the code integrates (NZ 5.1–5.4):

.. math::

   \partial_t \mathbf{v}_1 + \mathcal{D}_{v1}(\mathbf{v}_0,
       \mathbf{v}_1) + f\hat{\mathbf{k}}\times\mathbf{v}_1
     &= -\,R\,\kappa_1 \nabla T_1
        - \epsilon_1 \mathbf{v}_1
        - \epsilon_\tau \boldsymbol\tau_s + \mathbf{K}_v\!\cdot\!, \\
   \hat a_1\,\partial_t T_1 + \mathcal{D}_{T1}(\mathbf{v}, T_1)
     &= -\,M_s\, \nabla\!\cdot\!\mathbf{v}_1
        + \frac{Q_c + F^{SW} + F^{LW} + H}{C_{pg}}, \\
   \hat b_1\,\partial_t q_1 + \mathcal{D}_{q1}(\mathbf{v}, q_1)
     &= +\,M_q\, \nabla\!\cdot\!\mathbf{v}_1
        + \frac{E - Q_c}{C_{pg}},

together with the barotropic vorticity equation of :doc:`dynamics`.
Here :math:`\mathcal{D}` are the projected advection operators
(:mod:`qtcm1.dynamics.advection`), :math:`\epsilon_1` the internal
momentum damping, :math:`\epsilon_\tau` the projected surface-stress
coefficient, and the right-hand columns the physics closures of the
other pages.

The adiabatic terms deserve the emphasis: projecting
:math:`\omega\,\partial_p` of dry static energy :math:`s = c_p T +
\phi` and of moisture gives

.. math::

   M_s \equiv \big\langle \Omega_1\, \partial_p s \big\rangle
     = M_{sr} + M_{sp}\,T_1, \qquad
   M_q \equiv -\big\langle \Omega_1\, \partial_p q \big\rangle
     = M_{qr} + M_{qp}\,q_1 ,

linear in the amplitudes because :math:`s` and :math:`q` are — the
*gross dry stability* and *gross moisture stratification*. Their
difference :math:`M = M_s - M_q`, the **gross moist stability**,
controls the effective phase speed of convectively coupled motions and
the strength of the large-scale circulation response to heating; its
moisture dependence (moister columns → smaller :math:`M`) is the
central QTCM feedback. Implementation note: v2.3 evaluates :math:`M_s`
with the ZNC cloud-top correction :math:`M_s = M_{sr} +
M_{qp}\max(q_1, q_{1m})` rather than the :math:`T_1` form — reproduced
verbatim in :mod:`qtcm1.dynamics.baroclinic`.

What the truncation buys and costs
----------------------------------

Because :math:`a_1` is the QE structure, regions of active deep
convection are described by the leading term with the remainder
entering only through the closures; the model is effectively a
primitive-equation model *linearized about a moist-convecting
atmosphere* while retaining full nonlinearity in advection and the
closures. The cost is that motions far from QE — shallow circulations,
extratropical baroclinic eddies, boundary-layer decoupling — are
represented crudely or not at all, which bounds the model's domain of
validity to the tropics (the walls at 78.75° and the sponge-free v2.3
configuration reflect this).

**References.** NZ §§2–4 (derivation, structure functions, inner
products); ZNC §2 (closure evaluations); Emanuel, Neelin & Bretherton
(1994) for the quasi-equilibrium framework.
