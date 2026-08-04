Model physics
=============

QTCM1 is built on a Galerkin truncation of the primitive equations in
the vertical, keeping a single deep baroclinic structure plus a
barotropic component. Temperature and moisture are expanded about
reference profiles,

.. math::

   T(x,y,p,t) \approx T_{ref}(p) + a_1(p)\,T_1(x,y,t), \qquad
   q(x,y,p,t) \approx q_{ref}(p) + b_1(p)\,q_1(x,y,t),

and the velocity field as :math:`\mathbf{v} = \mathbf{v}_0(x,y,t) +
V_1(p)\,\mathbf{v}_1(x,y,t)`, where the basis functions
:math:`a_1, b_1, V_1` are chosen for consistency with deep
quasi-equilibrium convection: :math:`a_1(p)` is a moist-adiabatic
temperature structure, and :math:`V_1` follows from it hydrostatically.
The expansion is *analytic in the vertical*: all vertical integrals
(gross moist stability, flux projections, cloud-base winds) reduce to
precomputed inner products of the basis functions — the constants
tabulated in :mod:`qtcm1.constants`. The prognostic variables are the
mode amplitudes :math:`u_1, v_1, T_1, q_1`, the barotropic vorticity
and mean wind :math:`\zeta_0, \bar u_0`, and the surface state
:math:`T_s, W_D`. See Neelin & Zeng (2000; NZ) for the derivation and
Zeng, Neelin & Chou (2000; ZNC) for the physics closures; each page
below documents one component as implemented (module names in
parentheses), with the Fortran v2.3 heritage noted where behavior is
reproduced verbatim.

.. toctree::
   :maxdepth: 1

   convection
   cloudsrad
   surface
   land
   ocean
   dynamics

**References.**
Neelin, J. D. and N. Zeng (2000): A quasi-equilibrium tropical
circulation model — formulation. *J. Atmos. Sci.*, **57**, 1741–1766.
Zeng, N., J. D. Neelin and C. Chou (2000): A quasi-equilibrium tropical
circulation model — implementation and simulation. *J. Atmos. Sci.*,
**57**, 1767–1796.
