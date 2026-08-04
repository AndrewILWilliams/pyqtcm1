Convection (Betts–Miller, mode-projected)
=========================================

Module: :mod:`qtcm1.physics.convection` (Fortran ``mconvct`` +
``utilities.F90`` tables).

The scheme is a Betts–Miller (1986) relaxation projected onto the
single vertical mode: temperature and moisture are relaxed toward a
convective quasi-equilibrium reference profile on a fixed timescale
:math:`\tau_c`, and the column-integrated heating is

.. math::

   Q_c \;=\; \epsilon_c\, C_{pg}\, \langle \mathrm{CAPE}_1 \rangle_+ ,
   \qquad \epsilon_c = 1/\tau_c,\; \tau_c = 2\ \mathrm{h},

where :math:`C_{pg} = c_p\,\Delta p/g` converts the projected
temperature deficit to column energy and :math:`\langle\cdot\rangle_+`
denotes the positive part (no heating where the column is stable).
Because moisture is consumed one-for-one, :math:`Q_c` *is* the
precipitation in energy units (divide by :math:`L = 2.43\times10^6`
J kg\ :sup:`-1` for mm/day).

The projected convective available energy is

.. math::

   \mathrm{CAPE}_1 = \hat a_1\,(T_{1c} - T_1)
                   + (\widehat{T_{cref}} - \widehat{T_{ref}}),

with :math:`T_{1c}` the amplitude of the quasi-equilibrium profile the
column is relaxed toward. :math:`T_{1c}` is set by demanding that the
convective reference be in moisture balance with the actual column —
NZ eqs. (2.20)–(2.23) — which reduces to the scalar nonlinear equation

.. math::

   \hat a_1 T_{1c} + \widehat{q_{cp}}(T_{1c})
     \;=\; \widehat{T_{ref}} - \widehat{T_{cref}}
         + \hat a_1 T_1 + \widehat{q_{ref}} + \hat b_1 q_1 ,

where :math:`\widehat{q_{cp}}(T_{1c})` is the column projection of the
saturation humidity along the convective profile,

.. math::

   \widehat{q_{cp}}(T_{1c}) = \frac{1}{p_1 - p_N}\sum_k
      \tfrac{1}{2}\,(\phi_k + \phi_{k+1})\,(p_k - p_{k+1}),
   \quad
   \phi_k = \alpha_k\, q_{sat}\!\big(T_{cref,k} + a_{1,k} T_{1c}\big)
            \frac{L}{c_p}\frac{10^3}{p_k}.

Two lookup tables — deliberate parts of the numerics, reproduced
exactly — implement this: ``hsat``, the Tetens/Shuttleworth saturation
humidity tabulated every 0.1 K over 200–400 K and queried by linear
interpolation, and the ``t1ctable`` inversion of the equation above,
tabulated for :math:`T_{1c} \in [-300, 300]` K at 1-K spacing. A
``linear_closure`` flag reproduces the older v2.1 linearized closure
(compile option ``LINEAR_T1C``).

After the CAPE computation, :math:`Q_c` passes through the polar zonal
filter (see :doc:`dynamics`) and is clipped at zero. The heating enters
the :math:`T_1` equation as :math:`+Q_c/(C_{pg}\hat a_1)` and the
moisture equation as :math:`-Q_c/(C_{pg}\hat b_1)`.

**Parameters.** :math:`\tau_c` = ``eps_c`` :sup:`-1` (default the
package literal ``0.00013888889`` s\ :sup:`-1`); projection constants
:math:`\hat a_1, \hat b_1, \widehat{T_{ref}},\ldots` from
:mod:`qtcm1.constants` (``A1HAT``, ``B1HAT``, ...); the table pressure
levels/weights in ``T1C_TABLE``.

**References.** Betts, A. K. and M. J. Miller (1986), *Q. J. R.
Meteorol. Soc.*, **112**, 693–709; NZ §2; ZNC §2.
