Land surface (Simple-LAND / SLAND1)
===================================

Module: :mod:`qtcm1.physics.land` (Fortran ``sland1``; Zeng, Neelin &
Chou land scheme, "version 1B").

One soil layer per land point, with surface types forest (1), grassland
(2) and desert/ice (3) setting the field capacity :math:`W_{D0}` = 500,
400, 300 kg m\ :sup:`-2`, roughness length, leaf-area index and minimum
stomatal resistance. Prognostics are the soil water :math:`W_D` and
ground temperature :math:`T_s`:

.. math::

   \frac{dW_D}{dt} = \frac{P - E_{land} - R}{L}, \qquad
   c_{soil}\frac{dT_s}{dt} = F^{SW}_{net} + F^{LW}_{net} - E_{land} - H,

with :math:`W_D` floored at zero and
:math:`c_{soil} = 4.18\times10^5` J K\ :sup:`-1` m\ :sup:`-2`
(a 0.1-m water-equivalent layer; the fast ~hours ground response).
All water/energy terms are in W m\ :sup:`-2` (:math:`P = Q_c`).

Evapotranspiration scales the potential ("swamp") evaporation from the
bulk formula by a canopy efficiency built from the aerodynamic and
stomatal resistances, with :math:`w = W_D/W_{D0}` the relative wetness:

.. math::

   E_T = \frac{r_a^{-1}}{r_s + r_a^{-1}}\,E_{pot}
   \quad\text{with}\quad
   r_a^{-1} = \frac{w^{1/4}}{C_V},\qquad r_s = r_{s,min},

plus an interception loss :math:`E_i`: the available energy times a
stochastic-rainfall interception function (storm intensity
:math:`1.06\times10^{-3}` mm s\ :sup:`-1`, duration 72 min, ARME
calibration), capped at half the precipitation.

Runoff has a BATS-style surface component and a steep subsurface
drainage,

.. math::

   R_s = (P - E_i)\, w^4, \qquad
   R_g = L\,R_{g0}\, w^{11},\qquad R_{g0} = 4\times10^{-4}\
   \mathrm{mm\,s^{-1}},

so the bucket sheds water rapidly as it approaches saturation. Ocean
points pass through untouched; land diagnostics (``Evapi``, ``wet``,
``Runs``, ``Runf``) are archived on request. The alternative
Manabe-style ``bucket`` scheme is also ported.

The soil moisture is the slowest state in the model (~1–2 year
cold-start spin-up; see :doc:`../output`), and interactive
:math:`W_D`–precipitation feedback is a large part of QTCM1's land
climate variability.

**References.** ZNC §2f and Zeng et al. land-scheme papers referenced
therein; Dickinson et al. (1986) BATS for the runoff form.
