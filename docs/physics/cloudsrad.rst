Clouds and radiation
====================

Modules: :mod:`qtcm1.physics.clouds`, :mod:`qtcm1.physics.radiation`
(Fortran ``clrad.F90``: ``cloud``, ``radsw``, ``radlw``).

Cloud fractions
---------------

Four radiatively active cloud types over clear sky, ZNC §2d / Chou
(1997). Deep convective cloud with anvil (type 1) is tied linearly to
precipitation and capped at total cover,

.. math::

   \mathrm{cl}_1 = \min(c_1\, Q_c,\; 1), \qquad
   c_1 = 7.763\times10^{-4}\ \mathrm{m^2\,W^{-1}},

cirrus/cirrostratus (type 2) is 1.5× the deep cover before overlap,
and stratus (3) and AsAc+CuSc (4) carry fixed reference covers.
Random-overlap bookkeeping converts these to non-overlapped fractions
summing to at most one, with clear sky the residual:

.. math::

   \mathrm{cld}_2 &= 1.5\,\mathrm{cl}_1 (1 - \mathrm{cl}_1), \\
   \mathrm{cld}_{3,4} &= \mathrm{cld}^{tot}_{3,4}
       (1 - \mathrm{cl}_1 - \mathrm{cld}_2), \qquad
   \mathrm{cld}_0 = 1 - \textstyle\sum_{i=1}^4 \mathrm{cld}_i .

An ``OBSCLD``-style override (prescribing the type-3 climatology) is
accepted via the ``cldtot3`` argument.

Shortwave
---------

Incoming solar :math:`S_0(\varphi, d)` follows the daily-mean
astronomical formula (no diurnal cycle; 365-day calendar). Column
absorption and the surface/top fluxes use the weakly nonlinear scheme
of ZNC (their §2c): per cloud type, transmissions and absorptions
fitted offline to detailed calculations, combined with the surface
albedo from the boundary data, and weighted by the overlap fractions
above. Outputs: :math:`S_0`, surface down/up (``FSWds``, ``FSWus``),
top-of-atmosphere up (``FSWut``), and the column absorption ``FSW``
that heats the :math:`T_1` equation.

Longwave
--------

The longwave scheme is the ZNC linearization about the reference
profiles: each flux (surface up/down, top) is expanded to first order
in the mode amplitudes and the surface state,

.. math::

   F^{LW} \approx F^{LW}_{ref}(\mathrm{cld})
      + \frac{\partial F}{\partial T_1} T_1
      + \frac{\partial F}{\partial q_1} q_1
      + \frac{\partial F}{\partial T_s} (T_s - T_{ref,s}),

with coefficients per cloud type from offline detailed-model fits
(Chou & Neelin 1996), combined by the overlap fractions. Outputs:
``FLWds``, ``FLWus``, ``FLWut`` (= OLR) and the column longwave
cooling ``FLW``. Greenhouse-gas perturbation experiments modify this
budget directly (see the examples gallery), since v2.3 carries no
explicit CO\ :sub:`2` parameter.

**References.** ZNC §2c–d; Chou, C. and J. D. Neelin (1996), *J.
Geophys. Res.*, **101**, 15271–15289; Chou (1997, UCLA thesis).
