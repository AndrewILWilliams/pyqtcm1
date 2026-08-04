Ocean: prescribed SST and the mixed layer
=========================================

Modules: :mod:`qtcm1.io.bnddata` (prescribed), :mod:`qtcm1.physics.ocean`
(slab; Fortran ``ocean.F90``/``cplmean.F90``, options
``MXL_OCEAN``/``BLEND_SST``).

Prescribed SST
--------------

The standard configuration prescribes SST over ocean points from
monthly fields — climatological (``seasonal``), observed-by-date
(``real_time``) or fixed (``perpetual``) — interpolated linearly
between mid-month anchors and applied once per coupling day at the
interval midpoint :math:`t = d + \tfrac{1}{2}`. The port reproduces
the Fortran's daily fields bitwise (see :doc:`../validation`,
including the calendar quirks pinned there).

Mixed-layer ("slab") ocean
--------------------------

With ``sst_mode='mixed_layer'`` the ocean temperature becomes
prognostic:

.. math::

   C_{mx}\,\frac{dT}{dt} \;=\; \overline{F}_{s,net} \;-\; Q_{flux},
   \qquad C_{mx} = 4.18\times10^{6}\, D_{mx},\; D_{mx} = 50\ \mathrm{m},

integrated once per coupling day over ocean points, where
:math:`\overline{F}_{s,net} = \overline{F^{SW}_{net} + F^{LW}_{net} -
E - H}` is averaged over the *previous* day's atmospheric steps (the
``cplmean`` accumulation; the first day of a branch uses zero fluxes,
as in the Fortran).

The Q-flux is the implied ocean heat transport that keeps the slab on a
target climatology. It is diagnosed from a fixed-SST control run
(``tools/make_qflux.py``, the ``aveflux`` port):

.. math::

   Q_{flux}(d) = f_{sn}(d) - d_{ts}(d),

with :math:`f_{sn}` the control's monthly-climatological net surface
flux (mid-month interpolated) and :math:`d_{ts} = C_{mx}\,
\partial T_s/\partial t` the control's SST tendency (piecewise-constant
per month, switching mid-month — the Fortran's two distinct file
readers, both reproduced). By construction the unperturbed slab then
reproduces the control's seasonal SST: branched from a spun-up state it
holds it to ~0.1 K mean / 0.5 K RMS. Perturbation experiments (e.g. a
longwave forcing) let the SST respond with the slab's ~1–2 yr
adjustment time.

``sst_mode='blend'`` keeps prescribed SST inside the ``ensopac`` mask
region (e.g. for pacemaker experiments) and slab SST elsewhere.

Validation status: formula-level and closure-tested; bit-level golden
validation against a ``-DMXL_OCEAN -DCPLMEAN`` Fortran build is on the
roadmap.

**References.** ZNC §2g; the Q-flux methodology follows the standard
slab-ocean practice (e.g. Hansen et al. 1984) as implemented in the
QTCM1 distribution's ``aveflux`` utility.
