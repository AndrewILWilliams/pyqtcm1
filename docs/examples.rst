Example gallery
===============

Runnable scripts in ``examples/`` (each needs the converted boundary
registry; point ``QTCM1_BNDDATA`` at it). They are deliberately short —
the API is small, and every experiment below is a plain Python script,
not a configuration dialect.

Control run (fixed climatological SST)
--------------------------------------

The standard experiment: cold start, seasonal Reynolds SST, monthly
means with provenance, a restart file for later branching.

.. literalinclude:: ../examples/01_control_run.py
   :language: python

SST-anomaly experiment (El Niño-like patch)
-------------------------------------------

SST is a boundary condition, so anomaly runs need no model changes:
write the day loop yourself and perturb the SST before it is applied.
This is the pattern for pacemaker runs, warming patches, uniform
+2 K experiments, and observed-SST (``sst_mode='real_time'``) cases.

.. literalinclude:: ../examples/02_sst_anomaly.py
   :language: python

Idealized greenhouse forcing (CO2-like), fixed SST
--------------------------------------------------

QTCM1 v2.3 has no explicit CO2 parameter; greenhouse experiments
perturb the longwave budget. Wrapping ``radlw`` at the model's import
site gives a clean +F W/m² forcing run. Note the caveat in the script:
with prescribed SST this is the *fast* response only — for the full
response with an interactive ocean, see the slab example below.

.. literalinclude:: ../examples/03_radiative_forcing.py
   :language: python

Restarts and last-bit twins
---------------------------

Bit-exact restart round-trips, and the model's own error-growth
behavior measured with a one-ulp initial perturbation.

.. literalinclude:: ../examples/04_restart_twin.py
   :language: python

Heritage vs recommended build
-----------------------------

Not a separate script — one line. ``build='f32'`` reproduces the
historical single-precision Fortran climate; ``build='f64'`` is the
equation set as written. Comparing the two quantifies the original
model's single-precision artifact (see :doc:`validation`):

.. code-block:: python

   ControlRun(config=RunConfig(data_path=DATA, build='f32'))

Slab ocean with greenhouse forcing
----------------------------------

The full version of the experiment above: spin up with fixed SST,
diagnose the Q-flux from a control (``tools/make_qflux.py``), branch a
slab-ocean control and a slab run with +4 W/m² — now the SST responds.
The slab holds the control climatology by construction (ocean-mean
drift ~0.1 K over 90 days when branched from a spun-up state).

.. literalinclude:: ../examples/05_slab_co2.py
   :language: python

Roadmap
-------

Not yet ported from the original option set: topography (``TOPO``) and
the ISCCP cloud climatology option. The slab ocean is formula-validated
(unit tests + closure against the control climatology); bit-level golden
validation against a ``-DMXL_OCEAN`` Fortran build is planned. Planned additions beyond option parity:
first-class intervention hooks (replacing the wrap-the-routine pattern
above), a batched ensemble dimension, and budget-closing xarray
diagnostics.
