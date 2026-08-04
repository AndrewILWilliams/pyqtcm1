Quickstart
==========

Install
-------

.. code-block:: bash

   git clone https://github.com/AndrewILWilliams/pyqtcm1
   cd pyqtcm1
   pip install -e .[numba]        # numba optional but ~1.7x faster

Numba is optional: without it every routine falls back to a pure-NumPy
path that produces **bit-identical** results.

Boundary data
-------------

The netCDF boundary registry for the standard r64x42 grid — SST
(climatological, dated 1949-2001, perpetual), Darnell albedo, surface
types, cloud climatologies, masks and the diagnosed slab-ocean Q-flux —
**ships with the repository** in ``data/r64x42`` (~17 MB, full float64,
with a sha256 manifest of the original ASCII sources). A checkout is
self-contained: ``RunConfig()`` finds it automatically. To regenerate
it, or to build a registry for another grid:

.. code-block:: bash

   python tools/convert_bnddata.py --bnddir <qtcm1>/bnddir/r64x42 \
                                   --out data/r64x42

A first run
-----------

A cold-started control run with climatological seasonal SST, in a dozen
lines:

.. code-block:: python

   from qtcm1.config import RunConfig
   from qtcm1.driver import ControlRun

   cfg = RunConfig()          # packaged data, build='f64' default
   run = ControlRun(config=cfg)

   run.run_years(2)                       # ~1 minute per simulated year
   run.save_monthly('control_monthly.npz')

   # instantaneous state, if you want it directly:
   Ts = run.state.Ts                      # (ny, nx) float64
   T1 = run.state.T1

``run_years`` integrates whole 365-day years; ``advance_day()`` steps a
single coupling day and returns the model date. Monthly means are
accumulated every time step (the Fortran's ``varmean`` semantics) for
eight prognostics and seven flux/diagnostic fields, and the output file
carries a provenance header (git hash, full configuration, input-file
hashes).

The two builds
--------------

``RunConfig(build=...)`` names the scientific configuration explicitly:

* ``'f64'`` (default, recommended) — float64 init-time constants: the
  equation set as written, with the polar filter acting on 5 rows per
  pole.
* ``'f32'`` (heritage) — mirrors the init-time constants of the original
  single-precision Fortran build (4 filtered rows, float32 lookup
  tables). Use this to reproduce the historical v2.3 climate; the two
  builds differ measurably (~0.1% in the global hydrological cycle; see
  :doc:`validation`).

Restarts
--------

Restarts are **bit-exact** — an improvement over the Fortran, whose
restart file omits the ABL warm-start winds:

.. code-block:: python

   run.save_restart('day730.restart.npz')
   ...
   run2 = ControlRun.from_restart('day730.restart.npz')
   run2.run_years(8)      # continues the trajectory to the last bit

Working at a lower level
------------------------

:class:`qtcm1.model.Model` is the stepping engine and
:class:`qtcm1.model.ModelState` the complete state — a plain dataclass
of arrays holding *everything* the model carries between steps
(prognostics, ABL warm start, AB3 histories, surface-geopotential
gradients). ``Model.step`` is a pure function of ``(state, forcing)``:

.. code-block:: python

   import numpy as np
   from qtcm1.driver import ControlRun

   run = ControlRun(config=RunConfig())
   model, state = run.model, run.state

   sst = run.bd.sst(year=1, dayofyear=100)     # boundary fields, any day
   alb = run.bd.albedo(100)
   state, alb64 = model.apply_boundary(state, sst, alb)
   for it in range(1, 73):                     # one coupling day = 72 steps
       state, diags = model.step(state, alb64, dayofyear=100, it=it)

   print(float(diags['Qc'].mean()) / 28.125)   # precipitation, mm/day

Every physics and dynamics routine (``mconvct``, ``radsw``, ``barcl``,
...) is itself an importable pure function returning its outputs in a
dict — see :doc:`api` — which makes single-process experimentation and
mechanism surgery straightforward (see the :doc:`examples`).
