# pyqtcm1

Pure-Python reimplementation of the Neelin–Zeng Quasi-Equilibrium Tropical
Circulation Model, version 1 (QTCM1 v2.3), migrated from the Fortran core of
J. W.-B. Lin's `qtcm` 0.1.2 package.

Status: **Phase 1 (foundations)** — constants (+ Tier-0 checker against the
Fortran source), model calendar, C-grid geometry, the FATD elliptic solver,
and the netCDF boundary-data reader. The physics/dynamics chain lands in
Phase 2. See `claude/pyqtcm1-rewrite-scope.md` in the project for the full
migration plan, and the control-baseline doc for the 40-year validation
target the finished model must reproduce.

Conventions: arrays are C-ordered `(lat, lon)` = Fortran `(ny, nx)`
transposed; scientific field names keep the paper notation (`u1, T1, q1`, …);
every ported function's docstring names its Fortran origin and the Neelin &
Zeng (2000) equations it implements. Boundary inputs are netCDF only,
produced by `tools/convert_bnddata.py` from the original ASCII `bnddir`.

Test tiers (scope §6): Tier 0 = constants vs Fortran source (runs here);
Tier 1/2 = golden per-routine + trajectory vs the compiled f2py oracle
(oracle-marked tests, skipped when the extension is absent); Tier 3 =
40-year control statistics.
