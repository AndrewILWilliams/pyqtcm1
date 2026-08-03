# Instrumented oracle build

Golden fixtures require an *extended* build of the f2py `parts` extension in
which `setbypy.F90` exposes the tendency/diagnostic arrays (adv*, dfs*,
div*, chi*, GM*, surface diagnostics). Apply `setbypy_extended.patch` to the
qtcm 0.1.2 `src/setbypy.F90`, then rebuild:

    patch src/setbypy.F90 < setbypy_extended.patch
    f2py -c -m _qtcm_parts_365 varptrinit.F90 wrapcall.F90 setbypy.F90 \
         -L$PWD -lqtcm -lnetcdff -lnetcdf \
         --f90flags="-fPIC -I$PWD -fallow-argument-mismatch -std=legacy" \
         --backend meson

and drop the renamed `.so` into a copy of the (py3-ported) qtcm package.
Then run `tools/gen_golden.py --oracle <that package dir> ...`.
