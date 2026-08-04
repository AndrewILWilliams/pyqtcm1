"""Run configuration and provenance.

A run is described by one declarative :class:`RunConfig`; the same dict
round-trips through JSON, is stamped into every output and restart file,
and names the *scientific configuration* explicitly:

* ``build='f64'`` (default, recommended): float64 init constants -- the
  equation set as written, polar filter on 5 rows per pole (js=5).
* ``build='f32'`` (heritage): mirrors the single-precision Fortran build's
  init constants (js=4, f32 lookup tables); use to reproduce the
  historical v2.3 climate.

:func:`provenance` collects the code version (git hash if available),
the config, and the boundary-data manifest hashes, so any output can be
traced to exact code + configuration + inputs.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess

import numpy as np

BUILDS = {'f64': np.float64, 'f32': np.float32}


#: the boundary-data registry shipped with the repository (data/r64x42)
PACKAGED_DATA = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'r64x42'))


@dataclasses.dataclass
class RunConfig:
    """Declarative description of a QTCM1 run.

    ``data_path`` defaults to the registry shipped in the repository
    (``data/r64x42``), so a checkout is fully self-contained; pass a
    path only for other grids or regenerated data.
    """

    data_path: str | None = None          #: netCDF boundary registry
    build: str = 'f64'                    #: 'f64' (recommended) | 'f32'
    year0: int = 1
    month0: int = 1
    day0: int = 1
    sst_mode: str = 'seasonal'
    params: dict = dataclasses.field(default_factory=dict)  #: DEFAULT_PARAMS overrides
    #: output request: {var: {'freq': 'step'|'<n>h'|'daily'|'monthly',
    #: 'kind': 'mean'|'inst'}}; None = the standard monthly-mean archive
    #: (qtcm1.io.output.DEFAULT_OUTPUT). See the docs "Model output" page.
    output: dict | None = None
    #: path to a custom surface netCDF (stype/top; see qtcm1.surface).
    #: None = the registry's surface.nc. In-memory surfaces go through
    #: ControlRun(surface=...) instead (not serializable here).
    surface: str | None = None
    #: albedo handling with a custom surface: 'auto' (Darnell climatology
    #: where stype is unchanged, static per-type values where it changed),
    #: 'by_stype' (static everywhere), 'darnell' (climatology everywhere).
    albedo_mode: str = 'auto'
    #: topographic lifting (the Fortran ``TOPO`` compile option): div0
    #: from terrain-following winds and -f*div0 vortex stretching in the
    #: barotropic vorticity equation.
    topo: bool = False

    def __post_init__(self):
        if self.build not in BUILDS:
            raise ValueError(f'build must be one of {sorted(BUILDS)}')
        if self.albedo_mode not in ('auto', 'by_stype', 'darnell'):
            raise ValueError("albedo_mode must be 'auto', 'by_stype' or "
                             "'darnell'")
        if self.data_path is None:
            self.data_path = PACKAGED_DATA
        self.data_path = os.path.expanduser(self.data_path)
        if self.surface is not None:
            self.surface = os.path.expanduser(self.surface)

    @property
    def init_dtype(self):
        return BUILDS[self.build]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'RunConfig':
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def _git_hash() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=root, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or 'unknown'
    except Exception:                      # pragma: no cover
        return 'unknown'


def provenance(config: RunConfig) -> dict:
    """Code + config + input identity for stamping into outputs."""
    prov = dict(code_git=_git_hash(), config=config.to_dict())
    manifest = os.path.join(config.data_path, 'manifest.json')
    if os.path.exists(manifest):
        with open(manifest) as f:
            prov['input_manifest'] = json.load(f)
    return prov
