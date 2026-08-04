"""Bit-exact restart files.

The Fortran QTCM1 is famously not exactly restartable (its restart file
omits the ABL warm-start winds, the surface-geopotential gradients and
part of the AB3 bookkeeping; the source itself notes the ABL warm start
"makes QTCM not strictly restartable"). The port's :class:`ModelState`
carries every array the model propagates, so a restart here round-trips
bit-identically: save -> load -> continue produces the same trajectory,
to the last bit, as an uninterrupted run (pinned by tests/test_restart.py).

Format: a single compressed npz holding every ModelState field plus the
driver position (dayofmodel, boundary first-call flag) and a provenance
header (see :mod:`qtcm1.config`).
"""

from __future__ import annotations

import json

import numpy as np

from ..model import ModelState

_ARRAYS = ['u1', 'v1', 'T1', 'q1', 'u0', 'v0', 'vort0', 'Ts', 'WD',
           'us', 'vs', 'dphisdx', 'dphisdy']


def save_restart(path: str, state: ModelState, *, dayofmodel: int = 0,
                 getbnd_virgin: bool = False, header: dict | None = None,
                 extra: dict | None = None):
    """Write a complete, bit-exact restart file.

    ``extra`` holds additional component state (e.g. the slab ocean's
    Tnow and flux accumulators), stored under ``extra/<key>``.
    """
    payload = {k: getattr(state, k) for k in _ARRAYS}
    for k, v in (extra or {}).items():
        payload[f'extra/{k}'] = np.asarray(v)
    payload['rhs_hist0'] = state.rhs_hist[0]
    payload['rhs_hist1'] = state.rhs_hist[1]
    payload['rhsbar_hist'] = np.array(state.rhsbar_hist, dtype=np.float64)
    payload['u0bar'] = np.float64(state.u0bar)
    payload['psi0'] = (np.zeros(0) if state.psi0 is None else state.psi0)
    payload['div0'] = (np.zeros(0) if state.div0 is None else state.div0)
    payload['flags'] = np.array([int(state.gradphis_virgin),
                                 int(getbnd_virgin), int(dayofmodel)],
                                dtype=np.int64)
    payload['header'] = np.frombuffer(
        json.dumps(header or {}).encode(), dtype=np.uint8)
    np.savez_compressed(path, **payload)


def load_restart(path: str):
    """Read a restart file.

    Returns ``(state, dayofmodel, getbnd_virgin, header, extra)``.
    """
    z = np.load(path)
    extra = {k[len('extra/'):]: z[k] for k in z.files
             if k.startswith('extra/')}
    psi0 = z['psi0']
    div0 = z['div0'] if 'div0' in z.files else np.zeros(0)  # pre-TOPO files
    state = ModelState(
        u1=z['u1'], v1=z['v1'], T1=z['T1'], q1=z['q1'],
        u0=z['u0'], v0=z['v0'], vort0=z['vort0'],
        u0bar=float(z['u0bar']),
        rhs_hist=[z['rhs_hist0'], z['rhs_hist1']],
        rhsbar_hist=[float(x) for x in z['rhsbar_hist']],
        Ts=z['Ts'], WD=z['WD'], us=z['us'], vs=z['vs'],
        dphisdx=z['dphisdx'], dphisdy=z['dphisdy'],
        psi0=None if psi0.size == 0 else psi0,
        gradphis_virgin=bool(z['flags'][0]),
        div0=None if div0.size == 0 else div0)
    header = json.loads(bytes(z['header']).decode() or '{}')
    return state, int(z['flags'][2]), bool(z['flags'][1]), header, extra
