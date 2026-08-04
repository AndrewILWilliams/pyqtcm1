"""Bit-exact restart round-trip (an improvement over the Fortran, whose
restart file omits the ABL warm start and is therefore inexact).

Run A: 6 days straight. Run B: 3 days, save restart, resume from file,
3 more days. Every ModelState array must match A's final state bitwise.
Also pins config round-trip through the restart header.
"""

import os

import numpy as np
import pytest

from qtcm1.config import PACKAGED_DATA, RunConfig
from qtcm1.driver import ControlRun

_DATA = os.path.expanduser(os.environ.get(
    'QTCM1_BNDDATA', PACKAGED_DATA))

pytestmark = pytest.mark.skipif(not os.path.isdir(_DATA),
                                reason='boundary registry not found')

_ARRAYS = ['u1', 'v1', 'T1', 'q1', 'u0', 'v0', 'vort0', 'Ts', 'WD',
           'us', 'vs', 'dphisdx', 'dphisdy']


def test_restart_roundtrip_bitwise(tmp_path):
    cfg = RunConfig(data_path=_DATA)

    a = ControlRun(config=cfg)
    for _ in range(6):
        a.advance_day()

    b = ControlRun(config=cfg)
    for _ in range(3):
        b.advance_day()
    fn = str(tmp_path / 'restart.npz')
    b.save_restart(fn)

    c = ControlRun.from_restart(fn)
    assert c.config.to_dict() == cfg.to_dict()
    assert c.dayofmodel == 3 and not c._getbnd_virgin
    for _ in range(3):
        c.advance_day()

    for k in _ARRAYS:
        np.testing.assert_array_equal(getattr(c.state, k),
                                      getattr(a.state, k), err_msg=k)
    for s in range(2):
        np.testing.assert_array_equal(c.state.rhs_hist[s],
                                      a.state.rhs_hist[s])
    assert c.state.u0bar == a.state.u0bar
    assert c.state.rhsbar_hist == a.state.rhsbar_hist
