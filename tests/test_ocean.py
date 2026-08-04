"""Mixed-layer (slab) ocean: formula-level unit tests + a branched smoke run.

Bit-level golden validation against a -DMXL_OCEAN -DCPLMEAN Fortran build
is on the roadmap; these tests pin the ported formulas (mxstep energy
budget, Q-flux closure, the bndry1/bndry2 reading conventions) and the
driver integration.
"""

import os

import numpy as np
import pytest

from qtcm1.physics.ocean import CMX, MixedLayerOcean, QFlux

ANCHORS = np.array([-16, 15, 46, 74, 105, 135, 166, 196, 227, 258, 288,
                    319, 349, 380])


def _qflux(fsn_const=0.0, dts_const=0.0, shape=(3, 4)):
    fsn = np.full((12, *shape), fsn_const)
    dts = np.full((12, *shape), dts_const)
    return QFlux(fsn, dts, ANCHORS)


def test_qflux_bndry2_month_switch():
    """dts is piecewise-constant with the switch at day-of-month 15."""
    dts = np.zeros((12, 1, 1))
    dts[:, 0, 0] = np.arange(1, 13)
    qf = QFlux(np.zeros((12, 1, 1)), dts, ANCHORS)
    # Feb 14 (doy 45) reads dts for February; Feb 15 (doy 46) reads March
    assert qf(45, 2, 14)[0, 0] == -2.0
    assert qf(46, 2, 15)[0, 0] == -3.0
    # Dec 15 wraps to January
    assert qf(349, 12, 15)[0, 0] == -1.0


def test_qflux_fsn_midmonth_interpolation():
    fsn = np.zeros((12, 1, 1))
    fsn[0], fsn[1] = 10.0, 20.0                  # Jan, Feb
    qf = QFlux(fsn, np.zeros((12, 1, 1)), ANCHORS)
    # day 30: bracket (15, 46), instant 30.5 -> exactly midway
    assert qf(30, 1, 30)[0, 0] == pytest.approx(15.0)


def test_mxstep_energy_budget_exact():
    """Uniform net flux Q for one day warms the slab by Q*86400/Cmx."""
    stype = np.zeros((3, 4))
    ml = MixedLayerOcean(_qflux(), stype)
    ml.Tnow = np.full((3, 4), 300.0)
    Q = 10.0
    diags = dict(FSWds=np.full((3, 4), Q), FSWus=np.zeros((3, 4)),
                 FLWds=np.zeros((3, 4)), FLWus=np.zeros((3, 4)),
                 Evap=np.zeros((3, 4)), FTs=np.zeros((3, 4)))
    for _ in range(72):
        ml.accumulate(diags)
    T = ml.step_day(100, 4, 10)
    np.testing.assert_allclose(T, 300.0 + Q * 86400.0 / CMX, rtol=0)


def test_mxstep_qflux_closure():
    """Fluxes exactly equal to the Q-flux leave the slab unchanged."""
    stype = np.zeros((3, 4))
    ml = MixedLayerOcean(_qflux(fsn_const=25.0), stype)
    ml.Tnow = np.full((3, 4), 299.0)
    diags = dict(FSWds=np.full((3, 4), 25.0), FSWus=np.zeros((3, 4)),
                 FLWds=np.zeros((3, 4)), FLWus=np.zeros((3, 4)),
                 Evap=np.zeros((3, 4)), FTs=np.zeros((3, 4)))
    for _ in range(72):
        ml.accumulate(diags)
    T = ml.step_day(100, 4, 10)
    np.testing.assert_array_equal(T, 299.0)


def test_mxstep_land_untouched_and_first_day_zero_fluxes():
    stype = np.array([[0.0, 1.0]])
    ml = MixedLayerOcean(_qflux(fsn_const=50.0, shape=(1, 2)), stype)
    ml.Tnow = np.array([[300.0, 285.0]])
    T = ml.step_day(1, 1, 1)                     # no accumulation yet
    assert T[0, 1] == 285.0                      # land untouched
    # ocean gets the Fortran first-day kick: -(qfx)*86400/Cmx
    assert T[0, 0] == pytest.approx(300.0 - 50.0 * 86400.0 / CMX)


_DATA = os.path.expanduser(os.environ.get(
    'QTCM1_BNDDATA', '~/work/data/qtcm1_bnd_r64x42'))


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_DATA, 'qflux.nc')),
    reason='boundary registry with qflux.nc not found')
def test_slab_branched_from_spinup_stays_near_control(tmp_path):
    from qtcm1.config import RunConfig
    from qtcm1.driver import ControlRun

    a = ControlRun(config=RunConfig(data_path=_DATA))
    for _ in range(30):
        a.advance_day()
    fn = str(tmp_path / 'spin.npz')
    a.save_restart(fn)

    b = ControlRun.from_restart(fn, config=RunConfig(
        data_path=_DATA, sst_mode='mixed_layer'))
    ocean = b.model.stype == 0
    for _ in range(15):
        b.advance_day()
    doy = b.calendar.timemanager(b.dayofmodel).dayofyear
    d = (b.state.Ts - b.bd.sst(1, doy))[ocean]
    assert np.isfinite(b.state.Ts).all()
    assert abs(d.mean()) < 1.0 and np.sqrt((d ** 2).mean()) < 2.0
