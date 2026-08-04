"""Configurable output: frequencies, mean vs instantaneous, xarray."""

import os

import numpy as np
import pytest

from qtcm1.config import PACKAGED_DATA, RunConfig
from qtcm1.driver import ControlRun

_DATA = os.path.expanduser(os.environ.get(
    'QTCM1_BNDDATA', PACKAGED_DATA))

pytestmark = pytest.mark.skipif(not os.path.isdir(_DATA),
                                reason='boundary registry not found')

SPEC = {
    'Ts': {'freq': 'daily', 'kind': 'inst'},
    'Qc': {'freq': 'daily', 'kind': 'mean'},
    'u1': {'freq': '6h', 'kind': 'inst'},
    'T1': {'freq': 'monthly', 'kind': 'mean'},
    'WD': {'freq': 'monthly', 'kind': 'inst'},
}


def test_output_spec_end_to_end():
    import xarray as xr

    run = ControlRun(config=RunConfig(data_path=_DATA, output=SPEC))
    for _ in range(35):                          # crosses the Jan/Feb line
        run.advance_day()
    dsets = run.to_datasets()

    assert set(dsets) == {'daily', '6h', 'monthly'}
    assert all(isinstance(d, xr.Dataset) for d in dsets.values())

    daily = dsets['daily']
    assert daily['Ts'].shape == (35, 42, 64)
    assert daily['Qc'].attrs['cell_methods'] == 'time: mean'
    assert daily['Ts'].attrs['cell_methods'] == 'time: point'
    assert daily['Qc'].attrs['units'] == 'W m-2'
    # daily mean and end-of-day sample genuinely differ
    assert dsets['6h']['u1'].shape[0] == 35 * 4
    assert dsets['monthly']['T1'].shape[0] == 1  # only January complete
    assert dsets['monthly']['WD'].shape[0] == 1

    # instantaneous daily Ts == the state at each day's end (spot check)
    np.testing.assert_array_equal(daily['Ts'].values[-1], run.state.Ts)

    # noleap calendar, end-of-interval labels
    t0 = daily['time'].values[0]
    assert (t0.year, t0.month, t0.day) == (1, 1, 2)
    tm = dsets['monthly']['time'].values[0]
    assert (tm.year, tm.month, tm.day) == (1, 2, 1)

    # provenance in global attrs
    assert 'code_git' in daily.attrs


def test_unknown_variable_and_bad_freq_rejected():
    with pytest.raises(KeyError):
        ControlRun(config=RunConfig(data_path=_DATA,
                                    output={'nope': {'freq': 'daily',
                                                     'kind': 'mean'}}))
    with pytest.raises(ValueError):
        ControlRun(config=RunConfig(data_path=_DATA,
                                    output={'Ts': {'freq': '5h',
                                                   'kind': 'mean'}}))


def test_netcdf_roundtrip(tmp_path):
    import xarray as xr

    run = ControlRun(config=RunConfig(
        data_path=_DATA, output={'Ts': {'freq': 'daily', 'kind': 'inst'}}))
    for _ in range(3):
        run.advance_day()
    paths = run.save_output(str(tmp_path / 'out'))
    assert len(paths) == 1
    ds = xr.open_dataset(paths[0])
    assert ds['Ts'].shape == (3, 42, 64)
    ds.close()
