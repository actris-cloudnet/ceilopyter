import datetime

import numpy as np
from numpy import ma

from ceilopyter.average import average_time
from ceilopyter.ceilo import Ceilo
from ceilopyter.ceilo_raw import CeiloRaw


def _ceilo(n_time=6, step_s=10):
    t0 = datetime.datetime(2025, 6, 14, 0, 0, 0)
    times = np.array(
        [t0 + datetime.timedelta(seconds=step_s * i) for i in range(n_time)]
    )
    rng = np.arange(4) * 30.0
    beta = ma.array(np.arange(n_time * 4, dtype=float).reshape(n_time, 4))
    raw = CeiloRaw(times, rng, beta, 910.0)
    return Ceilo(raw, ma.filled(beta, 0.0), beta, 1.0)


def test_average_time_bins_and_means():
    c = _ceilo()  # 6 profiles, 10 s apart (0..50 s)
    avg = average_time(c, 30)  # bins [0,30): profiles 0-2, [30,60): profiles 3-5
    assert avg.beta.shape == (2, 4)
    assert np.array_equal(avg.range, c.range)  # range preserved
    assert np.allclose(avg.beta[0], ma.asarray(c.beta)[:3].mean(axis=0))
    assert np.allclose(avg.beta[1], ma.asarray(c.beta)[3:].mean(axis=0))


def test_average_time_handles_missing_depol():
    c = _ceilo()
    assert c.depol is None
    assert average_time(c, 30).depol is None


def test_average_time_handles_scalar_zenith():
    # chm15k exposes zenith_angle as a 0-d scalar, which has no time axis.
    c = _ceilo()
    c.zenith_angle = np.array(5.0)
    avg = average_time(c, 30)
    assert avg.beta.shape == (2, 4)
    assert np.ndim(avg.zenith_angle) == 0


def test_average_time_averages_internal_temperature():
    c = _ceilo()  # 6 profiles, 10 s apart
    c.internal_temperature = np.arange(6, dtype=float) + 273.15
    avg = average_time(c, 30)  # bins of 3 profiles each
    assert avg.internal_temperature is not None
    assert np.isclose(avg.internal_temperature[0], np.mean([0, 1, 2]) + 273.15)
    assert np.isclose(avg.internal_temperature[1], np.mean([3, 4, 5]) + 273.15)


def test_average_time_handles_missing_internal_temperature():
    c = _ceilo()
    assert c.internal_temperature is None
    assert average_time(c, 30).internal_temperature is None
