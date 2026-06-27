import datetime

import netCDF4
import numpy as np
import pytest
from numpy import ma

from ceilopyter import OverlapModel, correct_overlap, read_chm15k
from ceilopyter.ceilo import Ceilo
from ceilopyter.ceilo_raw import CeiloRaw

_WAVELENGTH = 1064.0


def _model(rng, *, alpha=2.0, intercept=0.0, tmin=-30.0, tmax=40.0):
    rng = np.asarray(rng, dtype=float)
    return OverlapModel(
        range=rng,
        alpha=np.full_like(rng, alpha),
        intercept=np.full_like(rng, intercept),
        reference_overlap=np.ones_like(rng),
        optical_module_id="TUB120011",
        temp_valid_min=tmin,
        temp_valid_max=tmax,
        wavelength=_WAVELENGTH,
    )


def _ceilo(temp_k, n_range=5):
    temp_k = np.asarray(temp_k, dtype=float)
    n_time = len(temp_k)
    t0 = datetime.datetime(2025, 1, 1)
    times = np.array([t0 + datetime.timedelta(seconds=i) for i in range(n_time)])
    rng = np.arange(n_range) * 30.0
    beta_raw = ma.ones((n_time, n_range))
    raw = CeiloRaw(times, rng, beta_raw, _WAVELENGTH, internal_temperature=temp_k)
    return Ceilo(raw, beta_raw, ma.array(beta_raw), 3e-12)


def test_divides_by_temperature_factor():
    # alpha=2 %/degC, T_C=10 -> factor = 1 + (2*10)/100 = 1.2
    c = _ceilo([283.15])  # 10 degC
    out = correct_overlap(c, _model(c.range, alpha=2.0))
    assert np.allclose(out.beta_raw, 1.0 / 1.2)


def test_identity_when_coefficients_zero():
    c = _ceilo([283.15])
    out = correct_overlap(c, _model(c.range, alpha=0.0, intercept=0.0))
    assert np.allclose(out.beta_raw, 1.0)


def test_cold_and_warm_profiles_differ():
    c = _ceilo([253.15, 293.15])  # -20 and +20 degC
    out = correct_overlap(c, _model(c.range, alpha=2.0))
    cold = out.beta_raw[0]
    warm = out.beta_raw[1]
    assert np.allclose(cold, 1.0 / (1.0 + 2.0 * -20.0 / 100.0))
    assert np.allclose(warm, 1.0 / (1.0 + 2.0 * 20.0 / 100.0))
    assert not np.allclose(cold, warm)


def test_min_range_leaves_near_gates_untouched():
    c = _ceilo([283.15])  # factor 1.2 where corrected
    out = correct_overlap(c, _model(c.range, alpha=2.0), min_range=45.0)
    # range = [0, 30, 60, 90, 120]; gates below 45 m stay at 1.0
    assert np.allclose(out.beta_raw[0, :2], 1.0)
    assert np.allclose(out.beta_raw[0, 2:], 1.0 / 1.2)


def test_temperature_is_clipped_to_valid_window():
    c = _ceilo([373.15])  # 100 degC, far above tmax=40
    out = correct_overlap(c, _model(c.range, alpha=2.0, tmax=40.0), clip_temp=True)
    assert np.allclose(out.beta_raw, 1.0 / (1.0 + 2.0 * 40.0 / 100.0))


def test_requires_internal_temperature():
    c = _ceilo([283.15])
    c.internal_temperature = None
    with pytest.raises(ValueError, match="internal_temperature"):
        correct_overlap(c, _model(c.range))


def _write_chm15k(path, *, temps_k):
    n_time = len(temps_k)
    n_range = 5
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("time", n_time)
        nc.createDimension("range", n_range)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "seconds since 2025-01-01 00:00:00 +00:00"
        t[:] = np.arange(n_time)
        nc.createVariable("range", "f4", ("range",))[:] = np.arange(n_range) * 30.0
        nc.createVariable("beta_raw", "f4", ("time", "range"))[:] = np.ones(
            (n_time, n_range)
        )
        nc.createVariable("wavelength", "f4")[:] = _WAVELENGTH
        nc.createVariable("zenith", "f4")[:] = 0.0
        nc.createVariable("temp_int", "f4", ("time",))[:] = np.asarray(temps_k)
        nc.software_version = "12.12.1 2.13 1.040 0"


def test_read_chm15k_applies_overlap_model(tmp_path):
    path = tmp_path / "chm15k.nc"
    _write_chm15k(path, temps_k=[283.15])  # 10 degC
    model = _model(np.arange(5) * 30.0, alpha=2.0)
    uncorrected = read_chm15k(path, calibration_factor=1.0)
    corrected = read_chm15k(path, calibration_factor=1.0, overlap_model=model)
    assert corrected.internal_temperature is not None
    # beta_raw was 1.0 * calibration; correction divides by factor 1.2
    assert np.allclose(corrected.beta_raw, uncorrected.beta_raw / 1.2)
