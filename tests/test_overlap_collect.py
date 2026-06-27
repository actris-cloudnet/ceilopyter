import netCDF4
import numpy as np

from training.collect import CollectConfig, collect_overlap_samples

_RANGE = np.arange(101) * 15.0  # 0..1500 m
_REF = np.clip(_RANGE / 300.0, 0.0, 1.0)  # full overlap at/above 300 m


def _ratio():
    """True overlap / reference overlap: a 10 % near-range deficit below 300 m."""
    ratio = np.ones_like(_RANGE)
    ratio[(_RANGE >= 150.0) & (_RANGE < 300.0)] = 0.9
    return ratio


def _write_day(
    path, *, sci=0.0, cloud_base=5000.0, temp_k=268.15, jitter=0.0, ratio=None
):
    if ratio is None:
        ratio = _ratio()
    n_time = 91  # 1-min profiles over 90 min
    rng_default = np.random.default_rng(0)
    molecular = 10.0 ** (-0.0005 * _RANGE)
    profile = molecular * ratio
    beta = np.tile(profile, (n_time, 1))
    if jitter:
        beta = beta * (1.0 + jitter * rng_default.standard_normal((n_time, 1)))
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("time", n_time)
        nc.createDimension("range", _RANGE.size)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "seconds since 2024-02-01 00:00:00 +00:00"
        t[:] = np.arange(n_time) * 60.0
        nc.createVariable("range", "f4", ("range",))[:] = _RANGE
        nc.createVariable("beta_raw", "f4", ("time", "range"))[:] = beta
        nc.createVariable("temp_int", "f4", ("time",))[:] = np.full(n_time, temp_k)
        nc.createVariable("sci", "f4", ("time",))[:] = np.full(n_time, sci)
        nc.createVariable("cbh", "f4", ("time",))[:] = np.full(n_time, cloud_base)


def test_collects_clear_sky_sample_and_recovers_overlap(tmp_path):
    path = tmp_path / "day.nc"
    _write_day(path)
    samples = collect_overlap_samples(path, _REF, _RANGE)
    assert samples  # at least one window accepted
    expected = _REF * _ratio()
    for s in samples:
        assert np.allclose(s.overlap, expected, atol=1e-6)
        assert np.isclose(s.internal_temperature, 268.15)
        assert np.isclose(s.range_start, 300.0)
        assert np.isclose(s.range_end, 1200.0)
        assert np.isclose(s.range_resolution, 15.0)


def test_rejects_cloudy_profiles(tmp_path):
    path = tmp_path / "cloudy.nc"
    _write_day(path, sci=1.0)  # not clear sky
    assert collect_overlap_samples(path, _REF, _RANGE) == []


def test_rejects_low_cloud_base(tmp_path):
    path = tmp_path / "lowcloud.nc"
    _write_day(path, cloud_base=400.0)  # cloud below the fit band
    assert collect_overlap_samples(path, _REF, _RANGE) == []


def test_rejects_unstable_signal(tmp_path):
    path = tmp_path / "unstable.nc"
    _write_day(path, jitter=0.2)  # 20 % temporal variability >> max_std_over_mean
    assert collect_overlap_samples(path, _REF, _RANGE) == []


def test_config_tightens_acceptance(tmp_path):
    path = tmp_path / "day.nc"
    _write_day(path)
    # Requiring more profiles than a window holds yields no samples.
    config = CollectConfig(min_profiles=1000)
    assert collect_overlap_samples(path, _REF, _RANGE, config=config) == []
