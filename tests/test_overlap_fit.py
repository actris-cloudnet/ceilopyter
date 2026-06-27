import datetime

import numpy as np
import pytest

from ceilopyter import correct_overlap
from ceilopyter.ceilo import Ceilo
from ceilopyter.ceilo_raw import CeiloRaw
from ceilopyter.overlap.model import OverlapSample
from training.fit import FitConfig, fit_temperature_model

_WAVELENGTH = 1064.0
_RANGE = np.arange(6) * 30.0
_REF = np.linspace(0.5, 1.0, 6)
_ALPHA = np.linspace(-2.0, 2.0, 6)
_INTERCEPT = np.linspace(0.5, -0.5, 6)


def _samples(temps_c, *, per_day=15):
    """Build noiseless samples whose overlap matches the apply formula exactly."""
    t0 = datetime.datetime(2024, 1, 1)
    samples = []
    for day, t_c in enumerate(temps_c):
        overlap = _REF * (1.0 + (_ALPHA * t_c + _INTERCEPT) / 100.0)
        for k in range(per_day):
            time = t0 + datetime.timedelta(days=day, seconds=k)
            samples.append(
                OverlapSample(
                    time=time,
                    range_start=_RANGE[1],
                    range_end=_RANGE[-1],
                    internal_temperature=t_c + 273.15,
                    range_resolution=30.0,
                    overlap=overlap,
                )
            )
    return samples


def _fit(temps_c, *, config=None, **kwargs):
    return fit_temperature_model(
        _samples(temps_c, **kwargs),
        _REF,
        _RANGE,
        "TUB120011",
        _WAVELENGTH,
        config=config,
    )


def test_recovers_injected_coefficients():
    temps = np.linspace(-10.0, 12.0, 12)
    model = _fit(temps)
    assert np.allclose(model.alpha, _ALPHA, atol=1e-6)
    assert np.allclose(model.intercept, _INTERCEPT, atol=1e-6)
    assert np.allclose(model.reference_overlap, _REF)
    assert model.optical_module_id == "TUB120011"
    assert np.isclose(model.temp_valid_min, -10.0)
    assert np.isclose(model.temp_valid_max, 12.0)
    assert model.attrs["n_days"] == "12"


def test_fitted_model_round_trips_through_apply():
    temps = np.linspace(-10.0, 12.0, 12)
    model = _fit(temps)
    # A profile at 5 degC should be corrected by the factor the model encodes.
    beta_raw = np.ma.ones((1, 6))
    raw = CeiloRaw(
        np.array([datetime.datetime(2025, 1, 1)]),
        _RANGE,
        beta_raw,
        _WAVELENGTH,
        internal_temperature=np.array([5.0 + 273.15]),
    )
    ceilo = Ceilo(raw, beta_raw, np.ma.array(beta_raw), 1.0)
    out = correct_overlap(ceilo, model)
    expected_factor = 1.0 + (_ALPHA * 5.0 + _INTERCEPT) / 100.0
    assert np.allclose(out.beta_raw[0], 1.0 / expected_factor)


def test_too_few_days_raises():
    temps = np.linspace(-10.0, 12.0, 5)  # only 5 qualifying days
    with pytest.raises(ValueError, match="qualifying day"):
        _fit(temps, config=FitConfig(min_days=10))


def test_sparse_days_are_dropped():
    # 9 full days (min_samples_per_day met) + 1 sparse day -> 9 qualifying < 10.
    full = _samples(np.linspace(-10.0, 8.0, 9), per_day=15)
    sparse = _samples([12.0], per_day=3)
    with pytest.raises(ValueError, match="qualifying day"):
        fit_temperature_model(full + sparse, _REF, _RANGE, "TUB120011", _WAVELENGTH)


def test_narrow_temperature_span_raises():
    temps = np.full(12, 5.0)  # enough days, no temperature spread
    with pytest.raises(ValueError, match="narrow"):
        _fit(temps)
