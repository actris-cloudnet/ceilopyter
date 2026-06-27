from pathlib import Path

import netCDF4
import numpy as np
import pytest

import ceilopyter
from ceilopyter.overlap import (
    OverlapModel,
    read_overlap_model,
    read_reference_overlap,
    write_overlap_model,
)

_CFG = (
    Path(ceilopyter.__file__).parent
    / "overlap_functions"
    / "TUB120011_20121112_1024.cfg"
)


def _model(n=1024):
    rng = np.arange(n) * 14.985
    return OverlapModel(
        range=rng,
        alpha=np.linspace(-1.0, 1.0, n),
        intercept=np.linspace(0.0, 0.5, n),
        reference_overlap=np.linspace(0.0, 1.0, n),
        optical_module_id="TUB120011",
        temp_valid_min=-10.0,
        temp_valid_max=30.0,
        wavelength=1064.0,
        attrs={"n_samples": "42", "source": "test"},
    )


def test_read_reference_overlap():
    ref = read_reference_overlap(_CFG)
    assert ref.overlap.shape == (1024,)
    assert ref.serial == "TUB120011"
    assert ref.overlap[0] < 1e-3  # near zero overlap at the ground
    assert np.isclose(ref.overlap[-1], 1.0)  # full overlap far away
    assert ref.overlap.max() <= 1.02
    assert ref.metadata["scaling"] == "0.359762"


def test_read_reference_overlap_rejects_bad_header(tmp_path):
    path = tmp_path / "bad.cfg"
    path.write_text("not_ovl\n1 2 3\n")
    with pytest.raises(ValueError, match="ovl"):
        read_reference_overlap(path)


def test_overlap_model_round_trip(tmp_path):
    model = _model()
    path = tmp_path / "model.nc"
    write_overlap_model(model, path)
    back = read_overlap_model(path)
    assert np.allclose(back.range, model.range)
    assert np.allclose(back.alpha, model.alpha)
    assert np.allclose(back.intercept, model.intercept)
    assert np.allclose(back.reference_overlap, model.reference_overlap)
    assert back.optical_module_id == "TUB120011"
    assert back.temp_valid_min == -10.0
    assert back.temp_valid_max == 30.0
    assert back.wavelength == 1064.0
    assert back.attrs["n_samples"] == "42"
    assert back.attrs["source"] == "test"


def test_read_overlap_model_missing_variable(tmp_path):
    path = tmp_path / "incomplete.nc"
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("range", 3)
        nc.createVariable("range", "f4", ("range",))[:] = [0, 1, 2]
        nc.wavelength = 1064.0
    with pytest.raises(ValueError, match="missing variable: alpha"):
        read_overlap_model(path)
