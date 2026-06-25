import netCDF4
import numpy as np
from numpy import ma

from ceilopyter.lidar import _detect_instrument, read_lidar


def _write_lidar(path, *, with_depol=False):
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("time", 3)
        nc.createDimension("range", 4)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "hours since 2026-06-21 00:00:00 +00:00"
        t[:] = [0.0, 1.0, 2.0]
        nc.createVariable("range", "f4", ("range",))[:] = [15.0, 30.0, 45.0, 60.0]
        profile = np.arange(12).reshape(3, 4)
        nc.createVariable("beta", "f4", ("time", "range"))[:] = profile
        nc.createVariable("beta_raw", "f4", ("time", "range"))[:] = profile
        nc.createVariable("wavelength", "f4")[:] = 910.0
        nc.createVariable("calibration_factor", "f4")[:] = 2.0
        nc.createVariable("zenith_angle", "f4")[:] = 3.0  # scalar in the product
        if with_depol:
            nc.createVariable("depolarisation", "f4", ("time", "range"))[:] = 0.1


def test_detect_instrument():
    assert _detect_instrument("Lufft CHM15k") == "chm15k"
    assert _detect_instrument("Lufft CHM15kx") == "chm15k"
    assert _detect_instrument("Vaisala CL51") == "cl51"
    assert _detect_instrument("Vaisala CL61d") == "cl61"
    assert _detect_instrument("Campbell Scientific CS135") == "cs135"
    assert _detect_instrument("Some unknown lidar") is None


def test_read_lidar_rescreen_false_uses_product_beta(tmp_path):
    path = tmp_path / "lidar.nc"
    _write_lidar(path)
    c = read_lidar(path, rescreen=False)
    assert np.array_equal(ma.filled(c.beta, -1), np.arange(12).reshape(3, 4))


def test_read_lidar_basic(tmp_path):
    path = tmp_path / "lidar.nc"
    _write_lidar(path)
    c = read_lidar(path, rescreen=False)
    assert c.beta.shape == (3, 4)
    assert np.array_equal(c.range, [15.0, 30.0, 45.0, 60.0])
    assert c.wavelength == 910.0
    assert c.calibration_factor == 2.0
    assert c.depol is None
    assert c.zenith_angle.shape == (3,)  # scalar broadcast to the time axis


def test_read_lidar_exposes_depolarisation(tmp_path):
    path = tmp_path / "lidar.nc"
    _write_lidar(path, with_depol=True)
    c = read_lidar(path, rescreen=False)
    assert c.depol is not None
    assert c.depol.shape == (3, 4)
