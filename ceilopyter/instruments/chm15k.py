import logging
from os import PathLike

import netCDF4
import numpy as np
import numpy.typing as npt
from cftime import num2pydate
from numpy import ma

from ..ceilo import Ceilo
from ..ceilo_raw import CeiloRaw


def read_chm15k(
    files: str | PathLike | list[str | PathLike],
    calibration_factor: float | None = None,
) -> Ceilo:
    if not isinstance(files, list):
        files = [files]
    if calibration_factor is None:
        calibration_factor = 3e-12
        logging.warning("Using default calibration factor: %s", calibration_factor)
    raw = []
    for file in files:
        raw.append(_read_file(file))
    return Ceilo(raw, calibration_factor)


def _read_file(file: str | PathLike) -> CeiloRaw:
    with netCDF4.Dataset(file) as nc:
        time = num2pydate(nc["time"][:], nc["time"].units)
        range = nc["range"][:]
        beta = _get_beta(nc)
        wavelength = nc["wavelength"][:]
        zenith_angle = nc["zenith"][:]
        return CeiloRaw(time, range, beta, wavelength, zenith_angle)


def _get_beta(nc: netCDF4.Dataset) -> npt.NDArray[np.floating]:
    """Get range-corrected backscatter coefficient.

    References:
        Mattis, I. & Wagner, F. (2014). E-PROFILE: Glossary of lidar and
            ceilometer variables. https://www.eumetnet.eu/wp-content/uploads/2016/10/ALC_glossary.pdf
    """
    # When netcdf_mode is 1 (introduced in firmware version 1.050) we have:
    # beta_att = beta_raw * c_cal
    if hasattr(nc, "netcdf_mode") and nc.netcdf_mode == 1:
        return nc["beta_att"][:] / nc["c_cal"][:]

    # When netcdf_mode is 2 (or unspecified in old files), we have beta_raw
    # without calibration factor applied. This mode is recommended in ACTRIS
    # CCRES ALC SOP.
    beta = nc["beta_raw"][:]

    # In old files, beta_raw is normalized by stddev and not range corrected:
    # beta_raw = (P_raw / laser_pulses - base) / stddev
    if _is_old_version(nc):
        stddev = nc["stddev"][:]
        rng = nc["range"][:]
        normalised_apd = _get_nn(nc)
        correction = stddev / normalised_apd
        beta *= correction[:, np.newaxis]
        beta *= rng**2

    return beta


def _is_old_version(nc: netCDF4.Dataset) -> bool:
    version = nc.software_version
    # In old files, the version is a single integer.
    if isinstance(version, np.integer):
        return True
    # In newer files, the version is a space-separated list: Operating system,
    # FPGA, firmware, CloudDetectionMode (added in firmware version 0.747).
    if isinstance(version, str):
        parts = version.split()
        firmware = parts[2]
        return firmware < "0.702"
    msg = f"Cannot determine version: {version}"
    raise ValueError(msg)


def _get_nn(nc: netCDF4.Dataset) -> float:
    """Correct for changing avalanche photodiode (APD) voltage.

    References:
        Wiegner, M. & Geiß, A. (2012): Aerosol profiling with the Jenoptik
            ceilometer CHM15kx. https://doi.org/10.5194/amt-5-1953-2012
    """
    if "nn1" in nc.variables:
        nn1 = nc["nn1"][:]
    elif "NN1" in nc.variables:
        nn1 = nc["NN1"][:]
    else:
        logging.warning("Unable to compute normalized APD: variable nn1 not found")
        return 1
    median_nn1 = ma.median(nn1)
    # Parameters taken from the MATLAB implementation of Cloudnet.
    if 120 < median_nn1 < 160:
        step_factor, reference, scale = 1.24, 140, 5
    elif 3200 < median_nn1 < 4000:
        step_factor, reference, scale = 1.035, 3685, 1
    else:
        logging.warning(
            "Unable to compute normalized APD: "
            "median nn1 (%s) outside of expected range",
            median_nn1,
        )
        return 1
    return step_factor ** (-(nn1 - reference) / scale)
