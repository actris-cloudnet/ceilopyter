import logging
from os import PathLike

import netCDF4
import numpy as np
import numpy.typing as npt
from cftime import num2pydate
from numpy import ma

from ceilopyter.noise import remove_noise

from ..ceilo import Ceilo
from ..ceilo_raw import CeiloRaw, concatenate_raw


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
    concat = concatenate_raw(raw)
    beta_raw = concat.beta * calibration_factor

    beta_uncorr = beta_raw / (concat.range * 1e-3) ** 2
    is_noise = remove_noise(concat.time, concat.range, beta_uncorr, noise_floor=4e-9)
    # noise_floor=1e-9 # for beta
    # noise_floor=4e-9 # for beta_smooth
    beta = ma.masked_where(is_noise, beta_raw)

    return Ceilo(concat, beta_raw, beta, calibration_factor)


def _read_file(file: str | PathLike) -> CeiloRaw:
    with netCDF4.Dataset(file) as nc:
        time = num2pydate(nc["time"][:], nc["time"].units)
        range = nc["range"][:]
        beta = _get_beta(nc)
        wavelength = nc["wavelength"][:]
        zenith_angle = nc["zenith"][:] if "zenith" in nc.variables else None
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
    # beta = ((P_raw / lp) - b) / (cs * o(r) * p_calc) * r * r
    beta = nc["beta_raw"][:]

    sw_version = _get_sw_version(nc)
    if isinstance(sw_version, np.integer) and sw_version <= 135:
        # beta_raw = P_raw
        laser_pulses = nc["laser_pulses"][:]
        base = np.mean(beta[:, -5:], axis=1) / laser_pulses
        rng = nc["range"][:]
        overlap = _awful_overlap(rng)
        signal = beta / laser_pulses[:, np.newaxis] - base[:, np.newaxis]
        beta = signal / overlap * rng**2
    elif isinstance(sw_version, np.integer) or sw_version < "0.702":
        # beta_raw = (P_raw / laser_pulses - base) / stddev
        apd = _get_nn(nc)
        rng = nc["range"][:]
        overlap = _awful_overlap(rng)
        stddev = nc["stddev"][:]
        signal = beta * stddev[:, np.newaxis]
        beta = signal / (overlap * apd[:, np.newaxis]) * rng**2

    return beta


def _get_sw_version(nc: netCDF4.Dataset) -> np.integer | str:
    version = nc.software_version
    # In old files, the version is a single integer.
    if isinstance(version, np.integer):
        return version
    # In newer files, the version is a space-separated list: Operating system,
    # FPGA, firmware, CloudDetectionMode (added in firmware version 0.747).
    if isinstance(version, str):
        parts = version.split()
        return parts[2]
    msg = f"Cannot determine version: {version}"
    raise ValueError(msg)


def _awful_overlap(rng):
    n = len(rng)
    i = np.nonzero(rng > 200)[0][0]
    j = np.nonzero(rng > 800)[0][0]
    zero = np.zeros(i)
    mids = np.linspace(0, 1, j - i)
    ones = np.ones(n - j)
    return np.concatenate([zero, mids, ones])


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
    # print(nn1)
    # asd = ma.compressed(nn1)
    # asd = asd[asd > 0]
    # print(asd)
    if len(nn1) <= 1 or nn1[1] is ma.masked or nn1[1] == 0:
        return np.array([1])
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
        return np.array([1])
    return step_factor ** (-(nn1 - reference) / scale)
