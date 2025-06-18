import concurrent.futures
import logging
from os import PathLike

import netCDF4
from cftime import num2pydate

from ..ceilo import Ceilo
from ..ceilo_raw import CeiloRaw, concatenate_raw


def read_cl61(
    files: str | PathLike | list[str | PathLike],
    calibration_factor: float | None = None,
) -> Ceilo:
    if calibration_factor is None:
        calibration_factor = 1.0
        logging.warning("Using default calibration factor: %s", calibration_factor)
    if isinstance(files, list):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            raw = list(executor.map(_read_file, files))
    else:
        raw = [_read_file(files)]
    concat = concatenate_raw(raw)
    beta_raw = concat.beta * calibration_factor
    return Ceilo(concat, beta_raw, None, calibration_factor)


def _read_file(file: str | PathLike) -> CeiloRaw:
    with netCDF4.Dataset(file) as nc:
        time = num2pydate(nc["time"][:], nc["time"].units)
        range = nc["range"][:]
        beta = nc["beta_att"][:]
        zenith_angle = nc["tilt_angle"][:]
        return CeiloRaw(time, range, beta, 910.55, zenith_angle)
