import concurrent.futures
import logging
from os import PathLike

import netCDF4
from cftime import num2pydate
from numpy import ma

from ..ceilo import Ceilo
from ..ceilo_raw import CeiloRaw, concatenate_raw
from ..noise import NOISE_FLOORS, screen_noise

NOISE_FLOOR = NOISE_FLOORS["cl61"]


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
    beta = screen_noise(beta_raw, concat.range, noise_floor=NOISE_FLOOR)
    # The raw depolarization ratio is garbage where there is no signal, so reuse
    # the backscatter noise mask to keep only the meaningful values.
    if concat.depol is not None:
        concat.depol = ma.masked_where(ma.getmaskarray(beta), concat.depol)
    return Ceilo(concat, beta_raw, beta, calibration_factor)


def _read_file(file: str | PathLike) -> CeiloRaw:
    with netCDF4.Dataset(file) as nc:
        time = num2pydate(nc["time"][:], nc["time"].units)
        range = nc["range"][:]
        beta = nc["beta_att"][:]
        zenith_angle = nc["tilt_angle"][:]
        # Older firmware/exports may lack depolarization; treat it as absent
        # rather than failing the whole read.
        depol = (
            nc["linear_depol_ratio"][:]
            if "linear_depol_ratio" in nc.variables
            else None
        )
        return CeiloRaw(time, range, beta, 910.55, zenith_angle, depol=depol)
