import logging
from os import PathLike

from ..ceilo import Ceilo
from ..ceilo_raw import concatenate_raw
from ..common import read_msgs
from ..readers.read_cl import read_cl_file


def read_cl31(
    files: str | PathLike | list[str | PathLike],
    calibration_factor: float | None = None,
) -> Ceilo:
    if calibration_factor is None:
        calibration_factor = 1.0
        logging.warning("Using default calibration factor: %s", calibration_factor)
    raw = read_msgs(files, read_cl_file, wavelength=910.0)
    concat = concatenate_raw(raw)
    beta_raw = concat.beta * calibration_factor
    return Ceilo(concat, beta_raw, None, calibration_factor)
