import logging
from os import PathLike

from ceilopyter.ceilo_raw import concatenate_raw

from ..ceilo import Ceilo
from ..common import read_msgs
from ..readers.read_cs import read_cs_file


def read_cs135(
    files: str | PathLike | list[str | PathLike],
    calibration_factor: float | None = None,
) -> Ceilo:
    if calibration_factor is None:
        calibration_factor = 1.0
        logging.warning("Using default calibration factor: %s", calibration_factor)
    raw = read_msgs(files, read_cs_file, wavelength=905.0)
    concat = concatenate_raw(raw)
    beta_raw = concat.beta * calibration_factor
    return Ceilo(concat, beta_raw, None, calibration_factor)
