import logging
from os import PathLike

from ..ceilo import Ceilo
from ..common import read_msgs
from ..readers.read_cl import read_cl_file


def read_cl51(
    files: str | PathLike | list[str | PathLike],
    calibration_factor: float | None = None,
) -> Ceilo:
    if calibration_factor is None:
        calibration_factor = 1.0
        logging.warning("Using default calibration factor: %s", calibration_factor)
    raw = read_msgs(files, read_cl_file, 910.0)
    return Ceilo(raw, calibration_factor)
