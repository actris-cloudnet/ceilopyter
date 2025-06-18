import datetime
import logging
from os import PathLike

import numpy as np

from ..ceilo import Ceilo
from ..ceilo_raw import CeiloRaw, concatenate_raw
from ..common import InvalidMessageError


def read_ld40(
    files: str | PathLike | list[str | PathLike],
    calibration_factor: float | None = None,
) -> Ceilo:
    if not isinstance(files, list):
        files = [files]
    if calibration_factor is None:
        calibration_factor = 1.0
        logging.warning("Using default calibration factor: %s", calibration_factor)
    raw = []
    for file in files:
        raw.append(_read_file(file))
    concat = concatenate_raw(raw)
    beta_raw = concat.beta * calibration_factor
    return Ceilo(concat, beta_raw, None, calibration_factor)


def _read_file(filename: str | PathLike) -> CeiloRaw:
    """Read Vaisala LD40 file.

    This function supports .BSC files from Lindenberg.
    """
    times = []
    betas = []
    with open(filename) as f:
        header = f.readline().split()
        n_header = len(header)
        if n_header <= 3 or header[:2] != ["DATE", "TIME"]:
            raise InvalidMessageError("Invalid header")
        rng = np.array([int(rng) for rng in header[2:]])
        for line in f:
            values = line.split()
            if len(values) != n_header:
                raise InvalidMessageError("Values don't match header")
            times.append(
                datetime.datetime.strptime(
                    values[0] + " " + values[1], "%d.%m.%y %H:%M"
                )
            )
            betas.append(np.array([int(v) for v in values[2:]]))
    time = np.array(times)
    beta = np.array(betas) * rng**2 * 1e-12
    return CeiloRaw(time, rng, beta, 855)
