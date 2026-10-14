import datetime
import logging
import os.path
from os import PathLike

import numpy as np
from numpy import ma

from ceilopyter.noise import remove_noise
from ceilopyter.utils import awful_overlap

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
        try:
            raw.append(_read_file(file))
        except InvalidMessageError as e:
            logging.warning("Failed to read %s: %s", os.path.basename(file), e)
            continue
    concat = concatenate_raw(raw)
    r2 = (concat.range * 1e-3) ** 2
    beta_raw = concat.beta * calibration_factor
    beta_uncorr = beta_raw / r2
    cloud_mask = beta_uncorr > 1e-5
    is_noise, beta_smooth = remove_noise(
        concat.time,
        concat.range,
        beta_uncorr,
        noise_floor=0,
        cloud_mask=cloud_mask,
        snr_limit=12,
    )
    # Filter constant artefact at 8 km and assume noise above.
    is_noise[:, concat.range > 7500] = True
    beta = ma.masked_where(is_noise, beta_raw)
    beta_smooth = ma.masked_where(is_noise, beta_smooth)

    return Ceilo(
        concat,
        beta_raw=beta_raw,
        beta=beta,
        beta_smooth=beta_smooth,
        calibration_factor=calibration_factor,
    )


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
    if len(times) == 0:
        raise InvalidMessageError("No data")
    res = rng[1] - rng[0]
    rng = rng + res / 2
    time = np.array(times)
    overlap = awful_overlap(rng, 100, 200)
    beta = np.array(betas)
    beta = ma.divide(beta, overlap) * rng**2 * 1e-12
    return CeiloRaw(time, rng, beta, 855)
