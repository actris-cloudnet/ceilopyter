import logging
from os import PathLike

import numpy as np
import numpy.typing as npt
from numpy import ma

from ..ceilo import Ceilo
from ..ceilo_raw import concatenate_raw
from ..common import read_msgs
from ..noise import remove_noise
from ..readers.read_ct import read_ct_file


def read_ct25k(
    files: str | PathLike | list[str | PathLike],
    calibration_factor: float | None = None,
    noise_h2: bool = True,
) -> Ceilo:
    if calibration_factor is None:
        calibration_factor = 1.0
        logging.warning("Using default calibration factor: %s", calibration_factor)

    raw = read_msgs(files, read_ct_file, wavelength=905.0)
    concat = concatenate_raw(raw)

    r2 = (concat.range * 1e-3) ** 2
    beta_calib = concat.beta * calibration_factor
    beta_uncorr = beta_calib / r2 if noise_h2 else _fix_beta(r2, beta_calib)

    is_noise = remove_noise(beta_uncorr, noise_floor=6e-8)
    beta_raw = beta_uncorr * r2
    beta = ma.masked_where(is_noise, beta_raw)

    return Ceilo(concat, beta_raw, beta, calibration_factor)


def _fix_beta(
    r2: npt.NDArray[np.floating], beta_raw: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    range_broad = np.broadcast_to(r2, beta_raw.shape)
    is_strong = beta_raw > 1e-7
    beta_raw[is_strong] /= range_broad[is_strong]
    return beta_raw
