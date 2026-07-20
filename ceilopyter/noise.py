import datetime

import numpy as np
import numpy.typing as npt
from numpy import ma
from scipy.ndimage import gaussian_filter

from ceilopyter import utils


def remove_noise(
    time: npt.NDArray[np.floating],
    rng: npt.NDArray[np.floating],
    beta_uncorr: npt.NDArray[np.floating],
    noise_floor: float,
    snr_limit: float = 5,
) -> npt.NDArray[np.bool_]:
    n_time, n_range = beta_uncorr.shape
    zero_ranges = np.all(beta_uncorr == 0, axis=0)
    n_zeros = np.argmax(~zero_ranges[::-1])

    is_negative = beta_uncorr < 0

    is_above_negative = _mask_low_values_above_consequent_negatives(beta_uncorr)
    import matplotlib.pyplot as plt

    plt.pcolormesh(is_above_negative.T)
    beta_uncorr = np.where(is_above_negative, noise_floor, beta_uncorr)

    smooth, smooth_uncorr = calc_beta_smooth(time, rng, beta_uncorr)

    fraction = 0.1
    n_top_gates = round(n_range * fraction)
    beta_top = smooth_uncorr[:, n_range - n_zeros - n_top_gates : n_range - n_zeros]
    noise = ma.std(beta_top, axis=1)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.pcolormesh(beta_top.T)

    noise = np.maximum(noise, noise_floor)
    snr = smooth_uncorr / noise[:, np.newaxis]

    is_noise = snr < snr_limit

    return is_noise | is_negative


def calc_beta_smooth(
    time: npt.NDArray[np.floating],
    rng: npt.NDArray[np.floating],
    beta_uncorr: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    cloud_mask, cloud_values, cloud_limit = _estimate_clouds_from_beta(beta_uncorr)

    beta_uncorr = np.where(cloud_mask, cloud_limit, beta_uncorr)
    sigma = _calc_sigma_units(time, rng)
    smooth_uncorr = gaussian_filter(beta_uncorr, sigma)
    smooth_uncorr[cloud_mask] = cloud_values

    smooth = smooth_uncorr * (rng * 1e-3) ** 2
    return smooth, smooth_uncorr


def _calc_sigma_units(
    time: npt.NDArray,
    rng: npt.NDArray[np.floating],
    sigma_minutes: float = 1,
    sigma_meters: float = 10,
) -> tuple[float, float]:
    """Calculates Gaussian peak std parameters.

    The amount of smoothing is hard coded. This function calculates how many
    steps in time and height corresponds to this smoothing.

    Args:
        time: Time.
        rng: Range (m).
        sigma_minutes: Smoothing in minutes.
        sigma_meters: Smoothing in meters.

    Returns:
        tuple: Two element tuple containing number of steps in time and height
            to achieve wanted smoothing.
    """
    time_step = ma.median(np.diff(time))
    alt_step = ma.median(np.diff(rng))
    x_std = datetime.timedelta(minutes=sigma_minutes) / time_step
    y_std = sigma_meters / alt_step
    return x_std, y_std


def _estimate_clouds_from_beta(
    beta_uncorr: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.floating], float]:
    """Naively finds strong clouds from ceilometer backscatter."""
    cloud_limit = 5e-6
    cloud_mask = beta_uncorr > cloud_limit
    return cloud_mask, beta_uncorr[cloud_mask], cloud_limit


def _mask_low_values_above_consequent_negatives(
    beta_uncorr: npt.NDArray,
    n_negatives: int = 5,
    threshold: float = 8e-6,
    n_skip_lowest: int = 5,
) -> npt.NDArray:
    n_time, n_gates = beta_uncorr.shape
    bottom_n_gates = n_gates // 5
    print(n_skip_lowest, "..", bottom_n_gates)
    negative_data = beta_uncorr[:, n_skip_lowest : n_skip_lowest + bottom_n_gates] < 0
    n_consequent_negatives = utils.cumsumr(negative_data, axis=1)
    time_indices, alt_indices = np.where(n_consequent_negatives > n_negatives)
    alt_indices += n_skip_lowest
    is_negative = np.zeros(beta_uncorr.shape, dtype=np.bool_)
    for time_ind, alt_ind in zip(time_indices, alt_indices, strict=True):
        profile = beta_uncorr[time_ind, alt_ind:]
        is_negative[time_ind, alt_ind:][profile < threshold] = True
    return is_negative
