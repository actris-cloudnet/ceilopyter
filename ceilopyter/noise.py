import numpy as np
import numpy.typing as npt
from numpy import ma


def remove_noise(
    beta_uncorr: npt.NDArray[np.floating], noise_floor: float, snr_limit: float = 5
) -> npt.NDArray[np.bool]:
    zero_ranges = np.all(beta_uncorr == 0, axis=0)
    n_zeros = np.argmax(~zero_ranges[::-1])

    fraction = 0.1
    n_top_gates = round(beta_uncorr.shape[1] * fraction)
    beta_top = beta_uncorr[:, -n_top_gates - n_zeros : -n_zeros]
    noise = ma.std(beta_top, axis=1)

    noise = np.maximum(noise, noise_floor)
    snr = beta_uncorr / noise[:, np.newaxis]

    is_noise = snr < snr_limit

    return is_noise
