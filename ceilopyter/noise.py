import numpy as np
import numpy.typing as npt
from numpy import ma

NOISE_FLOORS: dict[str, float] = {
    "cl31": 4e-7,
    "cl51": 8e-8,
    "cl61": 2e-8,
    "chm15k": 5e-8,
    "cs135": 1e-7,
    "ct25k": 6e-8,
}
"""Per-instrument minimum noise level for `screen_noise` (see each reader)."""


def remove_noise(
    beta_uncorr: npt.NDArray[np.floating], noise_floor: float, snr_limit: float = 5
) -> npt.NDArray[np.bool_]:
    zero_ranges = np.all(beta_uncorr == 0, axis=0)
    n_zeros = np.argmax(~zero_ranges[::-1])

    fraction = 0.1
    n_top_gates = round(beta_uncorr.shape[1] * fraction)
    # When no top gates are zero-padded (e.g. netCDF instruments), n_zeros is 0
    # and the slice end must be None rather than -0 (which would be empty).
    end = -n_zeros if n_zeros > 0 else None
    beta_top = beta_uncorr[:, -n_top_gates - n_zeros : end]
    noise = ma.std(beta_top, axis=1)

    noise = np.maximum(noise, noise_floor)
    snr = beta_uncorr / noise[:, np.newaxis]

    is_noise = snr < snr_limit

    return is_noise


def screen_noise(
    beta: npt.NDArray[np.floating],
    range: npt.NDArray[np.floating],
    noise_floor: float,
    snr_limit: float = 5,
) -> npt.NDArray[np.floating]:
    """Mask noise in range-corrected backscatter.

    The backscatter is un-range-corrected before estimating the noise level and
    the resulting mask is applied to the original range-corrected values.

    Args:
        beta: Range-corrected backscatter coefficient (sr-1 m-1).
        range: Range (m).
        noise_floor: Minimum noise level used when the profile is too clear to
            estimate noise from its top gates.
        snr_limit: Signal-to-noise ratio below which a value is masked.
    """
    r2 = (range * 1e-3) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_uncorr = ma.masked_invalid(beta / r2)
    is_noise = remove_noise(beta_uncorr, noise_floor=noise_floor, snr_limit=snr_limit)
    return ma.masked_where(is_noise, beta)
