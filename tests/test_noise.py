import numpy as np
from numpy import ma

from ceilopyter.noise import remove_noise, screen_noise


def _profiles():
    """Synthetic backscatter with no zero-padded top gates (e.g. netCDF data).

    Low gates carry strong signal; upper gates are pure noise.
    """
    rng = np.random.default_rng(0)
    n_time, n_gates = 50, 200
    beta = rng.normal(0, 1e-8, (n_time, n_gates))
    beta[:, :10] += 1e-5
    return beta


def test_remove_noise_without_zero_padding():
    # Regression: when no top gates are all-zero, the noise estimate must still
    # be computed (the slice end is None, not -0 which yields an empty window).
    beta = _profiles()
    assert not np.any(np.all(beta == 0, axis=0))  # no zero-padded gates

    is_noise = remove_noise(beta, noise_floor=1e-9)

    assert not ma.is_masked(is_noise)  # not degenerate
    assert not is_noise[:, :10].any()  # strong signal is kept
    assert is_noise[:, -10:].mean() > 0.9  # noisy top gates are flagged


def test_screen_noise_masks_clear_air():
    beta = _profiles()
    rng = np.arange(beta.shape[1]) * 30.0 + 15.0  # range in m, strictly positive

    screened = screen_noise(beta, rng, noise_floor=1e-9)

    assert ma.getmaskarray(screened)[:, -10:].all()  # clear air masked
    assert not ma.getmaskarray(screened)[:, :10].any()  # signal preserved
