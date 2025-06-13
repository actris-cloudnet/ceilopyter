import numpy as np
import numpy.typing as npt
from numpy import ma


class CeiloRaw:
    """Raw ceilometer data.

    Attributes:
        time: Time
        range: Range (m)
        beta: Range-corrected backscatter coefficient (sr-1 m-1)
        wavelength: Wavelength (nm)
        zenith_angle: Zenith angle (deg)
    """

    def __init__(
        self,
        time: npt.NDArray[np.object_],
        range: npt.NDArray[np.floating],
        beta: npt.NDArray[np.floating],
        wavelength: float,
        zenith_angle: npt.NDArray[np.floating] | None = None,
    ):
        self.time = time
        self.range = range
        self.beta = beta
        self.wavelength = wavelength
        self.zenith_angle = zenith_angle


def concatenate_raw(raw: list[CeiloRaw]) -> CeiloRaw:
    if len(raw) == 0:
        raise ValueError("No data given")
    if len(raw) == 1:
        return raw[0]

    all_time = np.concatenate([r.time for r in raw])
    all_time, time_ind = np.unique(all_time, return_index=True)
    all_zenith_angle = (
        None
        if all(r.zenith_angle is None for r in raw)
        else np.concatenate([r.zenith_angle for r in raw])[time_ind]
    )

    wavelength = raw[0].wavelength
    if any(r.wavelength != wavelength for r in raw):
        raise ValueError("Inconsistent wavelength")

    all_rngs = [r.range for r in raw]
    max_rng = max(all_rngs, key=len)
    for rng in all_rngs:
        if not np.array_equal(rng, max_rng[: len(rng)]):
            raise ValueError("Inconsistent ranges")

    all_beta = ma.masked_all((len(all_time), len(max_rng)))
    i = 0
    for r in raw:
        all_beta[i : i + len(r.time), : len(r.range)] = r.beta
        i += len(r.time)
    all_beta = all_beta[time_ind]

    return CeiloRaw(
        all_time,
        max_rng,
        all_beta,
        wavelength,
        all_zenith_angle,
    )
