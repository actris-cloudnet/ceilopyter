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
        serial_number: str | None = None,
    ):
        self.time = time
        self.range = range
        self.beta = beta
        self.wavelength = wavelength
        self.zenith_angle = zenith_angle
        self.serial_number = serial_number


def concatenate_raw(raw: list[CeiloRaw]) -> CeiloRaw:
    if len(raw) == 0:
        raise ValueError("No data given")
    if len(raw) == 1:
        return raw[0]

    serial_number = raw[0].serial_number
    if any(r.serial_number != serial_number for r in raw):
        raise ValueError("Inconsistent serial number")

    wavelength = raw[0].wavelength
    if any(r.wavelength != wavelength for r in raw):
        raise ValueError("Inconsistent wavelength")

    all_time = np.concatenate([r.time for r in raw])
    n_time = len(all_time)

    uniq_time, time_ind = np.unique(all_time, return_index=True)
    all_zenith_angle = (
        None
        if all(r.zenith_angle is None for r in raw)
        else np.concatenate(
            [np.broadcast_to(r.zenith_angle, len(r.time)) for r in raw]  # type: ignore[arg-type]
        )[time_ind]
    )

    all_rngs = [r.range for r in raw]
    max_rng = max(all_rngs, key=len)
    for rng in all_rngs:
        if not np.array_equal(rng, max_rng[: len(rng)]):
            raise ValueError("Inconsistent ranges")

    all_beta = ma.masked_all((n_time, len(max_rng)))
    i = 0
    for r in raw:
        all_beta[i : i + len(r.time), : len(r.range)] = r.beta
        i += len(r.time)
    all_beta = all_beta[time_ind]

    return CeiloRaw(
        uniq_time,
        max_rng,
        all_beta,
        wavelength,
        all_zenith_angle,
        serial_number,
    )
