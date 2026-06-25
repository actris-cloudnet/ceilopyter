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
        depol: Linear depolarization ratio (CL61 only, else None)
    """

    def __init__(
        self,
        time: npt.NDArray[np.object_],
        range: npt.NDArray[np.floating],
        beta: npt.NDArray[np.floating],
        wavelength: float,
        zenith_angle: npt.NDArray[np.floating] | None = None,
        depol: npt.NDArray[np.floating] | None = None,
    ):
        self.time = time
        self.range = range
        self.beta = beta
        self.wavelength = wavelength
        self.zenith_angle = zenith_angle
        self.depol = depol


def concatenate_raw(raw: list[CeiloRaw]) -> CeiloRaw:
    if len(raw) == 0:
        raise ValueError("No data given")
    if len(raw) == 1:
        return raw[0]

    all_time = np.concatenate([r.time for r in raw])
    all_time, time_ind = np.unique(all_time, return_index=True)

    wavelength = raw[0].wavelength
    if any(r.wavelength != wavelength for r in raw):
        raise ValueError("Inconsistent wavelength")

    all_rngs = [r.range for r in raw]
    max_rng = max(all_rngs, key=len)
    for rng in all_rngs:
        if not np.array_equal(rng, max_rng[: len(rng)]):
            raise ValueError("Inconsistent ranges")

    # Fill in the original (possibly time-overlapping) order, then select the
    # unique, sorted rows with `time_ind`. The buffer must be sized to the total
    # number of profiles, not the unique count, or overlapping timestamps across
    # files overflow the slice assignment.
    n_total = sum(len(r.time) for r in raw)

    def concat_field(attr: str, *, per_gate: bool) -> npt.NDArray[np.floating] | None:
        if all(getattr(r, attr) is None for r in raw):
            return None
        shape = (n_total, len(max_rng)) if per_gate else (n_total,)
        field = ma.masked_all(shape)
        i = 0
        for r in raw:
            values = getattr(r, attr)
            if values is not None:
                if per_gate:
                    field[i : i + len(r.time), : len(r.range)] = values
                else:
                    field[i : i + len(r.time)] = values
            i += len(r.time)
        return field[time_ind]

    all_beta = concat_field("beta", per_gate=True)
    assert all_beta is not None  # beta is always present

    return CeiloRaw(
        all_time,
        max_rng,
        all_beta,
        wavelength,
        concat_field("zenith_angle", per_gate=False),
        depol=concat_field("depol", per_gate=True),
    )
