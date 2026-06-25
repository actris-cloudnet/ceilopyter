"""Time-averaging of ceilometer data.

Averaging in time only (range resolution is preserved, so thin liquid layers
stay detectable) reduces noise and speeds up classification and plotting.
"""

import datetime

import numpy as np
import numpy.typing as npt
from numpy import ma

from .ceilo import Ceilo
from .ceilo_raw import CeiloRaw


def average_time(ceilo: Ceilo, resolution: float) -> Ceilo:
    """Average a Ceilo into fixed-width time bins.

    Args:
        ceilo: Input data (time must be sorted, as produced by the readers).
        resolution: Bin width in seconds.

    Returns:
        A new Ceilo on the coarsened time grid.
    """
    times = ceilo.time
    reference = times[0]
    seconds = np.array([(t - reference).total_seconds() for t in times])
    bins = (seconds // resolution).astype(int)
    unique = np.unique(bins)
    starts = np.searchsorted(bins, unique, side="left")
    ends = np.searchsorted(bins, unique, side="right")

    n_profiles = len(bins)

    def average(field: npt.NDArray | None) -> npt.NDArray | None:
        if field is None:
            return None
        values = ma.asarray(field)
        if values.ndim == 0 or values.shape[0] != n_profiles:
            return values  # not per-profile (e.g. a single zenith angle)
        out = ma.masked_all((len(unique), *values.shape[1:]))
        for i, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
            out[i] = ma.mean(values[lo:hi], axis=0)
        return out

    new_seconds = np.array(
        [seconds[lo:hi].mean() for lo, hi in zip(starts, ends, strict=True)]
    )
    new_time = np.array(
        [reference + datetime.timedelta(seconds=float(s)) for s in new_seconds]
    )

    beta_raw = average(ceilo.beta_raw)
    assert beta_raw is not None  # beta_raw is always present
    beta = average(ceilo.beta)
    zenith = None if ceilo.zenith_angle is None else average(ceilo.zenith_angle)

    raw = CeiloRaw(
        new_time,
        ceilo.range,
        beta_raw,
        ceilo.wavelength,
        zenith,
        depol=average(ceilo.depol),
    )
    return Ceilo(raw, beta_raw, beta, ceilo.calibration_factor)
