"""Reader for Cloudnet harmonized lidar product files.

These daily netCDF files already contain calibrated and noise-screened `beta`
(and `depolarisation` for the CL61), so reading one is much faster than reading
and processing thousands of raw files — useful for iterating on the classifier.
"""

import logging
from os import PathLike

import netCDF4
import numpy as np
import numpy.typing as npt
from cftime import num2pydate
from numpy import ma

from .ceilo import Ceilo
from .ceilo_raw import CeiloRaw
from .noise import NOISE_FLOORS, screen_noise

# Substrings of the product `source` attribute mapped to a ceilopyter
# instrument. Ordered so more specific names (CHM15kx) are matched first.
_SOURCE_TO_INSTRUMENT: list[tuple[str, str]] = [
    ("chm15kx", "chm15k"),
    ("chm15k", "chm15k"),
    ("cl61", "cl61"),
    ("cl51", "cl51"),
    ("cl31", "cl31"),
    ("cs135", "cs135"),
    ("ct25k", "ct25k"),
]


def read_lidar(
    files: str | PathLike | list[str | PathLike],
    calibration_factor: float | None = None,
    *,
    rescreen: bool = True,
) -> Ceilo:
    """Read one or more Cloudnet harmonized lidar product files.

    Args:
        files: Lidar product netCDF file(s) (must share the same range grid).
        calibration_factor: Ignored — the product is already calibrated (a
            warning is logged if a value is given).
        rescreen: Re-screen the product `beta_raw` with ceilopyter's
            `screen_noise` instead of using the product's `beta`. Cloudnet's
            screening is less aggressive, so this keeps the classification
            consistent with reading raw files. Needs a recognizable instrument
            in the `source` attribute; otherwise the product `beta` is used.

    Returns:
        A `Ceilo` for classification.
    """
    if calibration_factor is not None:
        logging.warning("Ignoring calibration factor: lidar product is calibrated")
    if not isinstance(files, list):
        files = [files]
    if not files:
        raise ValueError("No lidar files given")

    times, betas, betas_raw, depols, zeniths = [], [], [], [], []
    rng: npt.NDArray | None = None
    wavelength = 0.0
    calibration = 1.0
    source = ""
    for file in files:
        with netCDF4.Dataset(file) as nc:
            r = nc["range"][:]
            if rng is None:
                rng = r
            elif not np.array_equal(rng, r):
                raise ValueError("Inconsistent ranges between lidar files")
            times.append(num2pydate(nc["time"][:], nc["time"].units))
            betas.append(nc["beta"][:])
            betas_raw.append(nc["beta_raw"][:])
            wavelength = float(nc["wavelength"][:])
            if "calibration_factor" in nc.variables:
                # Some products (e.g. PollyXT) store a per-time array rather than
                # a scalar; collapse to one representative value. It is only kept
                # as metadata -- the product beta is already calibrated.
                calibration = float(np.asarray(nc["calibration_factor"][:]).mean())
            source = getattr(nc, "source", "")
            zeniths.append(_read_zenith(nc, len(times[-1])))
            depols.append(
                nc["depolarisation"][:] if "depolarisation" in nc.variables else None
            )

    assert rng is not None  # files is non-empty, so the loop set rng
    time = np.concatenate(times)
    beta = ma.concatenate(betas)
    beta_raw = ma.concatenate(betas_raw)
    if rescreen:
        instrument = _detect_instrument(source)
        if instrument is None:
            logging.warning(
                "Unrecognized lidar source %r; using the product's screening", source
            )
        else:
            screened = screen_noise(beta_raw, rng, noise_floor=NOISE_FLOORS[instrument])
            beta = ma.asarray(screened)
    # Keep depolarization wherever it exists; only drop it entirely if no file
    # has it. Files that lack it contribute masked spans rather than discarding
    # depol for the whole (possibly multi-day) concatenation.
    n_missing = sum(d is None for d in depols)
    depol: ma.MaskedArray | None
    if n_missing == len(depols):
        depol = None
    else:
        if n_missing:
            logging.warning(
                "%d of %d lidar file(s) lack depolarization; masking those spans",
                n_missing,
                len(depols),
            )
        depol = ma.concatenate(
            [
                ma.masked_all((len(t), len(rng))) if d is None else ma.asarray(d)
                for d, t in zip(depols, times, strict=True)
            ]
        )
    zenith = (
        None
        if any(z is None for z in zeniths)
        else np.concatenate([z for z in zeniths if z is not None])
    )

    raw = CeiloRaw(time, rng, beta_raw, wavelength, zenith, depol=depol)
    return Ceilo(raw, beta_raw, beta, calibration)


def _detect_instrument(source: str) -> str | None:
    return next(
        (inst for key, inst in _SOURCE_TO_INSTRUMENT if key in source.lower()), None
    )


def _read_zenith(nc: netCDF4.Dataset, n_time: int) -> npt.NDArray | None:
    if "zenith_angle" not in nc.variables:
        return None
    zenith = nc["zenith_angle"][:]
    if np.ndim(zenith) == 0:  # stored as a scalar in the product
        return np.full(n_time, float(zenith))
    return zenith
