"""Collect clear-sky overlap samples from raw CHM15k files.

Stage 1 of the Hervo et al. (2016) method. For each daily file we slide a time
window over the data and, in windows that are clear-sky and temporally stable,
derive a candidate overlap function:

In a cloud- and aerosol-free atmosphere the molecular range-corrected signal is
log-linear in range. A straight line is fitted to the mean ``log10`` signal in
the altitude band where the overlap is already complete (where the reference
overlap reaches 1, up to ``max_fit_range``) and extrapolated downward. The ratio
of the actual signal to the extrapolated molecular line is the deviation of the
current overlap from the reference::

    factor(r)    = 10 ** (mean_log_signal(r) - molecular_line(r))
    candidate(r) = reference_overlap(r) * factor(r).

Because the CHM15k ``beta_raw`` is the signal divided by the (reference) factory
overlap, ``candidate`` is the actual overlap at the window's temperature. Each
accepted window yields one ``OverlapSample`` consumed by ``overlap/fit.py``.

This module reads raw netCDF directly and lives in the offline ``training``
package (not shipped); use ``from training.collect import collect_overlap_samples``.
"""

import datetime
import logging
from dataclasses import dataclass
from os import PathLike

import netCDF4
import numpy as np
import numpy.typing as npt
from cftime import num2pydate
from numpy import ma

from ceilopyter.overlap.model import OverlapSample

_PathType = str | PathLike


@dataclass
class CollectConfig:
    """Configuration for ``collect_overlap_samples``.

    Attributes:
        window_minutes: Length of each fitting window.
        step_minutes: Step between consecutive windows.
        min_profiles: Minimum profiles required in a window.
        max_fit_range: Upper range (m) of the molecular fit band.
        min_fit_length: Minimum length (m) of the molecular fit band.
        overlap_full_threshold: Reference overlap value at/above which the
            overlap is considered complete; the fit band starts here.
        max_std_over_mean: Maximum temporal std/mean of the signal in the fit
            band for the window to count as stable.
        min_cloud_base: Cloud base must be absent or above this range (m).
        clear_sky_value: Value of ``sci`` that means clear sky.
        max_overlap_rel_error: Maximum relative error between candidate and
            reference overlap within the fit band.
        max_overlap_value: Maximum allowed candidate overlap value.
    """

    window_minutes: float = 30.0
    step_minutes: float = 5.0
    min_profiles: int = 10
    max_fit_range: float = 1200.0
    min_fit_length: float = 150.0
    overlap_full_threshold: float = 0.99
    max_std_over_mean: float = 0.015
    min_cloud_base: float = 1200.0
    clear_sky_value: int = 0
    max_overlap_rel_error: float = 0.01
    max_overlap_value: float = 1.01


@dataclass
class _FileData:
    time: npt.NDArray[np.object_]
    range: npt.NDArray[np.floating]
    signal: ma.MaskedArray  # range-corrected backscatter, (time, range)
    temperature: npt.NDArray[np.floating]  # internal temperature (K), (time,)
    sci: npt.NDArray[np.floating]  # sky condition index, (time,)
    cloud_base: npt.NDArray[np.floating]  # lowest cloud base (m), inf if none
    range_resolution: float


def collect_overlap_samples(
    files: _PathType | list[_PathType],
    reference_overlap: npt.NDArray[np.floating],
    range: npt.NDArray[np.floating],
    *,
    config: CollectConfig | None = None,
) -> list[OverlapSample]:
    """Collect clear-sky overlap samples from one or more raw CHM15k files.

    Args:
        files: CHM15k netCDF file path(s) from a single optical module.
        reference_overlap: Reference overlap on ``range``.
        range: Range grid (m); must match the files' range grid.
        config: Collection configuration (defaults applied if omitted).

    Returns:
        The accepted overlap samples (possibly empty).
    """
    config = config or CollectConfig()
    reference_overlap = np.asarray(reference_overlap, dtype=float)
    range = np.asarray(range, dtype=float)
    if reference_overlap.shape != range.shape:
        raise ValueError("reference_overlap and range must have the same length")
    if not isinstance(files, list):
        files = [files]

    fit_band = _fit_band(reference_overlap, range, config)

    samples: list[OverlapSample] = []
    for file in files:
        try:
            data = _read_file(file, range)
        except (OSError, ValueError) as err:
            logging.warning("Skipping %s: %s", file, err)
            continue
        samples.extend(_collect_from_file(data, reference_overlap, fit_band, config))
    return samples


def _fit_band(
    reference_overlap: npt.NDArray[np.floating],
    range: npt.NDArray[np.floating],
    config: CollectConfig,
) -> npt.NDArray[np.bool_]:
    full = np.flatnonzero(reference_overlap >= config.overlap_full_threshold)
    if full.size == 0:
        raise ValueError("Reference overlap never reaches the full-overlap threshold")
    band = (range >= range[full[0]]) & (range <= config.max_fit_range)
    if np.count_nonzero(band) < 2 or np.ptp(range[band]) < config.min_fit_length:
        raise ValueError("Molecular fit band is too short; check max_fit_range")
    return band


def _collect_from_file(
    data: _FileData,
    reference_overlap: npt.NDArray[np.floating],
    fit_band: npt.NDArray[np.bool_],
    config: CollectConfig,
) -> list[OverlapSample]:
    t0 = data.time[0]
    seconds = np.array([(t - t0).total_seconds() for t in data.time])
    window = config.window_minutes * 60.0
    step = config.step_minutes * 60.0
    near = data.range <= config.max_fit_range

    samples = []
    start = 0.0
    while start + window <= seconds[-1] + step:
        in_window = (seconds >= start) & (seconds < start + window)
        start += step
        if np.count_nonzero(in_window) < config.min_profiles:
            continue
        if not _is_clear_and_stable(data, in_window, fit_band, config):
            continue
        candidate = _candidate_overlap(
            data.signal[in_window], data.range, reference_overlap, fit_band, near
        )
        if candidate is None or not _is_valid(
            candidate, reference_overlap, fit_band, config
        ):
            continue
        mid = t0 + datetime.timedelta(seconds=start - step + window / 2)
        samples.append(
            OverlapSample(
                time=mid,
                range_start=float(data.range[fit_band][0]),
                range_end=float(data.range[fit_band][-1]),
                internal_temperature=float(np.mean(data.temperature[in_window])),
                range_resolution=data.range_resolution,
                overlap=candidate,
            )
        )
    return samples


def _is_clear_and_stable(
    data: _FileData,
    in_window: npt.NDArray[np.bool_],
    fit_band: npt.NDArray[np.bool_],
    config: CollectConfig,
) -> bool:
    if np.any(data.sci[in_window] != config.clear_sky_value):
        return False
    if np.any(data.cloud_base[in_window] < config.min_cloud_base):
        return False
    band_signal = data.signal[in_window][:, fit_band]
    mean = ma.mean(band_signal, axis=0)
    std = ma.std(band_signal, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        std_over_mean = ma.abs(std / mean)
    return bool(ma.max(std_over_mean) <= config.max_std_over_mean)


def _candidate_overlap(
    signal_window: ma.MaskedArray,
    range: npt.NDArray[np.floating],
    reference_overlap: npt.NDArray[np.floating],
    fit_band: npt.NDArray[np.bool_],
    near: npt.NDArray[np.bool_],
) -> npt.NDArray[np.floating] | None:
    mean_log = ma.mean(ma.log10(ma.masked_less_equal(signal_window, 0.0)), axis=0)
    if ma.any(ma.getmaskarray(mean_log)[fit_band]):
        return None  # noise in the fit band; cannot fit the molecular line
    slope, intercept = np.polyfit(range[fit_band], np.asarray(mean_log[fit_band]), 1)
    line = slope * range + intercept
    factor = 10.0 ** (mean_log - line)
    candidate = reference_overlap.copy()
    candidate[near] = reference_overlap[near] * ma.filled(factor[near], np.nan)
    return candidate


def _is_valid(
    candidate: npt.NDArray[np.floating],
    reference_overlap: npt.NDArray[np.floating],
    fit_band: npt.NDArray[np.bool_],
    config: CollectConfig,
) -> bool:
    if np.nanmax(candidate) > config.max_overlap_value:
        return False
    if np.any(~np.isfinite(candidate[fit_band])):
        return False
    rel_error = (
        np.abs(candidate[fit_band] - reference_overlap[fit_band])
        / (reference_overlap[fit_band])
    )
    return bool(np.max(rel_error) <= config.max_overlap_rel_error)


def _read_file(file: _PathType, range: npt.NDArray[np.floating]) -> _FileData:
    with netCDF4.Dataset(file) as nc:
        file_range = nc["range"][:]
        if not np.array_equal(np.asarray(file_range), range):
            raise ValueError("File range grid does not match the reference range")
        time = num2pydate(nc["time"][:], nc["time"].units)
        signal = ma.asarray(nc["beta_raw"][:])
        temperature = _read_1d(nc, "temp_int", len(time))
        sci = _read_1d(nc, "sci", len(time), fill=0.0)
        cloud_base = _read_cloud_base(nc, len(time))
        range_resolution = float(np.median(np.diff(range)))
    return _FileData(
        time=time,
        range=range,
        signal=signal,
        temperature=temperature,
        sci=sci,
        cloud_base=cloud_base,
        range_resolution=range_resolution,
    )


def _read_1d(
    nc: netCDF4.Dataset, name: str, n_time: int, *, fill: float = np.nan
) -> npt.NDArray[np.floating]:
    if name not in nc.variables:
        raise ValueError(f"Missing variable: {name}")
    return ma.filled(ma.asarray(nc[name][:], dtype=float), fill)


def _read_cloud_base(nc: netCDF4.Dataset, n_time: int) -> npt.NDArray[np.floating]:
    name = next((n for n in ("cloud_base_height", "cbh") if n in nc.variables), None)
    if name is None:
        return np.full(n_time, np.inf)  # no cloud information -> assume clear
    cbh = ma.asarray(nc[name][:], dtype=float)
    if cbh.ndim > 1:
        cbh = cbh[:, 0]  # lowest cloud layer
    # Missing / non-positive cloud base means no cloud detected.
    return ma.filled(ma.masked_less_equal(cbh, 0.0), np.inf)
