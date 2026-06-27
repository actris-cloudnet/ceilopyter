"""Fit a temperature-dependent overlap model from collected samples.

Stage 2 of the Hervo et al. (2016) method: given clear-sky overlap samples
(``OverlapSample``, produced by ``overlap/collect.py``), model the per-range-gate
deviation of the overlap from the reference as a linear function of the internal
temperature.

The fit target is chosen to be self-consistent with the apply step
(``overlap/apply.py``). The corrected overlap there is

    overlap(r, T) = reference_overlap(r) * (1 + (alpha(r)*T_C + intercept(r)) / 100),

so for an observed candidate overlap ``o_obs`` at temperature ``T`` the regressed
quantity is the relative deviation in percent::

    Y(r) = (o_obs(r) - reference_overlap(r)) / reference_overlap(r) * 100
         = alpha(r) * T_C + intercept(r).

Per range gate we therefore regress ``Y`` against ``T_C`` (degrees Celsius).
Daily medians are used (rather than raw samples) to avoid over-weighting days
with many accepted windows, following the paper.
"""

import datetime
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy import ma

from ceilopyter.overlap.model import OverlapModel, OverlapSample


@dataclass
class FitConfig:
    """Configuration for ``fit_temperature_model``.

    Attributes:
        min_samples_per_day: Minimum accepted samples for a day to contribute a
            daily-median point.
        min_days: Minimum number of qualifying days required to fit.
        min_temperature_span: Minimum spread (deg C) of daily temperatures
            required to fit a slope.
        temp_window: Optional ``(min, max)`` internal-temperature window (deg C)
            to restrict the fit to; ``None`` uses the full range of qualifying
            days.
    """

    min_samples_per_day: int = 15
    min_days: int = 10
    min_temperature_span: float = 5.0
    temp_window: tuple[float, float] | None = None


def fit_temperature_model(
    samples: list[OverlapSample],
    reference_overlap: npt.NDArray[np.floating],
    range: npt.NDArray[np.floating],
    optical_module_id: str,
    wavelength: float,
    *,
    config: FitConfig | None = None,
) -> OverlapModel:
    """Fit a temperature-dependent overlap model.

    Args:
        samples: Clear-sky overlap samples from one optical module.
        reference_overlap: Reference overlap on ``range``.
        range: Model range grid (m).
        optical_module_id: Optical module serial the samples belong to.
        wavelength: Instrument wavelength (nm).
        config: Fit configuration (defaults applied if omitted).

    Returns:
        The fitted overlap model.

    Raises:
        ValueError: If inputs are inconsistent or too few qualifying days remain.
    """
    config = config or FitConfig()
    reference_overlap = np.asarray(reference_overlap, dtype=float)
    range = np.asarray(range, dtype=float)
    if reference_overlap.shape != range.shape:
        raise ValueError("reference_overlap and range must have the same length")
    if not samples:
        raise ValueError("No samples given")

    temps_c, overlaps = _daily_medians(samples, len(range), config.min_samples_per_day)
    if len(temps_c) < config.min_days:
        raise ValueError(
            f"Only {len(temps_c)} qualifying day(s); need {config.min_days}"
        )

    in_window = _temperature_window(temps_c, config.temp_window)
    temps_c = temps_c[in_window]
    overlaps = overlaps[in_window]
    if ma.ptp(temps_c) < config.min_temperature_span:
        raise ValueError("Daily temperatures span too narrow a range to fit")

    # Relative deviation in percent, gates with ~zero reference are undefined.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = (overlaps - reference_overlap) / reference_overlap * 100.0
    rel_diff = ma.masked_invalid(rel_diff)
    rel_diff[:, reference_overlap <= 0] = ma.masked

    alpha, intercept = _fit_per_gate(temps_c, rel_diff, config.min_temperature_span)

    times = [s.time for s in samples]
    attrs = {
        "training_start": min(times).isoformat(),
        "training_end": max(times).isoformat(),
        "n_samples": str(len(samples)),
        "n_days": str(len(temps_c)),
        "source": "ceilopyter overlap.fit",
    }
    return OverlapModel(
        range=range,
        alpha=alpha,
        intercept=intercept,
        reference_overlap=reference_overlap,
        optical_module_id=optical_module_id,
        temp_valid_min=float(ma.min(temps_c)),
        temp_valid_max=float(ma.max(temps_c)),
        wavelength=wavelength,
        attrs=attrs,
    )


def _daily_medians(
    samples: list[OverlapSample], n_range: int, min_samples_per_day: int
) -> tuple[npt.NDArray[np.floating], ma.MaskedArray]:
    by_day: dict[datetime.date, list[OverlapSample]] = defaultdict(list)
    for s in samples:
        by_day[s.time.date()].append(s)

    temps_c = []
    overlaps = []
    for day in sorted(by_day):
        day_samples = by_day[day]
        if len(day_samples) < min_samples_per_day:
            continue
        stack = ma.masked_invalid([s.overlap for s in day_samples])
        if stack.shape[1] != n_range:
            raise ValueError("Sample overlap length does not match range")
        overlaps.append(ma.median(stack, axis=0))
        temp_k = np.median([s.internal_temperature for s in day_samples])
        temps_c.append(temp_k - 273.15)

    if not temps_c:
        return np.array([]), ma.masked_all((0, n_range))
    return np.array(temps_c), ma.masked_invalid(ma.stack(overlaps))


def _temperature_window(
    temps_c: npt.NDArray[np.floating], window: tuple[float, float] | None
) -> npt.NDArray[np.bool_]:
    if window is None:
        return np.ones(temps_c.shape, dtype=bool)
    lo, hi = window
    return (temps_c >= lo) & (temps_c <= hi)


def _fit_per_gate(
    temps_c: npt.NDArray[np.floating],
    rel_diff: ma.MaskedArray,
    min_temperature_span: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    n_range = rel_diff.shape[1]
    alpha = np.full(n_range, np.nan)
    intercept = np.full(n_range, np.nan)
    for r in range(n_range):
        y = rel_diff[:, r]
        valid = ~ma.getmaskarray(y)
        x = temps_c[valid]
        if x.size < 2 or np.ptp(x) < min_temperature_span:
            continue
        slope, offset = np.polyfit(x, np.asarray(y[valid]), 1)
        alpha[r] = slope
        intercept[r] = offset
    return alpha, intercept
