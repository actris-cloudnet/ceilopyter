"""Apply the temperature-dependent overlap correction to CHM15k backscatter.

The CHM15k ``beta_raw`` already has the factory overlap ``o_factory(r)`` divided
out by the firmware. The model describes the *correct* overlap as

    overlap(r, T) = reference_overlap(r) * factor(r, T),
    factor(r, T) = 1 + (alpha(r) * T_C + intercept(r)) / 100.

Assuming ``reference_overlap == o_factory`` (the supplied TUB reference is the
factory overlap, "Case A" in the plan), re-dividing by the correct overlap
collapses to dividing the calibrated backscatter by ``factor`` alone::

    beta_corrected(r, t) = beta_raw(r, t) / factor(r, T_C(t)).

No overlap array is therefore needed at apply time.
"""

import logging

import numpy as np
import numpy.typing as npt
from numpy import ma

from ..ceilo import Ceilo
from ..ceilo_raw import CeiloRaw
from ..noise import NOISE_FLOORS, screen_noise
from .model import OverlapModel

# Below this the correction factor is treated as non-physical (a correction
# more negative than -90 %) and the gate is left uncorrected.
_MIN_FACTOR = 0.1


def correct_overlap(
    ceilo: Ceilo,
    model: OverlapModel,
    *,
    min_range: float = 0.0,
    clip_temp: bool = True,
) -> Ceilo:
    """Apply the temperature-dependent overlap correction to a CHM15k ``Ceilo``.

    Args:
        ceilo: CHM15k data with ``internal_temperature`` set (read by
            ``read_chm15k``).
        model: A trained overlap model for this optical module.
        min_range: Leave gates below this range (m) uncorrected. The default
            (0) relies on the model coefficients being zero outside its fit
            region.
        clip_temp: Clamp the internal temperature to the model's valid window
            instead of extrapolating outside it (which would only log a
            warning).

    Returns:
        A new ``Ceilo`` with corrected ``beta_raw`` and re-screened ``beta``.
    """
    if ceilo.internal_temperature is None:
        raise ValueError("Overlap correction needs internal_temperature (temp_int)")
    if not np.isclose(np.asarray(ceilo.wavelength, dtype=float), model.wavelength):
        logging.warning(
            "Overlap model wavelength (%s nm) differs from data (%s nm)",
            model.wavelength,
            ceilo.wavelength,
        )
    beta_raw = _apply_overlap(
        ceilo.beta_raw,
        ceilo.range,
        ceilo.internal_temperature,
        model,
        min_range=min_range,
        clip_temp=clip_temp,
    )
    logging.info(
        "Applied overlap correction (model %s)", model.optical_module_id or "unknown"
    )
    beta = screen_noise(beta_raw, ceilo.range, noise_floor=NOISE_FLOORS["chm15k"])
    raw = CeiloRaw(
        ceilo.time,
        ceilo.range,
        beta_raw,
        ceilo.wavelength,
        ceilo.zenith_angle,
        depol=ceilo.depol,
        internal_temperature=ceilo.internal_temperature,
    )
    return Ceilo(raw, beta_raw, beta, ceilo.calibration_factor)


def _apply_overlap(
    beta_raw: npt.NDArray[np.floating],
    range: npt.NDArray[np.floating],
    internal_temperature: npt.NDArray[np.floating],
    model: OverlapModel,
    *,
    min_range: float,
    clip_temp: bool,
) -> npt.NDArray[np.floating]:
    # Interpolate model coefficients onto the data range grid; identity (0)
    # outside the model's range or where the model did not fit (NaN).
    alpha = np.interp(
        range, model.range, np.nan_to_num(model.alpha), left=0.0, right=0.0
    )
    intercept = np.interp(
        range, model.range, np.nan_to_num(model.intercept), left=0.0, right=0.0
    )

    t_c = ma.asarray(internal_temperature, dtype=float) - 273.15
    if clip_temp:
        t_c = ma.clip(t_c, model.temp_valid_min, model.temp_valid_max)
    else:
        outside = (t_c < model.temp_valid_min) | (t_c > model.temp_valid_max)
        if ma.any(outside):
            logging.warning(
                "%d profile(s) have internal temperature outside the model's "
                "valid window [%s, %s] degC; correction is extrapolated",
                int(ma.sum(outside)),
                model.temp_valid_min,
                model.temp_valid_max,
            )

    factor = 1.0 + (alpha[np.newaxis, :] * t_c[:, np.newaxis] + intercept) / 100.0
    # Identity where the temperature is masked, the factor is non-physical, or
    # the gate is below min_range.
    factor = ma.filled(factor, 1.0)
    factor[factor < _MIN_FACTOR] = 1.0
    factor[:, range < min_range] = 1.0
    return beta_raw / factor
