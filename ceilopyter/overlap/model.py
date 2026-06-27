"""Temperature-dependent CHM15k overlap correction model.

Implements the data container and I/O for the empirical overlap correction of
Hervo, Poltera & Haefele (2016), https://amt.copernicus.org/articles/9/2947/2016/.

The model describes, per range gate, how the overlap function deviates from a
fixed reference overlap as a linear function of the instrument internal
temperature::

    overlap(r, T) = reference_overlap(r) * (1 + (alpha(r) * T_C + intercept(r)) / 100)

where ``T_C`` is the internal temperature in degrees Celsius, ``alpha`` is in
percent per degree Celsius and ``intercept`` is in percent. See
``ceilopyter/overlap/apply.py`` for how this is applied to backscatter.
"""

import datetime
from dataclasses import dataclass, field
from os import PathLike

import netCDF4
import numpy as np
import numpy.typing as npt

_PathType = str | PathLike

# Global attributes stored verbatim as model provenance (string-valued).
_META_ATTRS = (
    "training_start",
    "training_end",
    "n_samples",
    "n_days",
    "source",
    "created",
    "ceilopyter_version",
)


@dataclass
class ReferenceOverlap:
    """A device reference overlap function read from a Lufft/TUB ``.cfg`` file.

    Attributes:
        overlap: Overlap values on the instrument native range grid (rising from
            ~0 near the ground to ~1 once full overlap is reached).
        serial: Optical module serial (``serlom``), e.g. ``"TUB120011"``.
        metadata: Remaining ``key: value`` lines from the file, verbatim.
    """

    overlap: npt.NDArray[np.floating]
    serial: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class OverlapSample:
    """One accepted overlap sample from a clear-sky fitting window.

    Produced by ``overlap/collect.py`` and consumed by ``overlap/fit.py``.

    Attributes:
        time: Window mid-time.
        range_start: Lower range bound of the fitting window (m).
        range_end: Upper range bound of the fitting window (m).
        internal_temperature: Mean internal temperature in the window (K).
        range_resolution: Range resolution of the source data (m).
        overlap: Candidate overlap function on the model range grid.
    """

    time: datetime.datetime
    range_start: float
    range_end: float
    internal_temperature: float
    range_resolution: float
    overlap: npt.NDArray[np.floating]


@dataclass
class OverlapModel:
    """Temperature-dependent overlap correction model (Hervo et al. 2016).

    Attributes:
        range: Model range grid (m).
        alpha: Temperature sensitivity per range gate (percent per degree C).
        intercept: Temperature-independent offset per range gate (percent).
        reference_overlap: Reference overlap on ``range``; assumed equal to the
            factory overlap baked into the CHM15k ``beta_raw``.
        optical_module_id: Optical module serial the model was trained on.
        temp_valid_min: Lower bound of the internal-temperature range (deg C)
            over which the model was fitted.
        temp_valid_max: Upper bound of the internal-temperature range (deg C).
        wavelength: Instrument wavelength (nm); used to sanity-check the data.
        attrs: Free-form provenance metadata (training period, sample count...).
    """

    range: npt.NDArray[np.floating]
    alpha: npt.NDArray[np.floating]
    intercept: npt.NDArray[np.floating]
    reference_overlap: npt.NDArray[np.floating]
    optical_module_id: str
    temp_valid_min: float
    temp_valid_max: float
    wavelength: float
    attrs: dict[str, str] = field(default_factory=dict)


def read_reference_overlap(file: _PathType) -> ReferenceOverlap:
    """Read a Lufft/TUB reference overlap ``.cfg`` file.

    The file starts with an ``ovl`` header line, followed by a single
    whitespace-separated row of overlap values on the instrument native range
    grid, followed by ``key: value`` metadata lines.

    Args:
        file: Path to the ``.cfg`` file.

    Returns:
        The parsed reference overlap.
    """
    with open(file) as f:
        lines = f.read().splitlines()
    if not lines or lines[0].strip().lower() != "ovl":
        raise ValueError("Not a reference overlap file: missing 'ovl' header")
    overlap = np.array(lines[1].split(), dtype=float)
    metadata = {}
    for line in lines[2:]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()
    serial = metadata.get("serlom", "")
    return ReferenceOverlap(overlap=overlap, serial=serial, metadata=metadata)


def read_overlap_model(file: _PathType) -> OverlapModel:
    """Read an overlap correction model from a netCDF file.

    Args:
        file: Path to a model netCDF written by ``write_overlap_model``.

    Returns:
        The overlap model.
    """
    with netCDF4.Dataset(file) as nc:
        for var in ("range", "alpha", "intercept", "reference_overlap"):
            if var not in nc.variables:
                raise ValueError(f"Overlap model missing variable: {var}")
        attrs = {a: str(getattr(nc, a)) for a in _META_ATTRS if hasattr(nc, a)}
        return OverlapModel(
            range=nc["range"][:],
            alpha=nc["alpha"][:],
            intercept=nc["intercept"][:],
            reference_overlap=nc["reference_overlap"][:],
            optical_module_id=str(getattr(nc, "optical_module_id", "")),
            temp_valid_min=float(getattr(nc, "temp_valid_min", -np.inf)),
            temp_valid_max=float(getattr(nc, "temp_valid_max", np.inf)),
            wavelength=float(nc.wavelength),
            attrs=attrs,
        )


def write_overlap_model(model: OverlapModel, file: _PathType) -> None:
    """Write an overlap correction model to a netCDF file.

    Args:
        model: The model to serialize.
        file: Output path.
    """
    with netCDF4.Dataset(file, "w") as nc:
        nc.createDimension("range", len(model.range))
        for name, values in (
            ("range", model.range),
            ("alpha", model.alpha),
            ("intercept", model.intercept),
            ("reference_overlap", model.reference_overlap),
        ):
            nc.createVariable(name, "f4", ("range",))[:] = values
        nc["range"].units = "m"
        nc["alpha"].units = "percent per degree_C"
        nc["intercept"].units = "percent"
        nc["reference_overlap"].units = "1"
        nc.optical_module_id = model.optical_module_id
        nc.temp_valid_min = model.temp_valid_min
        nc.temp_valid_max = model.temp_valid_max
        nc.wavelength = model.wavelength
        for key, value in model.attrs.items():
            setattr(nc, key, value)
