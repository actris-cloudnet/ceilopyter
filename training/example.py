r"""End-to-end example of the CHM15k overlap workflow.

Ties the whole chain together so you can see it run on real files: collect
clear-sky samples, fit the temperature model, write it, read it back, and apply
it to a CHM15k file::

    python -m training.example \\
        --reference TUB200009_..._1024.cfg \\
        --output model.nc \\
        --apply-to 20230701_pay_CHM200110_000.nc \\
        day1.nc day2.nc ...

The reference overlap must be the device's own factory overlap (matching the
optical module ``serlom``); each model is specific to one optical module.

``--relax`` opens the collection and fit thresholds so the pipeline still
produces output when the reference does *not* match the instrument. This is only
useful as a smoke test of the code path: the resulting model is **not physically
valid** (its coefficients absorb the reference mismatch). For a real model, use a
matching reference and omit ``--relax``.
"""

import argparse
import logging
import sys
from pathlib import Path

import netCDF4
import numpy as np
from numpy import ma

from ceilopyter import correct_overlap, read_chm15k
from ceilopyter.overlap.model import (
    read_overlap_model,
    read_reference_overlap,
    write_overlap_model,
)

from .collect import CollectConfig, collect_overlap_samples
from .fit import FitConfig, fit_temperature_model


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end CHM15k overlap example (train and apply)."
    )
    parser.add_argument("files", nargs="+", help="Raw CHM15k netCDF files.")
    parser.add_argument(
        "--reference", required=True, help="Reference overlap .cfg file."
    )
    parser.add_argument("--output", required=True, help="Output model netCDF path.")
    parser.add_argument(
        "--range-from",
        help="CHM15k netCDF file to read the range grid from "
        "(defaults to the first input file).",
    )
    parser.add_argument(
        "--wavelength", type=float, default=1064.0, help="Wavelength (nm)."
    )
    parser.add_argument(
        "--apply-to",
        help="Optional CHM15k file to apply the trained model to, reporting the "
        "near-range effect.",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Loosen thresholds so a non-matching reference still yields a "
        "(non-physical) model; for demonstrating the code path only.",
    )
    return parser.parse_args(argv)


def _report_application(model_path: str, file: str) -> None:
    """Apply the model to one file and log the near-range change."""
    plain = read_chm15k(file, calibration_factor=1.0)
    model = read_overlap_model(model_path)
    corrected = correct_overlap(plain, model)
    near = corrected.range <= 600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ma.masked_invalid(corrected.beta_raw / plain.beta_raw)
    logging.info(
        "Applied model to %s: mean corrected/uncorrected beta_raw below 600 m = "
        "%.4f (1.0 means no change)",
        Path(file).name,
        float(ma.mean(ratio[:, near])),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)

    reference = read_reference_overlap(args.reference)
    range_from = args.range_from or args.files[0]
    with netCDF4.Dataset(range_from) as nc:
        rng = np.asarray(nc["range"][:], dtype=float)
    if reference.overlap.shape != rng.shape:
        logging.error(
            "Reference overlap (%d) and range grid (%d) lengths differ",
            reference.overlap.size,
            rng.size,
        )
        return 1

    if args.relax:
        # Loosened thresholds so a non-matching reference still yields output
        # (see module docstring); the resulting model is not physical.
        collect_config = CollectConfig(
            max_std_over_mean=0.08, max_overlap_value=1.10, max_overlap_rel_error=0.10
        )
        fit_config = FitConfig(
            min_samples_per_day=2, min_days=4, min_temperature_span=3.0
        )
        logging.warning("Running with relaxed thresholds; model will not be physical")
    else:
        collect_config = CollectConfig()
        fit_config = FitConfig()

    samples = collect_overlap_samples(
        args.files, reference.overlap, rng, config=collect_config
    )
    logging.info("Collected %d overlap sample(s)", len(samples))
    if not samples:
        logging.error("No samples collected; cannot fit a model")
        return 1

    model = fit_temperature_model(
        samples,
        reference.overlap,
        rng,
        reference.serial,
        args.wavelength,
        config=fit_config,
    )
    write_overlap_model(model, args.output)
    logging.info("Wrote overlap model to %s", Path(args.output))

    # Read it back to confirm the round-trip, then optionally apply it.
    read_overlap_model(args.output)
    if args.apply_to:
        _report_application(args.output, args.apply_to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
