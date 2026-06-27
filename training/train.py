r"""Command-line training of a CHM15k overlap model.

Ties the offline stages together: read a reference overlap, collect clear-sky
samples from raw files, fit the temperature model, and write it to netCDF::

    python -m training.train \\
        --reference TUB120011_20121112_1024.cfg \\
        --wavelength 1064 \\
        --output model.nc \\
        day1.nc day2.nc ...

The range grid is read from ``--range-from`` (defaults to the first input file),
so the reference overlap and the data share one grid.
"""

import argparse
import logging
import sys
from pathlib import Path

import netCDF4
import numpy as np

from ceilopyter.overlap.model import read_reference_overlap, write_overlap_model

from .collect import CollectConfig, collect_overlap_samples
from .fit import FitConfig, fit_temperature_model


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CHM15k overlap model.")
    parser.add_argument("files", nargs="+", help="Raw CHM15k netCDF files.")
    parser.add_argument(
        "--reference", required=True, help="Reference overlap .cfg file."
    )
    parser.add_argument(
        "--range-from",
        help="CHM15k netCDF file to read the range grid from "
        "(defaults to the first input file).",
    )
    parser.add_argument(
        "--wavelength", type=float, default=1064.0, help="Wavelength (nm)."
    )
    parser.add_argument("--output", required=True, help="Output model netCDF path.")
    return parser.parse_args(argv)


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

    samples = collect_overlap_samples(
        args.files, reference.overlap, rng, config=CollectConfig()
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
        config=FitConfig(),
    )
    write_overlap_model(model, args.output)
    logging.info("Wrote overlap model to %s", Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
