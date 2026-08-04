import argparse
import datetime
import gzip
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cloudnet_api_client import APIClient
from numpy import ma

from ceilopyter import (
    read_chm15k,
    read_cl31,
    read_cl51,
    read_cl61,
    read_cs135,
    read_ct25k,
    read_ld40,
)


def make_edges(centers):
    edges = np.empty(len(centers) + 1)
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", type=Path, nargs="+", help="path to raw data")
    parser.add_argument("-s", "--site", help="Cloudnet site identifier (e.g. hyytiala)")
    parser.add_argument(
        "-d",
        "--date",
        type=datetime.date.fromisoformat,
        help="measurement date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        choices=[
            "ld40",
            "ct25k",
            "cl31",
            "cl51",
            "cl61d",
            "cs135",
            "chm15k",
            "chm15kx",
        ],
        required=True,
        help="instrument identifier",
    )
    parser.add_argument(
        "--noise-h2",
        choices=["on", "off"],
        default="on",
        help="noise range correction (ct25k, cl31, cl51)",
    )
    parser.add_argument(
        "--calibration-factor", type=float, help="instrument calibration factor"
    )
    args = parser.parse_args()

    if args.files and (args.site or args.date):
        parser.error("cannot use --files with --site and --date")
    elif args.files:
        file_paths = args.files
    elif args.site and args.date:
        client = APIClient()
        raw_dir = "data_raw/"
        raw_metadata = client.raw_files(
            site_id=args.site,
            date=args.date,
            instrument_id=args.instrument,
            status=["uploaded", "processed"],
        )
        instrument_pids = sorted({r.instrument.pid for r in raw_metadata})
        if len(instrument_pids) > 1:
            logging.warning("Multiple instruments found, using %s", instrument_pids[0])
            raw_metadata = [
                r for r in raw_metadata if r.instrument.pid == instrument_pids[0]
            ]
        file_paths = client.download(raw_metadata, raw_dir)
    else:
        parser.error("use --files or --site and --date")

    file_paths2 = []
    for path_in in file_paths:
        if path_in.suffix.lower() == ".gz":
            path_out = path_in.parent / path_in.stem
            print(f"Decompressing {path_in} to {path_out}")
            with gzip.open(path_in, "rb") as file_in, open(path_out, "wb") as file_out:
                shutil.copyfileobj(file_in, file_out)
            file_paths2.append(path_out)
        else:
            file_paths2.append(path_in)

    if args.instrument == "ct25k":
        ceilo = read_ct25k(file_paths2, args.calibration_factor, args.noise_h2 == "on")
    elif args.instrument == "cl31":
        ceilo = read_cl31(file_paths2, args.calibration_factor)
    elif args.instrument == "cl51":
        ceilo = read_cl51(file_paths2, args.calibration_factor)
    elif args.instrument in ("chm15k", "chm15kx"):
        ceilo = read_chm15k(file_paths2, args.calibration_factor)
    elif args.instrument == "cl61d":
        ceilo = read_cl61(file_paths2, args.calibration_factor)
    elif args.instrument == "cs135":
        ceilo = read_cs135(file_paths2, args.calibration_factor)
    elif args.instrument == "ld40":
        ceilo = read_ld40(file_paths2, args.calibration_factor)
    else:
        raise NotImplementedError

    vmin = 1e-7
    vmax = 1e-4
    plt_beta_raw = ma.log10(np.maximum(vmin, ceilo.beta_raw))
    plt_beta = (
        ma.log10(np.maximum(vmin, ceilo.beta)) if ceilo.beta is not None else None
    )
    plt_time = make_edges(np.arange(len(ceilo.time)))
    range_km = ceilo.range / 1000
    plt_rng = make_edges(range_km)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(15, 5))
    ax1.set_title("beta_raw")
    ax1.set_ylabel("Range (km)")
    ax1.pcolorfast(
        plt_time, plt_rng, plt_beta_raw.T, vmin=np.log10(vmin), vmax=np.log10(vmax)
    )
    ax2.set_title("beta")
    ax2.set_ylabel("Range (km)")
    ax2.set_xlabel("Time index")
    if plt_beta is not None:
        ax2.pcolorfast(
            plt_time, plt_rng, plt_beta.T, vmin=np.log10(vmin), vmax=np.log10(vmax)
        )
    plt.tight_layout()

    def onclick(event):
        if fig.canvas.manager.toolbar.mode:
            return
        i = round(event.xdata)

        fig2, axs = plt.subplots(1, 2, sharey=True)
        axs[0].set_title(f"beta_raw i={i}")
        axs[0].plot(ceilo.beta_raw[i], range_km)
        axs[0].set_ylabel("Range (km)")
        axs[1].set_title(f"beta_uncorr i={i}")
        axs[1].plot(ceilo.beta_raw[i] / range_km**2, range_km)
        plt.show()

    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()


if __name__ == "__main__":
    main()
