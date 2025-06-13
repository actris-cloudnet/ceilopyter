import datetime
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os import PathLike

import numpy as np
import numpy.typing as npt

from ceilopyter.ceilo_raw import CeiloRaw


class InvalidMessageError(Exception):
    pass


@dataclass
class Message:
    """Data message from ceilometer.

    Attributes:
        range_resolution: Range resolution (m).
        laser_pulse_energy: Laser pulse energy (%).
        laser_temperature: Laser temperature (degC).
        tilt_angle: Tilt angle (deg).
        background_light: Background light (mV).
        n_pulses: Number of pulses.
        sample_rate: Sampling rate (MHz).
        beta: Backscatter coefficient (sr-1 m-1).
    """

    range_resolution: int
    laser_pulse_energy: int
    laser_temperature: int
    tilt_angle: int
    background_light: int
    n_pulses: int
    sample_rate: int
    beta: npt.NDArray[np.floating]


class Status:
    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(f"{key}=True" for key, value in vars(self).items() if value)
            + ")"
        )


def read_msgs(
    files: str | PathLike | list[str | PathLike],
    read_file: Callable[
        [str | PathLike], tuple[Sequence[datetime.datetime], Sequence[Message]]
    ],
    wavelength: float,
) -> list[CeiloRaw]:
    """Read ceilometer messages into CeiloRaw objects."""
    if not isinstance(files, list):
        files = [files]
    raw = []
    for file in files:
        time, msgs = read_file(file)

        groups: dict[
            tuple[int, int], tuple[list[datetime.datetime], list[Message]]
        ] = {}
        for t, msg in zip(time, msgs, strict=True):
            n_gates = len(msg.beta)
            key = (msg.range_resolution, n_gates)
            if key not in groups:
                groups[key] = ([], [])
            groups[key][0].append(t)
            groups[key][1].append(msg)

        for (res, n_gates), (grp_time, grp_msgs) in groups.items():
            time_arr = np.array(grp_time)
            rng = np.arange(n_gates) * res + res / 2
            beta = np.array([msg.beta for msg in grp_msgs])
            tilt_angle = np.array([msg.tilt_angle for msg in grp_msgs])
            ceilo = CeiloRaw(time_arr, rng, beta, wavelength, tilt_angle)
            raw.append(ceilo)

    return raw
