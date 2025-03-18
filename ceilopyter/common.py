from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


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
