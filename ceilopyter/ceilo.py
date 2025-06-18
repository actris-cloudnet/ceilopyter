import numpy as np
import numpy.typing as npt

from .ceilo_raw import CeiloRaw


class Ceilo:
    """Raw ceilometer data.

    Attributes:
        time: Time
        range: Range (m)
        beta_raw: Non-screened range-corrected backscatter coefficient (sr-1 m-1)
        beta: Screened range-corrected backscatter coefficient (sr-1 m-1)
        wavelength: Wavelength (nm)
        zenith_angle: Zenith angle (deg)
    """

    def __init__(
        self,
        raw: CeiloRaw,
        beta_raw: npt.NDArray[np.floating],
        beta: npt.NDArray[np.floating] | None,
        calibration_factor: float,
    ):
        self.time = raw.time
        self.range = raw.range
        self.beta = beta
        self.beta_raw = beta_raw
        self.calibration_factor = calibration_factor
        self.wavelength = raw.wavelength
        self.zenith_angle = raw.zenith_angle
