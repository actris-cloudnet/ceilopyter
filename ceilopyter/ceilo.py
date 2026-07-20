import numpy as np
import numpy.typing as npt

from .ceilo_raw import CeiloRaw


class Ceilo:
    """Ceilometer data.

    Attributes:
        time: Time
        range: Range (m)
        beta_raw: Non-screened range-corrected backscatter coefficient (sr-1 m-1)
        beta: Screened range-corrected backscatter coefficient (sr-1 m-1)
        beta_smooth: Screened range-corrected smoothed backscatter coefficient
            (sr-1 m-1)
        wavelength: Wavelength (nm)
        zenith_angle: Zenith angle (deg)
        serial_number: Instrument serial number
    """

    def __init__(
        self,
        raw: CeiloRaw,
        beta_raw: npt.NDArray[np.floating],
        calibration_factor: float,
        beta: npt.NDArray[np.floating] | None = None,
        beta_smooth: npt.NDArray[np.floating] | None = None,
    ):
        self.time = raw.time
        self.range = raw.range
        self.beta = beta
        self.beta_raw = beta_raw
        self.beta_smooth = beta_smooth
        self.calibration_factor = calibration_factor
        self.wavelength = raw.wavelength
        self.zenith_angle = raw.zenith_angle
        self.serial_number = raw.serial_number
