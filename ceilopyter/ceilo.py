from .ceilo_raw import CeiloRaw, concatenate_raw


class Ceilo:
    def __init__(self, raw: list[CeiloRaw], calibration_factor: float):
        concat = concatenate_raw(raw)
        self.time = concat.time
        self.range = concat.range
        self.beta_raw = concat.beta * calibration_factor
        self.calibration_factor = calibration_factor
        self.wavelength = concat.wavelength
        self.zenith_angle = concat.zenith_angle
