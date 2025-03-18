import datetime
import logging
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from . import utils
from .common import InvalidMessageError, Message, Status

FORMAT = utils.date_format_to_regex(rb"-%Y-%m-%d %H:%M:%S\r?\n")


class CtStatus(Status):
    """Decoded status bits from Vaisala CT25K.

    Attributes:
        laser_temp_shutoff_alarm: Laser temperature shut-off.
        laser_fail_alarm: Laser failure.
        receiver_fail_alarm: Receiver failure.
        voltage_fail_alarm: Voltage failure.
        window_contam_warning: Window contaminated.
        battery_low_warning: Battery low.
        laser_power_low_warning: Laser power low.
        laser_temp_warning: Laser temperature high or low.
        internal_temp_warning: Internal temperature high or low.
        voltage_range_warning: Voltage high or low.
        humidity_high_warning: Relative humidity is high > 85 %.
        receiver_crosstalk_warning: Receiver optical cross-talk compensation poor.
        blower_fail_warning: Blower failure.
        blower_status: Blower is ON.
        blower_heater_status: Blower heater is ON.
        internal_heater_status: Internal heater is ON.
        units_meters: Units are METERS if ON, else FEET.
        polling_mode_status: Polling mode is ON.
        battery_power_status: Working from battery.
        single_seq_mode_status: Single sequence mode is ON.
        manual_settings_status: Manual settings are effective.
        tilt_angle_warning: Tilt angle is > 45 degrees.
        background_radiance_warning: High background radiance.
        manual_blower_status: Manual blower control.
    """

    def __init__(self, status_bits: int):
        self.laser_temp_shutoff_alarm = bool(status_bits & 0x80000000)
        self.laser_fail_alarm = bool(status_bits & 0x40000000)
        self.receiver_fail_alarm = bool(status_bits & 0x20000000)
        self.voltage_fail_alarm = bool(status_bits & 0x10000000)
        self.window_contam_warning = bool(status_bits & 0x00800000)
        self.battery_low_warning = bool(status_bits & 0x00400000)
        self.laser_power_low_warning = bool(status_bits & 0x00200000)
        self.laser_temp_warning = bool(status_bits & 0x00100000)
        self.internal_temp_warning = bool(status_bits & 0x00080000)
        self.voltage_range_warning = bool(status_bits & 0x00040000)
        self.humidity_high_warning = bool(status_bits & 0x00020000)
        self.receiver_crosstalk_warning = bool(status_bits & 0x00010000)
        self.blower_fail_warning = bool(status_bits & 0x00008000)
        self.blower_status = bool(status_bits & 0x00000800)
        self.blower_heater_status = bool(status_bits & 0x00000400)
        self.internal_heater_status = bool(status_bits & 0x00000200)
        self.units_meters = bool(status_bits & 0x00000100)
        self.polling_mode_status = bool(status_bits & 0x00000080)
        self.battery_power_status = bool(status_bits & 0x00000040)
        self.single_seq_mode_status = bool(status_bits & 0x00000020)
        self.manual_settings_status = bool(status_bits & 0x00000010)
        self.tilt_angle_warning = bool(status_bits & 0x00000008)
        self.background_radiance_warning = bool(status_bits & 0x00000004)
        self.manual_blower_status = bool(status_bits & 0x00000002)


@dataclass
class CtMessage(Message):
    """Data message from Vaisala CT25K.

    Attributes:
        receiver_sensitivity: Receiver sensitivity (%).
        window_contamination: Window contamination (mV).
        status: Decoded status bits.
    """

    receiver_sensitivity: int
    window_contamination: int
    status: CtStatus


def read_ct_file(
    filename: str | PathLike,
) -> tuple[list[datetime.datetime], list[CtMessage]]:
    """Read Vaisala CT25K file."""
    content = Path(filename).read_bytes()
    time = []
    data = []
    for ts, msg in utils.parse_file(content, FORMAT):
        try:
            data.append(read_ct_message(msg))
            time.append(ts)
        except (InvalidMessageError, ValueError) as e:
            logging.debug("Invalid message: %s", e)
    return time, data


def read_ct_message(message: bytes) -> CtMessage:
    """Read Vaisala CT25K data message."""
    lines = iter(message.splitlines())

    # Line 1
    line1 = utils.next_line(lines, 7, prefix=b"\x01", suffix=b"\x02")
    if line1[:2] != b"CT":
        msg = "Invalid line 1"
        raise InvalidMessageError(msg)

    msg_no = line1[5:6]
    if msg_no not in (b"2", b"7"):
        msg = f"Invalid message number: {msg_no.decode()}"
        raise InvalidMessageError(msg)

    # Line 2
    line2 = utils.next_line(lines, 29)
    status_bits = int(line2[21:], 16)
    status = CtStatus(status_bits)

    # Line 3
    line3 = utils.next_line(lines, 42)
    scale = int(line3[0:3])
    laser_pulse_energy = int(line3[6:9])
    laser_temperature = int(line3[10:13])
    receiver_sensitivity = int(line3[14:17])
    window_contamination = int(line3[18:22])
    tilt_angle = int(line3[23:26])
    background_light = int(line3[27:31])
    n_pulses = 4 ** (int(line3[34:35]) + 1)
    sample_rate = 10 * int(line3[37:38])

    # Lines 4-19: profile
    data = []
    for _i in range(16):
        line = utils.next_line(lines, 67)
        data.append(line[3:])
    raw = utils.read_hex(b"".join(data), 4, 256)
    beta = raw * 1e-7 * scale / 100

    return CtMessage(
        range_resolution=30,
        laser_pulse_energy=laser_pulse_energy,
        laser_temperature=laser_temperature,
        receiver_sensitivity=receiver_sensitivity,
        window_contamination=window_contamination,
        tilt_angle=tilt_angle,
        background_light=background_light,
        n_pulses=n_pulses,
        sample_rate=sample_rate,
        status=status,
        beta=beta,
    )
