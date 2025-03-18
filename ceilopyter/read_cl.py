import datetime
import logging
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from . import utils
from .common import InvalidMessageError, Message, Status

FORMAT = utils.date_format_to_regex(rb"-%Y-%m-%d %H:%M:%S\r?\n")


class ClStatus(Status):
    """Decoded status bits from Vaisala CL31 or CL51.

    Attributes:
        transmitter_shutoff_alarm: Transmitter shut-off.
        transmitter_fail_alarm: Transmitter failure.
        receiver_fail_alarm: Receiver failure.
        voltage_fail_alarm: Voltage failure.
        memory_error_alarm: Memory error.
        light_path_obstruction_alarm: Light path obstruction.
        receiver_saturation_alarm: Receiver saturation.
        coaxial_cable_fail_alarm: Coaxial cable failure.
        ceilometer_board_fail_alarm: Ceilometer engine board failure.
        window_contam_warning: Window contamination.
        battery_low_warning: Battery voltage low.
        transmitter_expire_warning: Transmitter expires.
        humidity_high_warning: High humidity.
        blower_fail_warning: Blower failure.
        humidity_sensor_fail_warning: Humidity sensor failure.
        heater_fault_warning: Heater fault.
        background_radiance_warning: High background radiance.
        ceilometer_board_warning: Ceilometer engine board failure.
        battery_fail_warning: Battery failure.
        laser_monitor_fail_warning: Laser monitor failure.
        receiver_warning: Receiver warning.
        tilt_angle_warning: Tilt angle > 45 degrees warning.
        blower_status: Blower is on.
        blower_heater_status: Blower heater is on.
        internal_heater_status: Internal heater is on.
        battery_power_status: Working from battery.
        standby_status: Standby mode is on.
        self_test_status: Self test in progress.
        manual_data_status: Manual data acquisition settings are effective.
        units_meters: Units are meters if on, else feet.
        manual_blower_status: Manual blower control.
        polling_mode_status: Polling mode is on.
    """

    def __init__(self, status_bits: int):
        self.transmitter_shutoff_alarm = bool(status_bits & 0x800000000000)
        self.transmitter_fail_alarm = bool(status_bits & 0x400000000000)
        self.receiver_fail_alarm = bool(status_bits & 0x200000000000)
        self.voltage_fail_alarm = bool(status_bits & 0x100000000000)
        self.memory_error_alarm = bool(status_bits & 0x040000000000)
        self.light_path_obstruction_alarm = bool(status_bits & 0x020000000000)
        self.receiver_saturation_alarm = bool(status_bits & 0x010000000000)
        self.coaxial_cable_fail_alarm = bool(status_bits & 0x000200000000)
        self.ceilometer_board_fail_alarm = bool(status_bits & 0x000100000000)
        self.window_contam_warning = bool(status_bits & 0x000080000000)
        self.battery_low_warning = bool(status_bits & 0x000040000000)
        self.transmitter_expire_warning = bool(status_bits & 0x000020000000)
        self.humidity_high_warning = bool(status_bits & 0x000010000000)
        self.blower_fail_warning = bool(status_bits & 0x000004000000)
        self.humidity_sensor_fail_warning = bool(status_bits & 0x000001000000)
        self.heater_fault_warning = bool(status_bits & 0x000000800000)
        self.background_radiance_warning = bool(status_bits & 0x000000400000)
        self.ceilometer_board_warning = bool(status_bits & 0x000000200000)
        self.battery_fail_warning = bool(status_bits & 0x000000100000)
        self.laser_monitor_fail_warning = bool(status_bits & 0x000000080000)
        self.receiver_warning = bool(status_bits & 0x000000040000)
        self.tilt_angle_warning = bool(status_bits & 0x000000020000)
        self.blower_status = bool(status_bits & 0x000000008000)
        self.blower_heater_status = bool(status_bits & 0x000000004000)
        self.internal_heater_status = bool(status_bits & 0x000000002000)
        self.battery_power_status = bool(status_bits & 0x000000001000)
        self.standby_status = bool(status_bits & 0x000000000800)
        self.self_test_status = bool(status_bits & 0x000000000400)
        self.manual_data_status = bool(status_bits & 0x000000000200)
        self.units_meters = bool(status_bits & 0x000000000080)
        self.manual_blower_status = bool(status_bits & 0x000000000040)
        self.polling_mode_status = bool(status_bits & 0x000000000020)


@dataclass
class ClMessage(Message):
    """Data message from Vaisala CL31 or CL51.

    Attributes:
        window_transmission: Window transmission (%).
        status: Decoded status bits.
    """

    window_transmission: int
    status: ClStatus


def read_cl_file(
    filename: str | PathLike,
) -> tuple[list[datetime.datetime], list[ClMessage]]:
    """Read Vaisala CL31 or CL51 file."""
    content = Path(filename).read_bytes()
    time = []
    data = []
    for ts, msg in utils.parse_file(content, FORMAT):
        try:
            data.append(read_cl_message(msg))
            time.append(ts)
        except (InvalidMessageError, ValueError) as e:
            logging.debug("Invalid message: %s", e)
    return time, data


def read_cl_message(message: bytes) -> ClMessage:
    """Read Vaisala CL31 or CL51 data message."""
    lines = iter(message.splitlines())

    # Line 1
    line1 = utils.next_line(lines, 8, prefix=b"\x01", suffix=b"\x02")
    if line1[:2] != b"CL":
        msg = "Invalid line 1"
        raise InvalidMessageError(msg)
    msg_no = line1[6:7]
    if msg_no not in (b"1", b"2"):
        msg = f"Invalid message number: {msg_no.decode()}"
        raise InvalidMessageError(msg)
    subclass = line1[7:8]
    if subclass in (b"1", b"2", b"3", b"4"):
        line3_len = 31  # CL31
    elif subclass == b"6":
        line3_len = 40  # CL51
    else:
        msg = f"Invalid message subclass: {subclass.decode()}"
        raise InvalidMessageError(msg)
    check_content = line1 + b"\x02\r\n"

    # Line 2
    line2 = utils.next_line(lines, 33)
    status_bits = int(line2[21:], 16)
    status = ClStatus(status_bits)
    check_content += line2 + b"\r\n"

    # Line 3: sky condition
    if msg_no == b"2":
        line3 = utils.next_line(lines).rjust(line3_len)
        check_content += line3 + b"\r\n"

    # Line 3/4
    line4 = utils.next_line(lines, 47)
    scale = int(line4[0:5])
    range_resolution = int(line4[6:8])
    n_samples = int(line4[9:13])
    laser_pulse_energy = int(line4[14:17])
    laser_temperature = int(line4[18:21])
    window_transmission = int(line4[22:25])
    tilt_angle = int(line4[26:28])
    background_light = int(line4[29:33])
    n_pulses = 1024 * int(line4[35:39])
    sample_rate = int(line4[41:43])
    check_content += line4 + b"\r\n"

    # Line 4/5: profile
    line5 = utils.next_line(lines, 5 * n_samples)
    raw = utils.read_hex(line5, 5, n_samples)
    beta = raw * 1e-8 * scale / 100
    check_content += line5 + b"\r\n\x03"

    # Line 5/6: checksum
    line6 = utils.next_line(lines, 4, prefix=b"\x03", suffix=b"\x04")
    expected_checksum = int(line6, 16)

    actual_checksum = utils.crc16(check_content)
    if expected_checksum != actual_checksum:
        msg = (
            "Invalid checksum: "
            f"expected {expected_checksum:04x}, "
            f"got {actual_checksum:04x}"
        )
        raise InvalidMessageError(msg)

    return ClMessage(
        range_resolution=range_resolution,
        laser_pulse_energy=laser_pulse_energy,
        laser_temperature=laser_temperature,
        window_transmission=window_transmission,
        tilt_angle=tilt_angle,
        background_light=background_light,
        n_pulses=n_pulses,
        sample_rate=sample_rate,
        status=status,
        beta=beta,
    )
