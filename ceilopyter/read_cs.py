import datetime
import logging
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from . import utils
from .common import InvalidMessageError, Message, Status

FORMATS = [
    utils.date_format_to_regex(rb"%Y-%m-%dT%H:%M:%S.%f,"),
    utils.date_format_to_regex(rb"%%% %Y/%m/%d %H:%M:%S %%%\r?\n"),
]


class CsStatus(Status):
    """Decoded status bits from Campbell Scientific CS135.

    Attributes:
        units_meters: Units. Feet = 0, metre = 1.
        dsp_clock_warning: DSP clock out of specification.
        laser_temp_alarm: Laser shut down due to operating temperature out of range.
        battery_voltage_warning: The lead acid battery voltage is reading low.
        mains_supply_alarm: Mains supply has failed (Required a PSU to be present).
        heater_temp_warning: The external heater blower assembly temperature is out of
            bounds.
        heater_blower_alarm: External heater blower failure.
        psu_temp_warning: The PSUs internal temperature is high.
        psu_os_alarm: PSU OS has failed its signature check.
        dsp_psu_comm_alarm: No communications between DSP and PSU.
        laser_window_warning: Photo diode and Laser windows are dirty. This can only be
            set if the laser is on.
        tilt_angle_warning: Tilt beyond limit set by user, default 45 degrees.
        dsp_inclinometer_comm_alarm: No communications between DSP and inclinometer
            board.
        sensor_humidity_warning: The sensors internal humidity is high.
        dsp_temp_humidity_comm_alarm: Communications to the DSP board temperature and
            humidity chip have failed.
        dsp_voltage_warning: DSP input supply voltage is low.
        self_test_status: Self-test active.
        watchdog_status: Watch dog counter updated.
        user_settings_alarm: User setting stored in flash failed their signature checks.
        dsp_calibration_alarm: DSP factory calibration stored in flash has failed its
            signature check.
        dsp_os_alarm: DSP board OS signature test failed.
        dsp_ram_alarm: DSP board RAM test failed.
        dsp_psu_warning: DSP boards on board PSUs are out of bounds.
        top_storage_alarm: TOP board non-volatile storage is corrupt.
        top_os_alarm: TOP board OS signature test has failed.
        top_adc_dac_warning: TOP boards ADC and DAC are not within specifications.
        top_psu_warning: TOP boards on board PSUs are out of bounds.
        top_dsp_comm_alarm: Communications have failed between TOP board and the DSP.
        photo_diode_radiance_warning: Photo diode background radiance is out of range.
        photo_diode_temp_warning: Photo diode temperature is out of range.
        photo_diode_saturation_warning: Photo diode is saturated.
        photo_diode_calibrator_temp_warning: Photo diode calibrator temperature is out
            of range.
        photo_diode_calibrator_alarm: Photo diode calibrator has failed.
        sensor_gain_warning: The sensor could not reach the desired gain levels.
        laser_runtime_alarm: Laser run time or maximum laser drive current has been
            exceeded.
        laser_temp_warning: Laser temperature out of range.
        laser_thermistor_alarm: Laser thermistor failure.
        laser_obscured_warning: Laser is obscured. This can only be set if the laser is
            on.
        laser_power_alarm: Laser did not achieve significant output power.
        laser_max_power_alarm: Laser max power exceeded.
        laser_drive_current_alarm: Laser max drive current exceeded.
        laser_monitor_temp_warning: Laser power monitor temperature out of range.
        laser_monitor_test_alarm: Laser power monitor test fail.
        laser_shutdown_status: Laser shutdown by top board.
        laser_off_status: Laser is off.
    """

    def __init__(self, status_bits: int):
        self.units_meters = bool(status_bits & 0x800000000000)
        self.dsp_clock_warning = bool(status_bits & 0x080000000000)
        self.laser_temp_alarm = bool(status_bits & 0x040000000000)
        self.battery_voltage_warning = bool(status_bits & 0x020000000000)
        self.mains_supply_alarm = bool(status_bits & 0x010000000000)
        self.heater_temp_warning = bool(status_bits & 0x008000000000)
        self.heater_blower_alarm = bool(status_bits & 0x004000000000)
        self.psu_temp_warning = bool(status_bits & 0x002000000000)
        self.psu_os_alarm = bool(status_bits & 0x001000000000)
        self.dsp_psu_comm_alarm = bool(status_bits & 0x000800000000)
        self.laser_window_warning = bool(status_bits & 0x000400000000)
        self.tilt_angle_warning = bool(status_bits & 0x000200000000)
        self.dsp_inclinometer_comm_alarm = bool(status_bits & 0x000100000000)
        self.sensor_humidity_warning = bool(status_bits & 0x000080000000)
        self.dsp_temp_humidity_comm_alarm = bool(status_bits & 0x000040000000)
        self.dsp_voltage_warning = bool(status_bits & 0x000020000000)
        self.self_test_status = bool(status_bits & 0x000010000000)
        self.watchdog_status = bool(status_bits & 0x000008000000)
        self.user_settings_alarm = bool(status_bits & 0x000004000000)
        self.dsp_calibration_alarm = bool(status_bits & 0x000002000000)
        self.dsp_os_alarm = bool(status_bits & 0x000001000000)
        self.dsp_ram_alarm = bool(status_bits & 0x000000800000)
        self.dsp_psu_warning = bool(status_bits & 0x000000400000)
        self.top_storage_alarm = bool(status_bits & 0x000000200000)
        self.top_os_alarm = bool(status_bits & 0x000000100000)
        self.top_adc_dac_warning = bool(status_bits & 0x000000080000)
        self.top_psu_warning = bool(status_bits & 0x000000040000)
        self.top_dsp_comm_alarm = bool(status_bits & 0x000000020000)
        self.photo_diode_radiance_warning = bool(status_bits & 0x000000010000)
        self.photo_diode_temp_warning = bool(status_bits & 0x000000008000)
        self.photo_diode_saturation_warning = bool(status_bits & 0x000000004000)
        self.photo_diode_calibrator_temp_warning = bool(status_bits & 0x000000002000)
        self.photo_diode_calibrator_alarm = bool(status_bits & 0x000000001000)
        self.sensor_gain_warning = bool(status_bits & 0x000000000800)
        self.laser_runtime_alarm = bool(status_bits & 0x000000000400)
        self.laser_temp_warning = bool(status_bits & 0x000000000200)
        self.laser_thermistor_alarm = bool(status_bits & 0x000000000100)
        self.laser_obscured_warning = bool(status_bits & 0x000000000080)
        self.laser_power_alarm = bool(status_bits & 0x000000000040)
        self.laser_max_power_alarm = bool(status_bits & 0x000000000020)
        self.laser_drive_current_alarm = bool(status_bits & 0x000000000010)
        self.laser_monitor_temp_warning = bool(status_bits & 0x000000000008)
        self.laser_monitor_test_alarm = bool(status_bits & 0x000000000004)
        self.laser_shutdown_status = bool(status_bits & 0x000000000002)
        self.laser_off_status = bool(status_bits & 0x000000000001)


@dataclass
class CsMessage(Message):
    """Data message from Campbell Scientific CS135.

    Attributes:
        window_transmission: Window transmission (%).
        status: Decoded status bits.
    """

    window_transmission: int  # %
    status: CsStatus


def read_cs_file(
    filename: str | PathLike,
) -> tuple[list[datetime.datetime], list[CsMessage]]:
    """Read Campbell Scientific CS135 file."""
    content = Path(filename).read_bytes()
    time = []
    data = []
    for fmt in FORMATS:
        for ts, msg in utils.parse_file(content, fmt):
            try:
                data.append(read_cs_message(msg))
                time.append(ts)
            except (InvalidMessageError, ValueError) as e:
                logging.debug("Invalid message: %s", e)
    return time, data


def read_cs_message(message: bytes) -> CsMessage:
    """Read Campbell Scientific CS135 data message."""
    lines = iter(message.splitlines())

    # Line 1
    line1 = utils.next_line(lines, 9, b"\x01", b"\x02")
    if line1[:2] != b"CS":
        msg = "Invalid line 1"
        raise InvalidMessageError(msg)
    msg_no = line1[6:9]
    if msg_no not in (b"002", b"004"):
        msg = f"Invalid message number: {msg_no.decode()}"
        raise InvalidMessageError(msg)
    check_content = line1 + b"\x02\r\n"

    # Line 2
    line2 = utils.next_line(lines, 43)
    status_bits = int(line2[31:], 16)
    status = CsStatus(status_bits)
    window_transmission = int(line2[3:6])
    check_content += line2 + b"\r\n"

    # Line 3: sky condition
    if msg_no == b"004":
        line3 = utils.next_line(lines).rjust(40)
        check_content += line3 + b"\r\n"

    # Line 3/4
    line4 = utils.next_line(lines, 41)
    scale = int(line4[0:5])
    range_resolution = int(line4[6:8])
    n_samples = int(line4[9:13])
    laser_pulse_energy = int(line4[14:17])
    laser_temperature = int(line4[18:21])
    tilt_angle = int(line4[22:24])
    background_light = int(line4[25:29])
    n_pulses = 1000 * int(line4[30:34])
    sample_rate = int(line4[35:37])
    check_content += line4 + b"\r\n"

    # Line 4/5: profile
    line5 = utils.next_line(lines, 5 * n_samples)
    raw = utils.read_hex(line5, 5, n_samples)
    beta = raw * 1e-8 * scale / 100
    check_content += line5 + b"\r\n\x03"

    # Line 5/6: checksum
    line6 = utils.next_line(lines, 4, b"\x03", b"\x04")
    expected_checksum = int(line6, 16)

    actual_checksum = utils.crc16(check_content)
    if expected_checksum != actual_checksum:
        msg = (
            "Invalid checksum: "
            f"expected {expected_checksum:04x}, "
            f"got {actual_checksum:04x}"
        )
        raise InvalidMessageError(msg)

    return CsMessage(
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
