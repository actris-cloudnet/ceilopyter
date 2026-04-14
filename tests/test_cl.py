import datetime
from pathlib import Path

from ceilopyter import read_cl_file, read_cl_message


def test_skip_invalid():
    time, data = read_cl_file("tests/data/celio_chennai_2025-03-11.dat")
    assert len(time) == 2
    assert len(data) == 2

    assert time[0] == datetime.datetime(2025, 3, 11, 8, 4, 55)
    assert data[0].laser_temperature == 43
    assert data[0].status.blower_fail_warning

    assert time[1] == datetime.datetime(2025, 3, 11, 8, 6, 58)
    assert data[1].laser_temperature == 42
    assert not data[1].status.blower_fail_warning


def test_kenttarova_cl31_msg():
    path = Path("tests/data/kenttarova_cl31_msg.dat")
    msg = read_cl_message(path.read_bytes())
    assert msg.range_resolution == 10


def test_palaiseau_cl31_msg():
    path = Path("tests/data/palaiseau_cl31_msg.dat")
    msg = read_cl_message(path.read_bytes())
    assert msg.range_resolution == 5


def test_uto_cl31_msg():
    path = Path("tests/data/uto_cl31_msg.dat")
    msg = read_cl_message(path.read_bytes())
    assert msg.range_resolution == 10


def test_kauniainen_cl31():
    time, data = read_cl_file("tests/data/kauniainen_cl31.dat")
    assert len(time) == len(data) == 2
    assert time[0] == datetime.datetime(2025, 2, 2, 0, 0, 3)
    assert time[1] == datetime.datetime(2025, 2, 2, 0, 0, 18)
    assert data[0].range_resolution == 10
    assert data[0].laser_temperature == 26
