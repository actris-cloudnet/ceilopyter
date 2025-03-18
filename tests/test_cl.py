import datetime

from ceilopyter import read_cl_file


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
