import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_array_equal

from ceilopyter.utils import interpolate_masked


@pytest.mark.parametrize(
    "array,mask,expected",
    [
        ([0.0, 0.0, 3.0], [1, 1, 0], [3.0, 3.0, 3.0]),
        ([2.0, 0.0, 0.0], [0, 1, 1], [2.0, 2.0, 2.0]),
        ([1.0, 0.0, 3.0], [0, 1, 0], [1.0, 2.0, 3.0]),
        ([1.0, 0.0, 3.0, 0.0, 5.0], [0, 1, 0, 1, 0], [1.0, 2.0, 3.0, 4.0, 5.0]),
        ([1.0, 0.0, 0.0, 0.0, 3.0], [0, 1, 1, 1, 0], [1.0, 1.5, 2.0, 2.5, 3.0]),
    ],
)
def test_interpolate_masked(array, mask, expected):
    input = ma.array(array, mask=mask)
    actual = interpolate_masked(input)
    assert not ma.is_masked(actual)
    assert_array_equal(actual, expected)


def test_interpolate_all_masked():
    array = interpolate_masked(ma.masked_all(5))
    assert len(array) == 5
    assert np.all(array.mask)
