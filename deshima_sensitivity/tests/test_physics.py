from math import isclose
from deshima_sensitivity import physics


def test_rad_trans():
    expected = 1.2
    output = physics.rad_trans(2.0, 1.0, 0.2)
    assert isclose(expected, output)
