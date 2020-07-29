from math import isclose
from deshima_sensitivity import physics


def test_johnson_nyquist_psd():
    expected = 1.3833132 * 10 ** -21
    output = physics.johnson_nyquist_psd(10, 100)
    assert output == expected


def test_nph():
    expected = 2.0876827 * 10 ** 11
    output = physics.nph(10, 100)
    assert output == expected
