from math import isclose
from deshima_sensitivity import physics


def test_johnson_nyquist_psd():
    expected = 1.34778 * 10 ** -21
    output = physics.johnson_nyquist_psd(1e11, 100)
    assert isclose(output, expected, rel_tol=0.01)


def test_nph():
    expected = 20.3406
    output = physics.nph(1e11, 100)
    assert isclose(output, expected, rel_tol=0.01)


def test_T_from_psd():
    output = physics.T_from_psd(1e11, 1e-26, "Rayleigh-Jeans")
    expected = 0.00072429730  # calculated by python
    assert isclose(output, expected, rel_tol=0.01)


def test_rad_trans():
    expected = 1.2
    output = physics.rad_trans(2.0, 1.0, 0.2)
    assert isclose(expected, output)
