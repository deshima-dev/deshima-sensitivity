import numpy as np
from math import isclose
from deshima_sensitivity import physics


def test_T_from_psd():
    output = physics.T_from_psd(1e11, 1e-26, "Rayleigh-Jeans")
    expected = 0.00072429730  # calculated by python
    assert isclose(output, expected, rel_tol=0.01)

