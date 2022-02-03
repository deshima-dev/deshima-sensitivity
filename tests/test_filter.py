from math import isclose
from deshima_sensitivity import filter
import numpy as np


def test_lorentzian_gen():
    expected = 2 * 300e9 / 500
    (
        eta_filter,
        eta_inband,
        F,
        F_int,
        W_F_int,
        box_height,
        box_width,
        chi_sq,
    ) = filter.eta_filter_lorentzian(F=300e9, FWHM=300e9 / 500, eta_circuit=1)
    output = np.sum(eta_filter * W_F_int)
    assert isclose(output, expected, rel_tol=0.005)
