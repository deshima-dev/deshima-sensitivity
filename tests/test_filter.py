from math import isclose
from deshima_sensitivity import filter
import numpy as np


def test_lorentzian_gen():
    expected = np.pi * 300e9 / 500
    eta_filter, F, F_int, W_F_int, box_height, box_width = filter.eta_filter_lorentzian(
        F=300e9,
        HWHM=300e9 / 500,
        eta_filter=1,
        F_res=30,
        overflow=10,
    )
    output = np.sum(eta_filter * W_F_int)
    assert isclose(output, expected, rel_tol=0.05)
