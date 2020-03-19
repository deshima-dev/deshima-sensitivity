# dependent packages
import numpy as np
from pandas.testing import assert_series_equal
from desim.physics import Layer, Radiation


# constants
layer_1 = Layer(0.5, 4)
layer_2 = Layer(0.5, 1)


# test functions
def test_layer_12():
    assert (layer_1 | layer_2) == Layer(0.25, 2.0)


def test_layer_21():
    assert (layer_2 | layer_1) == Layer(0.25, 3.0)


def test_radiation_1():
    rad = Radiation(np.ones(3), np.linspace(100e9, 300e9, 3))
    assert_series_equal(rad.to_psd().to_temperature(), rad)


def test_radiation_2():
    rad = Radiation(np.full(3, 1e-26), np.linspace(100e9, 300e9, 3))
    rad.units = "W/Hz"
    assert_series_equal(rad.to_temperature().to_psd(), rad)


def test_transfer_1():
    rad_in = Radiation(np.ones(3))
    rad_out = Radiation(np.full(3, 2.5))
    assert_series_equal(rad_in | layer_1, rad_out)


def test_transfer_2():
    rad_in = Radiation(np.ones(3))
    rad_out = Radiation(np.ones(3))
    assert_series_equal(rad_in | layer_2, rad_out)
