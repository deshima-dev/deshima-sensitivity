# standard library
from dataclasses import dataclass


# dependent packages
import numpy as np


# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]


# constants
h = 6.62607004 * 10 ** -34  # Planck constant
k = 1.38064852 * 10 ** -23  # Boltzmann constant
e = 1.60217662 * 10 ** -19  # electron charge
c = 299792458.0  # velocity of light


# main classes
@dataclass(frozen=True)
class Layer:
    efficiency: ArrayLike
    brightness: ArrayLike = 0.0

    @property
    def b(self: "Layer") -> ArrayLike:
        """Alias of ``brightness`` attribute."""
        return self.brightness

    @property
    def eff(self: "Layer") -> ArrayLike:
        """Alias of ``efficiency`` attribute."""
        return self.efficiency

    def transfer(self: "Layer", radiation: "Radiation") -> "Radiation":
        """Compute radiative transfer through a layer."""
        return self.eff * radiation + (1 - self.eff) * self.b

    def append(self: "Layer", layer: "Layer") -> "Layer":
        """Append a layer to compose a new one."""
        eff_1, eff_2 = self.eff, layer.eff
        b_1, b_2 = self.b, layer.b

        if np.any(eff_1 * eff_2 == 1.0):
            raise ValueError("Cannot append a layer.")

        eff_12 = eff_1 * eff_2
        b_12 = (1 - eff_1) / (1 - eff_12) * (eff_2 * b_1)
        b_12 += (1 - eff_2) / (1 - eff_12) * b_2

        return Layer(eff_12, b_12)

    def __or__(self: "Layer", layer: "Layer") -> "Layer":
        """Operator for ``append`` method."""
        return self.append(layer)


class Radiation(np.ndarray):
    def __new__(cls, brightness: ArrayLike, **kwargs) -> "Radiation":
        """Create a radiation instance which express brightness."""
        return np.asarray(brightness, **kwargs).view(cls)

    def to_array(self: "Radiation") -> np.ndarray:
        """Convert it to a pure NumPy array."""
        return self.view(np.ndarray)

    def pass_through(self: "Radiation", layer: "Layer") -> "Radiation":
        """Compute radiative transfer through a layer."""
        return layer.transfer(self)

    def __or__(self: "Radiation", layer: "Layer") -> "Radiation":
        """Operator for ``path_through`` method."""
        return self.pass_through(layer)


# main functions
def rad_trans(rad_in: ArrayLike, medium: ArrayLike, eta: ArrayLike) -> ArrayLike:
    """Calculates radiation transfer through a semi-transparent medium.

    One can also use the same function for Johnson-Nyquist PSD
    (power spectral density) instead of temperature.

    Parameters
    ----------
    rad_in
        Brightness temperature (or PSD) of the input. Units: K (or W/Hz).
    medium
        Brightness temperature (or PSD) of the lossy medium. Units: K (or W/Hz).
    eta
        Transmission of the lossy medium. Units: K (or W/Hz).

    Returns
    -------
    rad_out
        Brightness temperature (or PSD) of the output.

    """
    return eta * rad_in + (1 - eta) * medium


def T_from_psd(F: ArrayLike, psd: ArrayLike, method: str = "Planck") -> ArrayLike:
    """Calculate Planck temperature from the PSD frequency (frequencies).

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    psd
        Power Spectral Density. Units: W / Hz.
    method
        Default: 'Planck'. Option: 'Rayleigh-Jeans'.

    Returns
    --------
    T
        Planck temperature. Units: K.

    """
    if method == "Planck":
        return h * F / (k * np.log(h * F / psd + 1.0))
    elif method == "Rayleigh-Jeans":
        return psd / k
    else:
        raise ValueError("Method should be Planck or Rayleigh-Jeans.")


def johnson_nyquist_psd(F: ArrayLike, T: ArrayLike) -> ArrayLike:
    """Johnson-Nyquist power spectral density.

    Don't forget to multiply with bandwidth to caculate the total power in W.

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    T
        Temperature. Units: K.

    Returns
    --------
    psd
        Power Spectral Density. Units: W / Hz.

    """
    return h * F * nph(F, T)


# helper functions
def nph(F: ArrayLike, T: ArrayLike) -> ArrayLike:
    """Photon occupation number of Bose-Einstein Statistics.

    If it is not single temperature, use nph = Pkid / (W_F * h * F).

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    T
        Temperature. Units: K.

    Returns
    --------
    n
        Photon occupation number. Units: None.

    """
    return 1.0 / (np.exp(h * F / (k * T)) - 1.0)
