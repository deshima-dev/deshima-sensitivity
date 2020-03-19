# standard library
from typing import Type, Union
from dataclasses import dataclass

# dependent packages
import numpy as np
import pandas as pd
from .consts import h, k


# type aliases
ArrayLike = Union[np.ndarray, float, int]


# constants
UNITS_T = "K"
UNITS_PSD = "W/Hz"


# main classes
class Radiation(pd.Series):
    """Radiation class for calculating radiative transfer.

    This is a subclass of pandas Series with custom attributes and methods.
    An instance is created by ``radiation = Radiation(data, index)``,
    where ``data`` is value(s) of radiation expressed as (brightness) temperature
    (K) and ``index`` is value(s) of frequency (Hz) which correspond to data.

    """

    _metadata = ["units"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.units = UNITS_T

    @property
    def _constructor(self) -> Type["Radiation"]:
        """Special property for subclassing pandas Series."""
        return Radiation

    @property
    def nu(self) -> np.ndarray:
        """Returns frequency index as a NumPy array."""
        return self.index.values

    def to_psd(self) -> "Radiation":
        """Convert temperature (K) to Johnson-Nyquist PSD (W/Hz).

        This method can only be used if radiation is expressed as
        (brightness) temperature (i.e., ``radiation.units == 'K'``).

        Returns
        -------
        radiation
            Radiation instance expressed as Johnson-Nyquist PSD.

        """
        if not self.units == UNITS_T:
            raise ValueError(f"Units must be {UNITS_T}.")

        new = h * self.nu / (np.exp(h * self.nu / (k * self)) - 1.0)
        new.units = UNITS_PSD
        return new

    def to_temperature(self, method: str = "planck") -> "Radiation":
        """Convert Johnson-Nyquist PSD (W/Hz) to temperature (K).

        This method can only be used if radiation is expressed as
        Johnson-Nyquist PSD (i.e., ``radiation.units == 'W/Hz'``).

        Parameters
        ----------
        method
            Conversion method. Must be either 'planck' (default: no RJ
            approximation) or 'rj' (Rayleigh-Jeans approximation).

        Returns
        -------
        radiation
            Radiation instance expressed as (brightness) temperature.

        """
        if not self.units == UNITS_PSD:
            raise ValueError(f"Units must be {UNITS_PSD}.")

        if method == "planck":
            new = h * self.nu / (k * np.log(h * self.nu / self + 1.0))
        elif method == "rj":
            new = self / k
        else:
            raise ValueError("Method must be either 'planck' or 'rj'.")

        new.units = UNITS_T
        return new

    def pass_through(self, layer: "Layer") -> "Radiation":
        """Compute radiative transfer through a layer."""
        return layer.transfer(self)

    def __or__(self, layer: "Layer") -> "Radiation":
        """Operator for ``pass_through`` method.

        This means that a code ``radiation | layer`` is
        equivalent to ``radiation.pass_through(layer)``.

        """
        return self.pass_through(layer)


@dataclass(frozen=True)
class Layer:
    """Semi-transparent layer class for calculating radiative transfer.

    An instance is created by ``layer = Layer(eff, src)``,
    where ``eff`` is efficiency (transparency) of the layer
    and ``src`` is a source function (either in K or W/Hz) of the layer.
    Radiative transfer is expressed by ``rad_out = rad_in | layer``,
    where ``rad_in`` and ``rad_out`` are instances of ``Radiation`` class.
    This is equivalent to ``rad_out = eff * rad_in + (1-eff) * src``.
    Radiative transfer with multiple layers is similarly expressed::

        rad_out = rad_in | layer_1 | layer_2 | ...

    Multiple layers can be combined before radiative transfer::

        # equivalent to the example above
        layers = layer_1 | layer_2 | ...
        rad_out = rad_in | layers

    This makes it easy to reuse the layer and reduce computation costs.

    Parameters
    ----------
    efficiency
        Efficiency (transparency) of the layer.
    source
        Source function expressed by either brighness temperature (K)
        or Johnson-Nyquist PSD (W/Hz). Units of input radiation must
        be same as those of source.

    """

    efficiency: ArrayLike = 1.0
    source: Union[Radiation, ArrayLike] = 0.0

    @property
    def eff(self: "Layer") -> ArrayLike:
        """Alias of ``efficiency`` attribute."""
        return self.efficiency

    @property
    def src(self: "Layer") -> Union[Radiation, ArrayLike]:
        """Alias of ``source`` attribute."""
        return self.source

    def transfer(self, radiation: Radiation) -> Radiation:
        """Compute radiative transfer through a layer."""
        return self.eff * radiation + (1 - self.eff) * self.src

    def append(self, layer: "Layer") -> "Layer":
        """Append a layer to compose a new one."""
        eff_1, eff_2 = self.eff, layer.eff
        src_1, src_2 = self.src, layer.src

        if np.any(eff_1 * eff_2 == 1.0):
            raise ValueError("Cannot append a layer.")

        eff_12 = eff_1 * eff_2
        src_12 = (1 - eff_1) / (1 - eff_12) * (eff_2 * src_1)
        src_12 += (1 - eff_2) / (1 - eff_12) * src_2

        return Layer(eff_12, src_12)

    def __or__(self, layer: "Layer") -> "Layer":
        """Operator for ``append`` method."""
        return self.append(layer)
