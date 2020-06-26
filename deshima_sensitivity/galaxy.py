# standard library
from pathlib import Path
from typing import List, Union, Tuple


# dependent packages
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]


# main functions
def lineflux(
    Lfir: float = 5.0e13, switch_dwarf: bool = False
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, float, float, float, ArrayLike]:
    """Calculate astronomical line fluxes, developed by Y. Tamura.

    Parameters
    ----------
    Lfir
        Total infrared luminosity. Units: L_Sun.
    switch_dwarf
        Whether to use line-to-TIR ratios for dwarf galaxies (True) or not (False).

    Returns
    -------
    Fcii_DGS
        Flux(es) of [CII]. Units: W m^-2.
    Foiii_DGS
        Flux(es) of [OIII]. Units: W m^-2.
    Foi_DGS
        Flux(es) of [OI]. Units: W m^-2.
    f_cii
        Frequency of [CII]. Units: GHz.
    f_oiii
        Frequency of [OIII]. Units: GHz.
    f_oi
        Frequency of [OI]. Units: GHz.
    z
        Redshift(s) at which fluxes are calculated.

    """
    # line-to-TIR luminosity ratio (L_Sun or Watt)
    Rcii_B08, Roiii_B08, Roi_B08 = 1.3e-3, 8.0e-4, 1.0e-3  # from Brauer+2008
    Rcii_DGS, Roiii_DGS, Roi_DGS = 2.5e-3, 5.0e-3, 1.7e-3  # from Cormier+2015

    # rest frequency (GHz)
    f_cii, f_oiii, f_oi = 1900.5369, 3393.00062, 4744.8

    z_Dl_df = pd.read_csv(
        Path(__file__).parent / "data" / "z_Dl.csv",
        skiprows=0,
        delim_whitespace=False,
        header=0,
    )

    z = z_Dl_df.values[:, 0]
    Dl = z_Dl_df.values[:, 1]
    Dl_at_z = interp1d(z, Dl)

    # luminosity distance (Mpc)
    z = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    d_l = Dl_at_z(z)

    Fcii_B08 = flux_from_line_luminosity(z, d_l, f_cii / (1 + z), Lfir * Rcii_B08)
    Foiii_B08 = flux_from_line_luminosity(z, d_l, f_oiii / (1 + z), Lfir * Roiii_B08)
    Foi_B08 = flux_from_line_luminosity(z, d_l, f_oi / (1 + z), Lfir * Roi_B08)

    Fcii_DGS = flux_from_line_luminosity(z, d_l, f_cii / (1 + z), Lfir * Rcii_DGS)
    Foiii_DGS = flux_from_line_luminosity(z, d_l, f_oiii / (1 + z), Lfir * Roiii_DGS)
    Foi_DGS = flux_from_line_luminosity(z, d_l, f_oi / (1 + z), Lfir * Roi_DGS)

    if switch_dwarf:
        Fcii_DGS, Foiii_DGS, Foi_DGS = Fcii_DGS, Foiii_DGS, Foi_DGS
    else:
        Fcii_DGS, Foiii_DGS, Foi_DGS = Fcii_B08, Foiii_B08, Foi_B08

    return Fcii_DGS, Foiii_DGS, Foi_DGS, f_cii, f_oiii, f_oi, z


# helper functions
def flux_from_line_luminosity(
    z: ArrayLike, d_l: ArrayLike, f_obs: float, L: float
) -> ArrayLike:
    """Calculate line flux from luminosity.

    Paramters
    ---------
    d_l
        Luminosity distance. Units: Mpc.
    f_obs
        Observing frequency. Units: GHz.
    L
        Line luminosity. Units: L_Sun.

    Returns
    -------
    flux
        Flux of line. Units: W m^-2.

    """
    L_for1Jykms = co_luminosity(z, d_l, f_obs, 1.000)  # Lsun
    F_for1Jykms = 1.000 * 1e-26 * (f_obs * 1e9 / 299792.458)  # W m^-2

    return L * (F_for1Jykms / L_for1Jykms)


def co_luminosity(z, d_l, f_obs, int):
    # c1 = 3.25e07
    c2 = 1.04e-03

    # Lp  = c1 * int * d_l**2 / (f_obs**2 * (1 + z)**3 )
    return c2 * int * d_l ** 2 * f_obs
