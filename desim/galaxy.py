__all__ = ["lineflux"]


# standard library
import os


# dependent packages
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# main functions
def lineflux(
    Lfir = 5.e+13, ## total infrared luminosity (L_Sun)
    switch_dwarf = False ## would you like to use line-to-TIR ratios for dwarf galaxies?
    ):
    ### Calculation of Astronomical line, developed by Y. Tamura --------

    ## line-to-TIR luminosity ratio (L_Sun or Watt)
    Rcii_B08, Roiii_B08, Roi_B08 = 1.3e-3, 8.0e-4, 1.0e-3 ## from Brauer+2008
    Rcii_DGS, Roiii_DGS, Roi_DGS = 2.5e-3, 5.0e-3, 1.7e-3 ## from Cormier+2015

    ## rest frequency (GHz)
    f_cii, f_oiii, f_oi = 1900.5369, 3393.00062, 4744.8

    z_Dl_df = pd.read_csv(
        # os.path https://qiita.com/ymdymd/items/d758110d429f72bc10fb
        os.path.dirname(__file__)+'/data/z_Dl.csv',
        skiprows=0,
        delim_whitespace=False,
        header=0
        )

    z = z_Dl_df.values[:,0]
    Dl = z_Dl_df.values[:,1]

    Dl_at_z = interp1d(z, Dl)

    ## redshift
    #     exec(open(desim_path + '/Dl_at_z2.py').read())
    #z = concatenate( (arange(0.1,0.5,0.1), arange(0.5,2,0.5), arange(2,6,1)), 1)
    #z = concatenate( (arange(0.1,0.5,0.1), arange(0.5,2,0.5), arange(2,12,1)), 1)
    #z = concatenate( (arange(0.1,1,0.1), arange(1,8,0.2)), 1)
    z = np.array([3,4,5,6,7,8,9,10,11,12])
    d_l = Dl_at_z(z) ## luminosity distance (Mpc)

    Fcii_B08  = flux_from_line_luminosity(z, d_l, f_cii/(1+z),  Lfir * Rcii_B08)
    Foiii_B08 = flux_from_line_luminosity(z, d_l, f_oiii/(1+z), Lfir * Roiii_B08)
    Foi_B08   = flux_from_line_luminosity(z, d_l, f_oi/(1+z),   Lfir * Roi_B08)

    Fcii_DGS  = flux_from_line_luminosity(z, d_l, f_cii/(1+z),  Lfir * Rcii_DGS)
    Foiii_DGS = flux_from_line_luminosity(z, d_l, f_oiii/(1+z), Lfir * Roiii_DGS)
    Foi_DGS   = flux_from_line_luminosity(z, d_l, f_oi/(1+z),   Lfir * Roi_DGS)

    if switch_dwarf:
        Fcii_DGS, Foiii_DGS, Foi_DGS = Fcii_DGS, Foiii_DGS, Foi_DGS
    else:
        Fcii_DGS, Foiii_DGS, Foi_DGS = Fcii_B08, Foiii_B08, Foi_B08

    return Fcii_DGS, Foiii_DGS, Foi_DGS, f_cii, f_oiii, f_oi, z


# helper functions
def flux_from_line_luminosity(z, d_l, f_obs, L):
    """
        d_l - luminosity distance (Mpc)
        f_obs - observing frequency (GHz)
        L - line luminosity (L_Sun)
    """

    L_for1Jykms = co_luminosity(z, d_l, f_obs,  1.000) ## Lsun
    F_for1Jykms = 1.000 * 1e-26 * (f_obs  * 1e9 / 299792.458) ## W m^-2

    return L * (F_for1Jykms / L_for1Jykms)


def co_luminosity(z, d_l, f_obs, int):

    c1, c2 = 3.25e+07, 1.04e-03

    #Lp  = c1 * int * d_l**2 / (f_obs**2 * (1 + z)**3 )
    L   = c2 * int * d_l**2 * f_obs

    return L
