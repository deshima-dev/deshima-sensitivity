# standard library
import sys


# dependent packages
import numpy as np


# constants
h = 6.62607004 * 10**-34  # Planck constant
k = 1.38064852 * 10**-23  # Boltzmann constant
e = 1.60217662 * 10**-19  # electron charge
c = 299792458.  # velocity of light


# main functions
def rad_trans(rad_in, medium, eta):
    """
    Calculates radiation transfer through a semi-transparent medium.
    One can also use the same function for
    Johnson-Nyquist PSD (power spectral density) instead of temperature.

    Parameters
    ----------
    rad_in : scalar or vector
        brightness temperature (or PSD) of the input
        Units: K (or W/Hz)
    medium : scalar
        brightness temperature (or PSD) of the lossy medium
        Units: K (or W/Hz)
    eta : scalar or vector
        transmission of the lossy medium
        Units: K (or W/Hz)

    Returns
    -------
    rad_out : brightness temperature (or PSD) of the output

    """
    rad_out = eta * rad_in + (1 - eta) * medium
    return rad_out


def T_from_psd(
        F,
        psd,
        method='Planck'
        ):
    """
    Calculate Planck temperature from the PSD a single frequency,
    or an array of frequencies.

    Parameters
    ----------
    F : scalar or vector.
        Frequency
        Units: Hz
    psd : scalar or vector
        Power Spectral Density.
        Units : W / Hz
    method: optional, sring.
        default: 'Planck'
        option: 'Rayleigh-Jeans'
    Returns
    --------
    T : scalar or vector.
        Planck temperature.
        Units : K

    """
    if method == 'Planck':
        T = h*F/(k*np.log(h*F/psd+1.))
    elif method is 'Rayleigh-Jeans':
        T = psd / k
    else:
        sys.exit("Error: Method should be Planck or Rayleigh-Jeans.")

    return T


def johnson_nyquist_psd(F, T):
    """
    Johnson-Nyquist power spectral density.
    Don't forget to multiply with bandwidth to caculate the total power in W.

    Parameters
    ----------
    F : scalar or vector.
        Frequency
        Units: Hz
    T : scalar or vector.
        temperature
        Units: K

    Returns
    --------
    psd : scalar or vector
        Power Spectral Density.
        Units : W / Hz
    """
    psd = h*F*nph(F, T)
    return psd


# helper functions
def nph(F, T):
    """
    Photon occupation number of Bose-Einstein Statistics.
    If it is not single temperature, use nph = Pkid/(W_F * h * F)

    Parameters
    ----------
    F : scalar or vector.
        Frequency
        Units: Hz
    T : scalar or vector.
        temperature
        Units: K

    Returns
    --------
    n : scalar or vector
        photon occupation number
        Units : None.
    """
    n = 1./(np.exp(h*F/(k*T))-1.)
    return n
