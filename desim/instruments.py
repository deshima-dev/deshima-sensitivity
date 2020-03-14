# standard library
from typing import List, Union, Tuple


# dependent packages
import numpy as np
from .physics import c, e, h, rad_trans


# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]


# constants
Delta_Al = 188.0 * 10 ** -6 * e  # gap energy of Al
eta_pb = 0.4  # Pair breaking efficiency
eta_Al_ohmic_850 = 0.9975  # Ohmic loss of an Al surface at 850 GHz.
# Shitov+, ISSTT2008. https://www.nrao.edu/meetings/isstt/papers/2008/2008263266.pdf


# main functions
def D2HPBW(F: ArrayLike) -> ArrayLike:
    """Get half-power beam width of DESHIMA 2.0 at given frequency (frequencies).

    Parameters
    ----------
    F
        Frequency. Units: Hz.

    Returns
    -------
    hpbw
        Half-power beam width. Units: radian.

    """
    return 29.0 * 240.0 / (F / 1e9) * np.pi / 180.0 / 60.0 / 60.0


def eta_mb_ruze(F: ArrayLike, LFlimit: float, sigma: float) -> ArrayLike:
    """Get main-beam efficiency by Ruze's equation.

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    LFlimit
        Main-beam efficiency at 0 Hz.
    sigma
        Surface error. Units: m.

    Returns
    -------
    eta_mb
        Main-beam efficiency. Units: None.

    """
    return LFlimit * np.exp(-((4.0 * np.pi * sigma * F / c) ** 2.0))


def photon_NEP_kid(F: ArrayLike, Pkid: ArrayLike, W_F: ArrayLike) -> ArrayLike:
    """NEP of the KID, with respect to the absorbed power.

    Parameters
    -----------
    F
        Frequency of the signal responsible for loading. Units: Hz.
    Pkid
        Power absorbed by the KID. Units: W.
    W_F
        Detection bandwidth, with respect to the power that sets the loading. Units: Hz.

    Returns
    -------
    NEP_kid
        Noise-equivalent power of the KID.

    Notes
    -----
    Pkid/(W_F * h * F) gives the occupation number.

    """
    # photon_term = 2 * Pkid * (h*F + Pkid/W_F)
    poisson_term = 2 * Pkid * h * F
    bunching_term = 2 * Pkid * Pkid / W_F
    r_term = 4 * Delta_Al * Pkid / eta_pb
    return np.sqrt(poisson_term + bunching_term + r_term)


def window_trans(
    F: ArrayLike,
    psd_in: ArrayLike,
    psd_cabin: ArrayLike,
    psd_co: ArrayLike,
    thickness: ArrayLike = 8.0e-3,
    tandelta: float = 4.805e-4,
    tan2delta: float = 1.0e-8,
    neffHDPE: float = 1.52,
    window_AR: bool = True,
) -> Tuple[ArrayLike, ArrayLike]:
    """Calculates the window transmission.

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    psd_in
        PSD of the incoming signal. Units : W / Hz.
    psd_cabin
        Johnson-Nyquist PSD of telescope cabin temperature. Units : W / Hz.
    psd_co
        Johnson-Nyquist PSD of cold-optics temperature. Units : W / Hz.
    thickness
        Thickness of the HDPE window. Units: m.
    tandelta
        Values from Stephen. "# 2.893e-8 %% tan delta, measured Biorat.
        I use 1e-8 as this fits the tail of the data better".
    tan2delta
        Values from Stephen. "# 2.893e-8 %% tan delta, measured Biorat.
        I use 1e-8 as this fits the tail of the data better".
    neffHDPE
        Refractive index of HDPE. Set to 1 to remove reflections. Units : None.
    window_AR
        Whether the window is supposed to be coated by Ar (True) or not (False).


    Returns
    -------
    psd_after_2nd_refl
        PSD looking into the window from the cold optics.
    eta_window
        Transmission of the window. Units: None.

    """
    # Parameters to calcualte the window (HDPE), data from Stephen
    # reflection. ((1-neffHDPE)/(1+neffHDPE))^2. Set to 0 for Ar coated.

    if window_AR:
        HDPErefl = 0.0
    else:
        HDPErefl = ((1 - neffHDPE) / (1 + neffHDPE)) ** 2

    eta_HDPE = np.exp(
        -thickness
        * 2
        * np.pi
        * neffHDPE
        * (tandelta * F / c + tan2delta * (F / c) ** 2)
    )

    # most of the reflected power sees the cold.
    psd_after_1st_refl = rad_trans(psd_in, psd_co, 1.0 - HDPErefl)
    psd_before_2nd_refl = rad_trans(psd_after_1st_refl, psd_cabin, eta_HDPE)
    # the reflected power sees the cold.
    psd_after_2nd_refl = rad_trans(psd_before_2nd_refl, psd_co, 1.0 - HDPErefl)

    eta_window = (1.0 - HDPErefl) ** 2 * eta_HDPE

    return psd_after_2nd_refl, eta_window
