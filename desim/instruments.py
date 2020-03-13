# dependent packages
import numpy as np
from .physics import rad_trans, h, e, c


# constants
Delta_Al = 188. * 10**-6 * e  # gap energy of Al
eta_pb = 0.4  # Pair breaking efficiency
eta_Al_ohmic_850 = 0.9975 # Ohmic loss of an Al surface at 850 GHz. Shitov et al., ISSTT2008. https://www.nrao.edu/meetings/isstt/papers/2008/2008263266.pdf


# main functions
def D2HPBW(F):
    HPBW = 29.*240./(F/1e9) * np.pi / 180. / 60. / 60.
    return HPBW


def eta_mb_ruze(F, LFlimit, sigma):
    '''F in Hz, LFlimit is the eta_mb at => 0 Hz, sigma in m'''
    eta_mb = LFlimit* np.exp(- (4.*np.pi* sigma * F/c)**2. )
    return eta_mb


def photon_NEP_kid(
        F,
        Pkid,
        W_F
        ):
    """
    NEP of the KID, with respect to the absorbed power.

    Parameters
    -----------
    F:  Frequency of the signal responsible for loading.
        Unit: Hz
    Pkid: Power absorbed by the KID.
        Unit: W
    W_F: detection bandwidth, with respect to the power that sets the loading.
        Unit: Hz

    Note
    --------
    Pkid/(W_F * h * F) gives the occupation number.
    """
    # photon_term = 2 * Pkid * (h*F + Pkid/W_F)
    poisson_term = 2 * Pkid * h * F
    bunching_term = 2 * Pkid * Pkid / W_F
    r_term = 4 * Delta_Al * Pkid / eta_pb
    NEPkid = np.sqrt(poisson_term + bunching_term + r_term)
    return NEPkid


def window_trans(
        F,
        psd_in,
        psd_cabin,
        psd_co,
        thickness=8.e-3,    # in m
        tandelta=4.805e-4,  # tan delta, measured Biorad
        tan2delta=1.e-8,
        neffHDPE=1.52,
        window_AR = True
        ):
    """
    Calculates the window transmission.

    Parameters
    ----------
    F : scalar or vector.
        Frequency
        Units: Hz
    psd_in : scalar or vector.
        PSD of the incoming signal.
        Units : W / Hz
    psd_cabin : scalar or vector.
        Johnson-Nyquist PSD of telescope cabin temperature.
        Units : W / Hz
    psd_co : scalar or vector.
        Johnson-Nyquist PSD of cold-optics temperature.
        Units : W / Hz
    thickness: scalar or vector.
        thickness of the HDPE window.
        Units: m
    tandelta, tan2delta : scalar
        values from Stephen.
            "# 2.893e-8 %% tan delta, measured Biorat. I use 1e-8 as this fits
            the tail of the data better"
    neffHDPE : scalar
        refractive index of HDPE. set to 1 to remove reflections.
        Units : None.

    Returns
    -------
    psd_after_2nd_refl : scalar or vector
        PSD looking into the window from the cold optics
    eta_window : scalar or vector
        transmission of the window

    """
    # Parameters to calcualte the window (HDPE), data from Stephen
    # reflection. ((1-neffHDPE)/(1+neffHDPE))^2. Set to 0 for Ar coated.

    if window_AR is True:
        HDPErefl = 0.
    else:
        HDPErefl = ((1-neffHDPE)/(1+neffHDPE))**2

    eta_HDPE = np.exp(-thickness * 2 * np.pi * neffHDPE *
                      (tandelta * F / c + tan2delta * (F / c)**2))
    # most of the reflected power sees the cold.
    psd_after_1st_refl = rad_trans(psd_in, psd_co, 1.-HDPErefl)
    psd_before_2nd_refl = rad_trans(psd_after_1st_refl, psd_cabin, eta_HDPE)
    # the reflected power sees the cold.
    psd_after_2nd_refl = rad_trans(psd_before_2nd_refl, psd_co, 1.-HDPErefl)

    eta_window = (1.-HDPErefl)**2 * eta_HDPE

    return psd_after_2nd_refl, eta_window
