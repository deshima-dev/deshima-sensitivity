import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import copy
import pandas as pd
from scipy.interpolate import interp2d
from matplotlib.backends.backend_pdf import PdfPages

h = 6.62607004 * 10**-34
k = 1.38064852 * 10**-23
e = 1.60217662 * 10**-19 # electron charge
c = 299792458.
Delta_Al = 188 * 10**-6 * e # gap energy of Al
eta_pb = 0.4

def deshima_sensitivity(
    F = 350.*10.**9.,
    Q = 500,
    eta_atm = 0.9,
    eta_M1_spill = 0.99,
    eta_M2_spill = 0.90,
    eta_wo = 0.99, # product of all cabin loss (ohmic and spillover)\
    n_wo_mirrors = 4,
    eta_co = 0.65, # product of co spillover, qo filter transmission
    eta_lens_antenna_rad = 0.81, # D2_2V3.pdf, p14: front-to-back ratio 0.93 * reflection efficiency 0.9 * matching 0.98 * antenna spillover 0.993
    eta_circuit = 0.45, # 'Alejandro Efficiency', from the feedpoint of the antenna to being absorbed in the KID.
    theta_maj = 22. * np.pi / 180. / 60. / 60.,
    theta_min = 22. * np.pi / 180. / 60. / 60.,
    eta_mb = 0.6,
    Tb_univ = 2.725,
    Tp_amb = 273.,
    Tp_cabin = 290.,
    Tp_co = 4.,
    Tp_chip = 0.12
    ):

    """
    "_co" is everything at 4K. (cold optics temperature)
    "_chip" is everything at 120 mK.  (chip temperature)
    Note that "eta_lens_antenna_rad" is part of "eta_chip" (even though Shahab calculated it.)
    """

    # Ohmic loss
    Al_refl_ohmic_loss = 0.9975
    eta_M1_ohmic = Al_refl_ohmic_loss
    eta_M2_ohmic = Al_refl_ohmic_loss

    # Collect efficiencies at the same temperature
    eta_M1 =  eta_M1_ohmic * eta_M1_spill
    eta_chip = eta_lens_antenna_rad * eta_circuit

    # Forward efficiency
    eta_forward = eta_M1*eta_M2_ohmic * eta_M2_spill * eta_wo + (1-eta_M2_spill)*eta_wo # does/should not include window loss

    # Equivalent Bandwidth of 1 channel, assuming Lorentzian
    # Note that this calculates the coupling to a continuum source.
    W_F = 0.5*np.pi*F/Q

    # Calculate Johnson-Nyquist power for the four temperatures
    Pjn_univ    = jn(F=F,T=Tb_univ)
    Pjn_amb     = jn(F=F,T=Tp_amb)
    Pjn_cabin   = jn(F=F,T=Tp_cabin)
    Pjn_co      = jn(F=F,T=Tp_co)
    Pjn_chip    = jn(F=F,T=Tp_chip)

    # Power density (W/Hz) at different stages
    PF_sky       = rad_trans(Pjn_univ,   Pjn_amb,   eta_atm)
    PF_M1        = rad_trans(PF_sky,      Pjn_amb,   eta_M1)
    PF_M2_spill  = rad_trans(PF_M1,       PF_sky,     eta_M2_spill)
    PF_M2        = rad_trans(PF_M2_spill, Pjn_amb,   eta_M2_ohmic)
    PF_wo        = rad_trans(PF_M2,       Pjn_cabin, eta_wo)
    [PF_window, eta_window] = window_trans(PF_wo,    Pjn_cabin,  Pjn_co, F)
    PF_co        = rad_trans(PF_window,   Pjn_co,    eta_co)
    PF_KID       = rad_trans(PF_co,       Pjn_chip,  eta_chip) # Power density _absorbed_ by the KID

    # Instrument optical efficiency
    eta_inst = eta_chip * eta_co * eta_window

    # Reverse calculate Tb from P at each stage using Planck.
    result = {
        "Tb_sky": T_CW(F,PF_sky),
        "Tb_M1" : T_CW(F,PF_M1),
        "Tb_M2" : T_CW(F,PF_M2),
        "Tb_wo" : T_CW(F,PF_wo),
        "Tb_window" : T_CW(F,PF_window),
        "W_F": W_F,
        "eta_forward":eta_forward,
        "eta_window":eta_window
    }

    # Telescope aperture efficiency
    result['eta_a'] = aperture_efficiency(
        theta_maj = theta_maj,
        theta_min = theta_min,
        eta_mb = eta_mb,
        F = F
        )

    # Source-window coupling efficiency
    # eta_pol = 0.5 is included in this function.
    result['eta_sw'] = eta_source_window(
        eta_a       = result['eta_a'],
        eta_atm     = eta_atm,
        eta_forward = eta_forward
        )

    # Photon + R(ecombination) NEP
    P_KID = PF_KID * W_F
    photon_term = 2 * P_KID * (h*F + P_KID/W_F )
    r_term = 4 * Delta_Al * P_KID / eta_pb
    NEP_KID = np.sqrt(photon_term + r_term) # KID NEP
    NEP_inst = NEP_KID / eta_inst # Instrument NEP

    result.update({
        "eta_chip":eta_chip,
        "eta_circuit":eta_circuit,
        "eta_inst":eta_inst,
        "Tb_co": T_CW(F,PF_co),
        "Tb_KID":T_CW(F,PF_KID),
        "P_KID":P_KID,
        "NEP_KID":NEP_KID,
        "NEP_inst":NEP_inst
    })

    result['NEFD'] = NEFD(result['NEP_inst'],
         eta_source_window = result["eta_sw"],
         F = F,
         Q = Q
        )

    result['NEF'] = result['NEFD'] * W_F
    result['equivalent Trx'] = result['NEP_inst']/k/np.sqrt(2*W_F) - result['Tb_wo']

    return result


def eta_atm_interp(eta_atm_dataframe):
    """
    Returns a function that interpolates atmospheric transmission data
    downloaded from ALMA (https://almascience.eso.org/about-alma/atmosphere-model)
    The returned function has the form of
    eta = func(F [GHz], pwv [mm])

    Example:
        % read csv file with pandas (in e.g., Jupyter)
        eta_atm_df = pd.read_csv("atm.csv",skiprows=4,delim_whitespace=True,header=0)
        % make function from pandas file
        etafun = desim.eta_atm_interp(eta_atm_df)
    """
    x = np.array(list(eta_atm_dataframe)[1:]).astype(np.float)
    y = eta_atm_dataframe['F'].values
    z = eta_atm_dataframe.iloc[:,1:].values
    func = interp2d(x,y,z,kind='cubic')
    return func

def rad_trans(T_bkg, T_mdm, eta):
    """Radiation transfer through a semi-transparent medium"""
    T_b = eta * T_bkg + (1 - eta) * T_mdm
    return T_b

def window_trans(PFin, Pjn_cabin, Pjn_co, F):
    # Parameters to calcualte the window (HDPE), data from Stephen
    thickness   = 8.e-3    # in m
    tandelta    = 4.805e-4 # tan delta, measured Biorad
    tan2delta   = 1.e-8     # 2.893e-8; %% tan delta, measured Biorat. I use 1e-8 as this fits the tail of the data better
    neffHDPE    = 1.52     # for window reflection calculation and loss calculation. set to 1 to remove reflections
    HDPErefl = ((1-neffHDPE)/(1+neffHDPE))**2;    #reflection. ((1-neffHDPE)/(1+neffHDPE))^2. Set to 0 for Ar coated.
    eta_HDPE = np.exp(-thickness*2*np.pi*neffHDPE*(tandelta*F/c + tan2delta*(F/c)**2))

    PF_after_1st_refl  = rad_trans(PFin, Pjn_co, 1.-HDPErefl) # most of the reflected power sees the cold.
    PF_before_2nd_refl = rad_trans(PF_after_1st_refl, Pjn_cabin, eta_HDPE)
    PF_after_2nd_refl  = rad_trans(PF_before_2nd_refl, Pjn_co, 1.-HDPErefl) # the reflected power sees the cold.

    eta_window = (1.-HDPErefl)**2 * eta_HDPE

    # print((1.-HDPErefl))
    # print(eta_HDPE)

    return PF_after_2nd_refl, eta_window

def nph(F,T):
    """Photon occupation number of Bose-Einstein Statistics"""
    n = 1./(np.exp(h*F/(k*T))-1.)
    return n

def jn(F,T):
    """
    Johnson-Nyquist power. Returns power per unit bandwidth, W/Hz.
    Don't forget to multiply with bandwidth to caculate the total power in W.
    """
    P_F = h*F*nph(F,T)
    return P_F

def T_CW(F,P_F):
    """Callen-Welton temperature. P_F is power per unit bandwidth, W/Hz"""
    T = h*F/(k*np.log(h*F/P_F+1.))
    return T

def T_RJ(F,P_F):
    """Reileigh-Jeans temperature. P_F is power per unit bandwidth, W/Hz"""
    T = P_F / k
    return T

def aperture_efficiency(
    theta_maj = 31.4 * np.pi / 180. / 60. / 60.,
    theta_min = 22.8 * np.pi / 180. / 60. / 60.,
    eta_mb = 0.35,
    c = 299792458.,
    F = 350. * 10**9,
    telescope_diameter = 10.
    ):

    omega_mb = np.pi * theta_maj * theta_min / np.log(2) /4
    omega_a = omega_mb / eta_mb
    lmd = c/F
    Ae = lmd**2 / omega_a
    Ag = np.pi * (telescope_diameter/2.)**2
    eta_a = Ae/Ag
    return eta_a

def eta_source_window(
    eta_a = 0.171,
    eta_pol = 0.5,
    eta_atm = 0.9,
    eta_forward = 0.94
    ):
    """
    Optical efficiency from an astronomical point source to the cryostat window.
    Factor 2 loss in polarization is included here.
    """
    eta_source_window_ = eta_pol * eta_atm * eta_a * eta_forward
    return eta_source_window_

def NEFD(NEP_inst,
         eta_source_window,
         F = 350.*10.**9.,
         Q = 300,
         diameter = 10.,
        ):
    NESP = NEP_inst / eta_source_window # noise equivalent source power
    radius = diameter / 2.
    Ag = np.pi * radius**2. # physical diameter of the telescope
    NEF = NESP /Ag / np.sqrt(2) # noise equivalent flux; sqrt(2) is because NEP is defined for 0.5 s integration.
    W_F = 0.5*np.pi*F/Q
    NEFD_ = NEF / W_F
    return NEFD_
