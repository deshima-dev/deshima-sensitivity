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
    Q = 300,
    eta_atm = 0.9,
    eta_M1_spill = 0.98,
    eta_M1_ohmic = 0.99,
    eta_M2_ohmic = 0.99,
    eta_M2_spill = 0.96,
    eta_wo_spill = 0.99,
    eta_wo_ohmic = 0.99**2,
    eta_co_spill = 0.73,
    eta_fl       = 0.92,
    eta_window   = 0.92,
    eta_co_ohmic = 0.99**2,
    eta_qof      = 0.4,
    eta_lens_antenna_rad = 0.8,
    eta_fb = 0.08,
    theta_maj = 31.4 * np.pi / 180. / 60. / 60.,
    theta_min = 22.8 * np.pi / 180. / 60. / 60.,
    eta_mb = 0.35,
    Tp_amb = 273.,
    Tb_univ = 0.,
    Tp_co = 4.,
    T_RX = False # If T_RX is not False, then the script will calculate for coherent RX.
    ):

    # Bandwidth of 1 channel, assuming Lorentzian
    W_nu = 0.5*np.pi*F/Q

    # Physical temperatures
    Tp_cabin = Tp_amb

    eta_forward = eta_M1_spill * eta_M1_ohmic * eta_M2_ohmic * eta_wo_ohmic * eta_wo_spill

    # brigtness temperatures of different stages
    Tb_sky       = rad_trans(Tb_univ,     Tp_amb,   eta_atm)
    Tb_M1        = rad_trans(Tb_sky,      Tp_amb,   eta_M1_ohmic*eta_M1_spill)
    Tb_M2_ohmic  = rad_trans(Tb_M1,       Tp_amb,   eta_M2_ohmic)
    Tb_M2        = rad_trans(Tb_M2_ohmic, Tb_sky,   eta_M2_spill)
    Tb_wo        = rad_trans(Tb_M2,       Tp_cabin, eta_wo_spill*eta_wo_ohmic)

    result = {
        "W_nu": W_nu,
        "eta_forward":eta_forward,
        "Tb_sky": Tb_sky,
        "Tb_M1":Tb_M1,
        "Tb_M2":Tb_M2,
        "Tb_wo":Tb_wo,
    }

    result['eta_a'] = aperture_efficiency(
        theta_maj = theta_maj,
        theta_min = theta_min,
        eta_mb = eta_mb,
        F = F
        )

    result['eta_sw'] = eta_source_window(
        eta_a = result['eta_a'],
        eta_atm = eta_atm,
        eta_forward =eta_forward
        )

    if T_RX != False: # Coherent (Heterodyne)
        result['T_sys']  = Tb_wo + T_RX
        result['NEP_inst'] = np.sqrt(2)*k*np.sqrt(W_nu)*result['T_sys']
    else: # Incoherent (DESHIMA)
        # More Efficiencies
        eta_chip = eta_lens_antenna_rad * eta_fb
        eta_co_total = eta_co_ohmic * eta_co_spill * eta_qof * eta_fl * eta_window
        eta_inst = eta_chip * eta_qof * eta_co_ohmic * eta_co_spill * eta_fl * eta_window

        # 4K loading is calculated separately, see below.
        # This is because 4K and 350 GHz is not Rayleigh Jeans.
        Tb_co        = rad_trans(Tb_wo,       0,
                                  eta_co_ohmic*eta_co_spill*eta_qof*eta_fl*eta_window)
        Tb_KID       = rad_trans(Tb_co,       0,        eta_chip)

        # Power
        P_4K_KID = jn(F=F,T=4) * (1- eta_co_total ) * eta_chip * W_nu  # calculate 4K loading separately
        P_amb_KID = k*Tb_KID*W_nu
        P_KID = P_amb_KID  # + P_4K_KID : See note below.

        # Total coupling efficiency from all sources at $T_{amb}$ to KID
        eta_amb_KID = Tb_KID/Tp_amb

        # Photon + R(ecombination) NEP
        photon_term = 2 * P_KID * h * F * (1 + eta_amb_KID * nph(F=F, T=Tp_amb) )
        r_term = 4 * Delta_Al * P_KID / eta_pb
        NEP_KID = np.sqrt(photon_term + r_term) # KID NEP
        NEP_inst = NEP_KID / eta_inst # Instrument NEP

        result.update({
            "eta_chip":eta_chip,
            "eta_inst":eta_inst,
            "eta_amb_KID":eta_amb_KID,
            "Tb_co":Tb_co,
            "Tb_KID":Tb_KID,
            "P_4K_KID":P_4K_KID,
            "P_KID":P_KID,
            "NEP_KID":NEP_KID,
            "NEP_inst":NEP_inst
        })

    result['NEFD'] = NEFD(result['NEP_inst'],
         eta_source_window = result["eta_sw"],
         F = F,
         Q = Q
        )
    result['NEF'] = result['NEFD'] * W_nu
    result['equivalent Trx'] = result['NEP_inst']/k/np.sqrt(2*W_nu) - result['Tb_wo']

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

def nph(F,T):
    """Photon occupation number of Bose-Einstein Statistics"""
    n = 1./(np.exp(h*F/(k*T))-1.)
    return n

def jn(F,T):
    """Johnson-Nyquist power"""
    jn = h*F*nph(F,T)
    return jn

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
    W_nu = 0.5*np.pi*F/Q
    NEFD_ = NEF / W_nu
    return NEFD_

def etainstForm(dict_input, eta_inst):
    dict_new = copy.deepcopy(dict_input)
    dict_new.update(
        eta_co_spill = 1,
        eta_fl       = 1,
        eta_window   = 1,
        eta_co_ohmic = 1,
        eta_qof      = 1,
        eta_lens_antenna_rad = 1,
        eta_fb = eta_inst
    )
    return dict_new
