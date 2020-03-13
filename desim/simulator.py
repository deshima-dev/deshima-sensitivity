__all__ = ["spectrometer_sensitivity"]


# dependent packages
import numpy as np
import pandas as pd


# main functions
def spectrometer_sensitivity(
        F=350.*10.**9.,  # Hz, scalar or vector
        pwv=0.5,  # mm, scalar
        EL=60.,  # deg, scalar
        R=500,  # scalar. R = Q = F/FWHM = F/dF is assumed.
        eta_M1_spill=0.99,  # scalar or vector
        eta_M2_spill=0.90,  # scalar or vector
        eta_wo_spill=0.99,  # scalar or vector. product of cabin spillover loss
        n_wo_mirrors = 4., # number of cabin optics, excluing telescope M1 and M2
        window_AR=True,
        # scalar or vector. product of co spillover, qo filter transmission
        eta_co=0.65,
        # scalar or vector. D2_2V3.pdf, p14:
        # front-to-back ratio 0.93 * reflection efficiency 0.9
        # * matching 0.98 * antenna spillover 0.993
        eta_lens_antenna_rad=0.81,
        # scalar or vector. 'Alejandro Efficiency',
        # from the feedpoint of the antenna to being absorbed in the KID.
        eta_circuit=0.35,
        eta_IBF=0.5,
        KID_excess_noise_factor = 1.1,
        theta_maj=22. * np.pi / 180. / 60. / 60.,  # scalar or vector.
        theta_min=22. * np.pi / 180. / 60. / 60.,  # scalar or vector.
        eta_mb=0.6,  # scalar or vector.
        telescope_diameter=10.,
        Tb_cmb=2.725,
        Tp_amb=273.,
        Tp_cabin=290.,
        Tp_co=4.,
        Tp_chip=0.12,
        snr=5.,
        obs_hours=10.,
        on_source_fraction=0.4*0.9,
        on_off = True
        ):
    """
    Calculate the sensitivity of a spectrometer.

    Parameters
    ------------
    One of the parameters can be a vector of length N.
    In this case the output of the script will be a 2D pandas.DataFrame
    with N rows.
    If all parameters are a scalar, then the output will be
    a 1D pandas DataFrame (or Series?)

    F : scalar or vector
        Frequency of the astronomical signal.
        Unit: Hz
    pwv : scalar or vector
        precipitable water vapour.
        Unit: mm
    EL : scalar or vector
        telescope elevation angle
        Unit: degrees
    R : scalar or vector
        spectral resolving power in F/W_F
        W_F is the 'equivalent bandwidth'
            http://www.astrosurf.com/buil/us/spe2/hresol7.htm
        Unit : None.
    eta_M1_spill : scalar or vector
        spillover efficiency at the telescope primary mirror.
        Unit: None.
    eta_M2_spill : scalar or vector
        spillover efficiency at the telescope secondary mirror.
        Unit: None.
    eta_wo_spill : scalar or vector
        Product of all spillover losses in the warm optics in the cabin.
        Unit: None.
    eta_co : scalar or vector
        Product of following:
            - Cold spillover
            - Cold ohmic losses
            - Filter transmission loss
        Unit: None.
    eta_lens_antenna_rad : scalar or vector
        The loss at chip temperature, *that is not in the circuit.*
        Product of the following:
            - Front-to-back ratio of the lens-antenna on the chip.
            - Reflection efficiency at the surface of the lens.
            - Matching efficiency, due to the mismatch
            - Spillover efficiency of the lens-antenna
        These values can be found in D2_2V3.pdf, p14
        Unit: None.
    eta_circuit : scalar or vector
        The loss at chip temperature, *in the circuit.*
        Unit: None.
    eta_IBF : in-band fraction efficiency
        Fraction of the filter power transmission that is
        within the filter channel bandwidth.
        The rest of the power is cross talk,
        picking up power that is in the bands of neighboring channels.
        This efficiency applies to the coupling to astronomical line signals.
        This efficiency does not apply to the coupling to continuum,
        including the the coupling to the atmosphere for calculating the NEP.
    theta_maj : scalar or vector
        The HPBW along the major axis, assuming a Gaussian beam.
        Unit: radians.
    theta_min : scalar or vector
        The HPBW along the minor axis, assuming a Gaussian beam.
        Unit: radians.
    eta_mb : scalar or vector
        main beam efficiency.
        Note that eta_mb includes the following terms from D2_2V3.pdf
        from Shahab's report.
            - eta_Phi
            - eta_amp
        Because a decrease in these will launch the beam to the sky b
        ut not couple it to the point source.
        (See also FAQ.)
        Unit: None.
    telescope_diameter : float or vector.
        diameter of the telescope.
        Units: m
    Tb_cmb : scalar or vector
        Brightness temperature of the CMB.
        Unit: K
    Tp_amb : scalar or vector
        Physical temperature of the atmosphere and ambient environment
        around the telescope.
        Unit: K
    Tp_cabin : scalar or vector
        Physical temperature of the telescope cabin.
        Unit: K
    Tp_co : scalar or vector
        Physical temperature of the cold optics inside the cryostat.
        Unit: K
    Tp_chip : scalar or vector
        Physical temperature of the chip.
        Unit: K
    snr : scalar or vector
        target signal to noise to be reached. (for calculating the MDLF)
        Unit: None.
    obs_hours :  scalar or vector
        observing hours, including off-source time and the slew
        overhead between on- and off-source.
        Unit : hours
    on_source_fraction : scalar or vector
        Fraction of the time on source (between 0. and 1.)
    on_off: True or False
        If the observation involves on_off chopping, then the SNR degrades by sqrt(2) because
        the signal difference includes the noise twice.

    Returns
    ----------
    The function returns one pandas.DataFrame called 'result',
    which contains the following rows that are all pandas.Series
    F : same as input
    pwv : same as input
    EL : same as input
    eta_atm : atmospheric transmission. Unit: None
    R : same as input
    W_F_spec : equivalent bandwidth within the bandwidth of F/R. Units: Hz
    W_F_cont : equivalent bandwidth of 1 channel including
        the power coupled outside of the filter channel band.
        Units: Hz
    theta_maj : same as input.
    theta_min : same as input.
    eta_a : aperture efficiency (https://deshima.kibe.la/notes/324)
    eta_mb : main beam efficiency
    eta_forward : forward efficiency (https://deshima.kibe.la/notes/324)
    eta_sw : coupling efficiency from a point source to the cryostat window
    eta_window : transmission of the cryostat window
    eta_inst : instrument optical efficiency (https://arxiv.org/abs/1901.06934)
    eta_circuit : same as input
    Tb_sky : Planck brightness temperature of the sky. Units : K
    Tb_M1 : Planckbrightness temperature looking
            into the telescope primary. Units: K
    Tb_M2 :
        Planck brightness temperature looking
        into the telescope secondary,
        including the spillover to the cold sky. Units: K
    Tb_wo : Planck brightness temperature looking into the warm optics.
        Units: K
    Tb_window : Planck brightness temperature looking into the window.
        Units: K
    Tb_co : Planck brightness temperature looking into the cold optis.
        Units: K
    Tb_KID : Planck brightness temperature looking into the filter
        _from_ the KID. Units: K
    Pkid : Power absorbed by the KID. Units: W
    n_ph : Photon occupation number. Units: None.
        (http://adsabs.harvard.edu/abs/1999ASPC..180..671R)
    NEPkid : Noise equivalent power at the KID
        with respect to the absorbed power. Units: W Hz^0.5
    NEPinst : Instrumnet NEP  Units: W Hz^0.5
        (https://arxiv.org/abs/1901.06934).
    NEFD_line : Noise Equivalent Flux Density for
        couploing to a line that is not wider than the
        filter bandwidth. Units: W/m^2/Hz * s^0.5
    NEFD_continuum : Noise Equivalent Flux Density for
        couploing to a countinuum source. Units: W/m^2/Hz * s^0.5
    NEF : Noise Equivalent Flux. Units: W/m^2 * s^0.5
    MDLF : Minimum Detectable Line Flux. Units: W/m^2
    MS : Mapping Speed. Units: (arcmin^2 mJy^-2 h^-1)
    snr : same as input.
    obs_hours : same as input.
    on_source_fraction : same as input.
    on_source_hours : obs_hours * obs_hours*on_source_fraction Units: None.
    equivalent_Trx :
        equivalent receiver noise temperature. Units: K
        at the moment this assumes Rayleigh-Jeans!

    Note
    ----------------
    - The parameters to calculate the window transmission / reflection
    is hard-coded in the function window_trans().

    """

    # Equivalent Bandwidth of 1 channel.
    W_F_cont = F/R/eta_IBF # Used for calculating loading and coupling to a continuum source
    W_F_spec = F/R # Used for calculating coupling to a line source, with a linewidth not wider than the filter channel

    # #############################################################
    # 1. Calculating loading power absorbed by the KID, and the NEP
    # #############################################################

    # .......................................................
    # Efficiencies for calculating sky coupling
    # .......................................................

    # Ohmic loss as a function of frequency, from skin effect scaling
    eta_Al_ohmic = (1.-(1.-eta_Al_ohmic_850)*np.sqrt(F/850.e9))
    eta_M1_ohmic = eta_Al_ohmic
    eta_M2_ohmic = eta_Al_ohmic

    # Collect efficiencies at the same temperature
    eta_M1 = eta_M1_ohmic * eta_M1_spill
    eta_wo = eta_Al_ohmic**n_wo_mirrors * eta_wo_spill
    eta_chip = eta_lens_antenna_rad * eta_circuit

    # Forward efficiency: does/should not include window loss
    # because it is defined as how much power out of the crystat window couples to the cold sky.
    eta_forward = (eta_M1 * eta_M2_ohmic * eta_M2_spill * eta_wo +
                    (1.-eta_M2_spill) * eta_wo)

    # Calcuate eta. scalar/vector depending on F.
    eta_atm = eta_atm_func(F=F, pwv=pwv, EL=EL, R=R)

    # Johnson-Nyquist Power Spectral Density (W/Hz) for the physical temperatures of each stage

    psd_jn_cmb   = johnson_nyquist_psd(F=F, T=Tb_cmb)
    psd_jn_amb   = johnson_nyquist_psd(F=F, T=Tp_amb)
    psd_jn_cabin = johnson_nyquist_psd(F=F, T=Tp_cabin)
    psd_jn_co    = johnson_nyquist_psd(F=F, T=Tp_co)
    psd_jn_chip  = johnson_nyquist_psd(F=F, T=Tp_chip)

    # Optical Chain
    # Sequentially calculate the Power Spectral Density (W/Hz) at each stage.
    # Uses only basic radiation transfer: rad_out = eta*rad_in + (1-eta)*medium

    psd_sky =       rad_trans(rad_in=psd_jn_cmb,   medium=psd_jn_amb,     eta=eta_atm     )
    psd_M1  =       rad_trans(rad_in=psd_sky,      medium=psd_jn_amb,     eta=eta_M1      )
    psd_M2_spill =  rad_trans(rad_in=psd_M1,       medium=psd_sky,        eta=eta_M2_spill) # Note that the spillover of M2 is to the
    psd_M2 =        rad_trans(rad_in=psd_M2_spill, medium=psd_jn_amb,     eta=eta_M2_ohmic)
    psd_wo =        rad_trans(rad_in=psd_M2,       medium=psd_jn_cabin,   eta=eta_wo      )
    [psd_window, eta_window] = (
                    window_trans(F=F, psd_in=psd_wo, psd_cabin=psd_jn_cabin, psd_co=psd_jn_co, window_AR=window_AR))
    psd_co =        rad_trans(rad_in=psd_window,   medium=psd_jn_co,      eta=eta_co      )
    psd_KID =       rad_trans(rad_in=psd_co,       medium=psd_jn_chip,    eta=eta_chip    )  # PSD absorbed by KID

    # Instrument optical efficiency as in JATIS 2019
    # (eta_inst can be calculated only after calculating eta_window)
    eta_inst = eta_chip * eta_co * eta_window

    # Calculating Sky loading, Warm loading and Cold loading individually for reference
    # (Not required for calculating Pkid, but serves as a consistency check.)
    # .................................................................................

    # Sky loading
    psd_KID_sky_1 = psd_sky * eta_M1 * eta_M2_spill * eta_M2_ohmic * eta_wo * eta_inst
    psd_KID_sky_2 = rad_trans(0, psd_sky, eta_M2_spill) * eta_M2_ohmic * eta_wo * eta_inst
    psd_KID_sky = psd_KID_sky_1 + psd_KID_sky_2

    skycoup = psd_KID_sky / psd_sky # To compare with Jochem

    # Warm loading
    psd_KID_warm =  window_trans(F=F,psd_in=
                        rad_trans(
                            rad_trans(
                                rad_trans(
                                    rad_trans(0, psd_jn_amb, eta_M1),
                                0, eta_M2_spill), # sky spillover does not count for warm loading
                            psd_jn_amb, eta_M2_ohmic),
                        psd_jn_cabin, eta_wo),
                    psd_cabin=psd_jn_cabin, psd_co=0, window_AR=window_AR)[0] * eta_co * eta_chip

    # Cold loading
    psd_KID_cold =  rad_trans(
                        rad_trans(
                            window_trans(F=F, psd_in=0., psd_cabin=0., psd_co=psd_jn_co, window_AR=window_AR)[0],
                        psd_jn_co, eta_co),
                    psd_jn_chip, eta_chip)

    # Loadig power absorbed by the KID
    # .............................................

    Pkid = psd_KID * W_F_cont
    Pkid_sky = psd_KID_sky * W_F_cont
    Pkid_warm = psd_KID_warm * W_F_cont
    Pkid_cold = psd_KID_cold * W_F_cont

    # if np.all(Pkid != Pkid_sky + Pkid_warm + Pkid_cold):
    #     print("WARNING: Pkid != Pkid_sky + Pkid_warm + Pkid_cold")

    # Photon + R(ecombination) NEP of the KID
    # .............................................

    NEPkid = photon_NEP_kid(F,Pkid,W_F_cont) * KID_excess_noise_factor

    # Instrument NEP as in JATIS 2019
    # .............................................

    NEPinst = NEPkid / eta_inst  # Instrument NEP

    # ##############################################################
    # 2. Calculating source coupling and sensitivtiy (MDLF and NEFD)
    # ##############################################################

    # Efficiencies
    # .........................................................

    Ag = np.pi * (telescope_diameter/2.)**2.  # Geometric area of the telescope
    omega_mb = np.pi * theta_maj * theta_min / np.log(2) / 4 # Main beam solid angle
    omega_a = omega_mb / eta_mb # beam solid angle
    Ae = (c/F)**2 / omega_a # Effective Aperture (m^2): lambda^2 / omega_a
    eta_a = Ae/Ag # Aperture efficiency

    # Coupling from the "S"ource to outside of "W"indow
    eta_pol = 0.5 # Instrument is single polarization
    eta_sw = eta_pol * eta_atm * eta_a * eta_forward # Source-Window coupling

    # NESP: Noise Equivalent Source Power (an intermediate quantitiy)
    # .........................................................

    NESP = NEPinst / eta_sw # Noise equivalnet source power

    # NEF: Noise Equivalent Flux (an intermediate quantitiy)
    # .........................................................

    # From this point, units change from Hz^-0.5 to t^0.5
    # sqrt(2) is because NEP is defined for 0.5 s integration.

    NEF = NESP / Ag / np.sqrt(2) # Noise equivalent flux

    # If the observation is involves ON-OFF sky subtraction,
    # Subtraction of two noisy sources results in sqrt(2) increase in noise.

    if on_off == True:
        NEF = np.sqrt(2) * NEF

    # MDLF (Minimum Detectable Line Flux)
    # .........................................................

    # Note that eta_IBF does not matter for MDLF because it is flux.

    MDLF = NEF * snr / np.sqrt(obs_hours*on_source_fraction*60.*60.)

    # NEFD (Noise Equivalent Flux Density)
    # .........................................................

    spectral_NEFD  = NEF / W_F_spec
    continuum_NEFD = NEF / W_F_cont # = spectral_NEFD * eta_IBF < spectral_NEFD

    # Mapping Speed (line, 1 channel) (arcmin^2 mJy^-2 h^-1)
    # .........................................................

    MS = 60.*60.*1.* omega_mb*(180./np.pi*60.)**2. /(np.sqrt(2)*spectral_NEFD*1e29)**2.

    # Equivalent Trx
    # .........................................................

    Trx = NEPinst/k/np.sqrt(2*W_F_cont) - T_from_psd(F, psd_wo)  # assumes RJ!

    # ############################################
    # 3. Output results as Pandas DataFrame
    # ############################################

    result = pd.concat([
        pd.Series(F, name='F'),
        pd.Series(pwv, name='PWV'),
        pd.Series(EL, name='EL'),
        pd.Series(eta_atm, name='eta_atm'),
        pd.Series(R, name='R'),
        pd.Series(W_F_spec, name='W_F_spec'),
        pd.Series(W_F_cont, name='W_F_cont'),
        pd.Series(theta_maj, name='theta_maj'),
        pd.Series(theta_min, name='theta_min'),
        pd.Series(eta_a, name='eta_a'),
        pd.Series(eta_mb, name='eta_mb'),
        pd.Series(eta_forward, name='eta_forward'),
        pd.Series(eta_sw, name='eta_sw'),
        pd.Series(eta_window, name='eta_window'),
        pd.Series(eta_inst, name='eta_inst'),
        pd.Series(eta_circuit, name='eta_circuit'),
        pd.Series(T_from_psd(F, psd_sky), name='Tb_sky'),
        pd.Series(T_from_psd(F, psd_M1), name='Tb_M1'),
        pd.Series(T_from_psd(F, psd_M2), name='Tb_M2'),
        pd.Series(T_from_psd(F, psd_wo), name='Tb_wo'),
        pd.Series(T_from_psd(F, psd_window), name='Tb_window'),
        pd.Series(T_from_psd(F, psd_co), name='Tb_co'),
        pd.Series(T_from_psd(F, psd_KID), name='Tb_KID'),
        pd.Series(psd_KID, name='psd_KID'),
        pd.Series(Pkid, name='Pkid'),
        pd.Series(Pkid_sky, name='Pkid_sky'),
        pd.Series(Pkid_warm, name='Pkid_warm'),
        pd.Series(Pkid_cold, name='Pkid_cold'),
        pd.Series(Pkid/(W_F_cont * h * F), name='n_ph'),
        pd.Series(NEPkid, name='NEPkid'),
        pd.Series(NEPinst, name='NEPinst'),
        pd.Series(spectral_NEFD, name='NEFD_line'),
        pd.Series(continuum_NEFD, name='NEFD_continuum'),
        pd.Series(NEF, name='NEF'),
        pd.Series(MDLF, name='MDLF'),
        pd.Series(MS, name='MS'),
        pd.Series(snr, name='snr'),
        pd.Series(obs_hours, name='obs_hours'),
        pd.Series(on_source_fraction, name='on_source_fraction'),
        pd.Series(obs_hours*on_source_fraction, name='on_source_hours'),
        pd.Series(Trx, name='equivalent_Trx'),
        pd.Series(skycoup, name='skycoup'),
        pd.Series(eta_Al_ohmic, name='eta_Al_ohmic'),
        # pd.Series(Pkid_warm_jochem, name='Pkid_warm_jochem')
        ], axis=1
        )

    # Turn Scalar values into vectors
    result = result.fillna(method='ffill')

    return result
