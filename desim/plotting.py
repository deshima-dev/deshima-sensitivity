__all__ = ["MDLF_simple", "MS_simple", "PlotD2HPBW"]


# standard library
import base64


# dependent packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML
from .deshima import D2HPBW
from .telescope import eta_mb_ruze
from .simulator import spectrometer_sensitivity


# main functions
def MDLF_simple(
        F,
        pwv = 0.5, # Precipitable Water Vapor in mm
        EL = 60., # Elevation angle in degrees
        snr = 5., # Target S/N of the detection
        obs_hours = 8. # Total hours of observation, including ON-OFF and calibration overhead
        ):

    # Main beam efficiency of ASTE
    eta_mb = eta_mb_ruze(F=F,LFlimit=0.805,sigma=37e-6) * 0.9 # see specs, 0.9 is from EM, ruze is from ASTE

    D2goal_input ={
        'F' : F, # Frequency in GHz
        'pwv':pwv, # Precipitable Water Vapor in mm
        'EL':EL, # Elevation angle in degrees
        'theta_maj' : D2HPBW(F), # Half power beam width (major axis)
        'theta_min' : D2HPBW(F), # Half power beam width (minor axis)
        'eta_mb' : eta_mb, # Main beam efficiency
        'snr' : snr, # Target S/N of the detection
        'obs_hours' :obs_hours, # Total hours of observation, including ON-OFF and calibration overhead
        'on_source_fraction':0.4*0.9 # ON-OFF 40%, calibration overhead of 10%
    }

    D2goal = spectrometer_sensitivity(**D2goal_input)

    D2baseline_input = {
        'F' : F,
        'pwv':pwv,
        'EL':EL,
        'eta_circuit' : 0.32 * 0.5, # <= eta_inst Goal 16%, Baseline 8%
        'eta_IBF' : 0.4, # <= Goal 0.6
        'KID_excess_noise_factor' : 1.2, # Goal 1.1
        'theta_maj' : D2HPBW(F), # Half power beam width (major axis)
        'theta_min' : D2HPBW(F), # Half power beam width (minor axis)
        'eta_mb' : eta_mb,
        'snr' : snr,
        'obs_hours' :obs_hours,
        'on_source_fraction':0.3*0.8 # <= Goal 0.4*0.9
    }

    D2baseline = spectrometer_sensitivity(**D2baseline_input)

    ### Plot-------------------

    fig, ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(D2baseline['F']/1e9,D2baseline['MDLF'],'--',linewidth=1,color='b',alpha=1,label='Baseline')
    ax.plot(D2goal['F']/1e9,D2goal['MDLF'],linewidth=1,color='b',alpha=1,label='Goal')
    ax.fill_between(D2baseline['F']/1e9,D2baseline['MDLF'],D2goal['MDLF'],color='b',alpha=0.2)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Minimum Detectable Line Flux ($\mathrm{W\ m^{-2}}$)")
    ax.set_yscale('log')
    ax.set_xlim(200,460)
    ax.set_ylim([10**-20,10**-17])
    ax.tick_params(direction='in',which='both')
    ax.grid(True)
    ax.set_title("$R="+str(int(D2goal['R'][0]))+", snr=" + str(D2goal['snr'][0]) + ',\ t_\mathrm{obs}='
                 +str(D2goal['obs_hours'][0])
                 +'\mathrm{h}$ (incl. overhead), PWV=' + str(D2goal['PWV'][0]) + "mm, EL="+str(int(D2goal['EL'][0]))+'deg',
                 fontsize=12)
    ax.legend()
    plt.tight_layout()

    # Create download link
    #................................

    df_download = D2goal[['F','MDLF']]
    df_download = df_download.rename(columns={'MDLF':'MDLF (goal)'})
    df_download = df_download.join(D2baseline[['MDLF']])
    df_download = df_download.rename(columns={'MDLF':'MDLF (baseline)'})

    return create_download_link(df_download,filename='MDLF.csv')


def MS_simple(
        F,
        pwv = 0.5, # Precipitable Water Vapor in mm
        EL = 60., # Elevation angle in degrees
        ):

    # Main beam efficiency of ASTE
    eta_mb = eta_mb_ruze(F=F,LFlimit=0.805,sigma=37e-6) * 0.9 # see specs, 0.9 is from EM, ruze is from ASTE

    D2goal_input ={
        'F' : F, # Frequency in GHz
        'pwv':pwv, # Precipitable Water Vapor in mm
        'EL':EL, # Elevation angle in degrees
        'theta_maj' : D2HPBW(F), # Half power beam width (major axis)
        'theta_min' : D2HPBW(F), # Half power beam width (minor axis)
        'eta_mb' : eta_mb, # Main beam efficiency
        'on_off':False
    }

    D2goal = spectrometer_sensitivity(**D2goal_input)

    D2baseline_input = {
        'F' : F,
        'pwv':pwv,
        'EL':EL,
        'eta_circuit' : 0.32 * 0.5, # <= eta_inst Goal 16%, Baseline 8%
        'eta_IBF' : 0.4, # <= Goal 0.6
        'KID_excess_noise_factor' : 1.2, # Goal 1.1
        'theta_maj' : D2HPBW(F), # Half power beam width (major axis)
        'theta_min' : D2HPBW(F), # Half power beam width (minor axis)
        'eta_mb' : eta_mb,
        'on_off':False
    }

    D2baseline = spectrometer_sensitivity(**D2baseline_input)

    ### Plot-------------------

    fig, ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(D2baseline['F']/1e9,D2baseline['MS'],'--',linewidth=1,color='b',alpha=1,label='Baseline')
    ax.plot(D2goal['F']/1e9,D2goal['MS'],linewidth=1,color='b',alpha=1,label='Goal')
    ax.fill_between(D2baseline['F']/1e9,D2baseline['MS'],D2goal['MS'],color='b',alpha=0.2)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Mapping Speed ($\mathrm{arcmin^2\ mJy^{-2}\ h^-1}$)")
    ax.set_yscale('log')
    ax.set_xlim(200,460)
    ax.set_ylim([10**-5,10**-2])
    ax.tick_params(direction='in',which='both')
    ax.grid(True)
    ax.set_title("R="+str(int(D2goal['R'][0]))+", PWV=" + str(D2goal['PWV'][0]) + "mm, EL="+str(int(D2goal['EL'][0]))+'deg',
                 fontsize=12)
    ax.legend()
    plt.tight_layout()

    # Create download link
    #................................

    df_download = D2goal[['F','MS']]
    df_download = df_download.rename(columns={'MS':'MS (goal)'})
    df_download = df_download.join(D2baseline[['MS']])
    df_download = df_download.rename(columns={'MS':'MS (baseline)'})

    return create_download_link(df_download,filename='MS.csv')


def PlotD2HPBW():

    F = np.logspace(np.log10(220),np.log10(440),349)*1e9

    fig, ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(F/1e9,D2HPBW(F)*180*60*60/np.pi,linewidth=1,color='b',alpha=1,label='HPBW')

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("HPBW (arcsec)")
    ax.set_yscale('linear')
    ax.set_xlim(200,460)
    # ax.set_ylim([10**-5,10**-2])
    ax.tick_params(direction='in',which='both')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # Create download link
    #................................

    df_download = pd.DataFrame(data=F,columns=['F'])
    df_download['HPBW'] = D2HPBW(F)*180*60*60/np.pi

    return create_download_link(df_download,filename='HPBW.csv')


# helper functions
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index =False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
