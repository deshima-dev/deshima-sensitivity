__all__ = ["MDLF_simple", "MS_simple", "PlotD2HPBW"]


# standard library
import base64
from typing import List, Union


# dependent packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML
from .instruments import D2HPBW, eta_mb_ruze
from .simulator import spectrometer_sensitivity


# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]


# main functions
def MDLF_simple(
    F: ArrayLike,
    pwv: float = 0.5,
    EL: float = 60.0,
    snr: float = 5.0,
    obs_hours: float = 8.0,
) -> HTML:
    """Plot minimum detectable line flux (MDLF) of DESHIMA 2.0 on ASTE.

    Parameters
    ----------
    F
        Frequency. Units: GHz.
    pwv
        Precipitable water vapor. Units: mm.
    EL
        Elevation angle. Units: degrees.
    snr
        Target S/N of the detection.
    obs_hours
        Total hours of observation including ON-OFF and calibration overhead.

    Returns
    -------
    html
        HTML object for download link (to be used in Jupyter notebook).

    """
    # Main beam efficiency of ASTE (0.9 is from EM, ruze is from ASTE)
    eta_mb = eta_mb_ruze(F=F, LFlimit=0.805, sigma=37e-6) * 0.9

    D2goal_input = {
        "F": F,
        "pwv": pwv,
        "EL": EL,
        "snr": snr,
        "obs_hours": obs_hours,
        "eta_mb": eta_mb,
        "theta_maj": D2HPBW(F),  # Half power beam width (major axis)
        "theta_min": D2HPBW(F),  # Half power beam width (minor axis)
        "on_source_fraction": 0.4 * 0.9,  # ON-OFF 40%, calibration overhead of 10%
    }

    D2goal = spectrometer_sensitivity(**D2goal_input)

    D2baseline_input = {
        "F": F,
        "pwv": pwv,
        "EL": EL,
        "snr": snr,
        "obs_hours": obs_hours,
        "eta_mb": eta_mb,
        "theta_maj": D2HPBW(F),  # Half power beam width (major axis)
        "theta_min": D2HPBW(F),  # Half power beam width (minor axis)
        "on_source_fraction": 0.3 * 0.8,  # Goal 0.4*0.9
        "eta_circuit": 0.32 * 0.5,  # eta_inst Goal 16%, Baseline 8%
        "eta_IBF": 0.4,  # Goal 0.6
        "KID_excess_noise_factor": 1.2,  # Goal 1.1
    }

    D2baseline = spectrometer_sensitivity(**D2baseline_input)

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(
        D2baseline["F"] / 1e9,
        D2baseline["MDLF"],
        "--",
        linewidth=1,
        color="b",
        alpha=1,
        label="Baseline",
    )
    ax.plot(
        D2goal["F"] / 1e9, D2goal["MDLF"], linewidth=1, color="b", alpha=1, label="Goal"
    )
    ax.fill_between(
        D2baseline["F"] / 1e9, D2baseline["MDLF"], D2goal["MDLF"], color="b", alpha=0.2
    )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"Minimum Detectable Line Flux ($\mathrm{W\, m^{-2}}$)")
    ax.set_yscale("log")
    ax.set_xlim(200, 460)
    ax.set_ylim([10 ** -20, 10 ** -17])
    ax.tick_params(direction="in", which="both")
    ax.grid(True)
    ax.set_title(
        f"R = {int(D2goal['R'][0])}, "
        f"snr = {int(D2goal['snr'][0])}, "
        f"t_obs = {D2goal['obs_hours'][0]} h (incl. overhead), "
        f"PWV = {D2goal['PWV'][0]} mm, "
        f"EL = {int(D2goal['EL'][0])} deg",
        fontsize=12,
    )
    ax.legend()
    fig.tight_layout()

    # Create download link
    df_download = D2goal[["F", "MDLF"]]
    df_download = df_download.rename(columns={"MDLF": "MDLF (goal)"})
    df_download = df_download.join(D2baseline[["MDLF"]])
    df_download = df_download.rename(columns={"MDLF": "MDLF (baseline)"})

    return create_download_link(df_download, filename="MDLF.csv")


def MS_simple(F: ArrayLike, pwv: float = 0.5, EL: float = 60.0) -> HTML:
    """Plot mapping speed of DESHIMA 2.0 on ASTE.

    Parameters
    ----------
    F
        Frequency. Units: GHz.
    pwv
        Precipitable water vapor. Units: mm.
    EL
        Elevation angle. Units: degrees.

    Returns
    -------
    html
        HTML object for download link (to be used in Jupyter notebook).

    """
    # Main beam efficiency of ASTE (0.9 is from EM, ruze is from ASTE)
    eta_mb = eta_mb_ruze(F=F, LFlimit=0.805, sigma=37e-6) * 0.9

    D2goal_input = {
        "F": F,
        "pwv": pwv,
        "EL": EL,
        "eta_mb": eta_mb,
        "on_off": False,
        "theta_maj": D2HPBW(F),  # Half power beam width (major axis)
        "theta_min": D2HPBW(F),  # Half power beam width (minor axis)
    }

    D2goal = spectrometer_sensitivity(**D2goal_input)

    D2baseline_input = {
        "F": F,
        "pwv": pwv,
        "EL": EL,
        "eta_mb": eta_mb,
        "on_off": False,
        "theta_maj": D2HPBW(F),  # Half power beam width (major axis)
        "theta_min": D2HPBW(F),  # Half power beam width (minor axis)
        "eta_circuit": 0.32 * 0.5,  # eta_inst Goal 16%, Baseline 8%
        "eta_IBF": 0.4,  # Goal 0.6
        "KID_excess_noise_factor": 1.2,  # Goal 1.1
    }

    D2baseline = spectrometer_sensitivity(**D2baseline_input)

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(
        D2baseline["F"] / 1e9,
        D2baseline["MS"],
        "--",
        linewidth=1,
        color="b",
        alpha=1,
        label="Baseline",
    )
    ax.plot(
        D2goal["F"] / 1e9, D2goal["MS"], linewidth=1, color="b", alpha=1, label="Goal"
    )
    ax.fill_between(
        D2baseline["F"] / 1e9, D2baseline["MS"], D2goal["MS"], color="b", alpha=0.2
    )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"Mapping Speed ($\mathrm{arcmin^2\, mJy^{-2}\, h^{-1}}$)")
    ax.set_yscale("log")
    ax.set_xlim(200, 460)
    ax.set_ylim([10 ** -5, 10 ** -2])
    ax.tick_params(direction="in", which="both")
    ax.grid(True)
    ax.set_title(
        f"R = {int(D2goal['R'][0])}, "
        f"PWV = {D2goal['PWV'][0]} mm, "
        f"EL = {int(D2goal['EL'][0])} deg",
        fontsize=12,
    )
    ax.legend()
    fig.tight_layout()

    # Create download link
    df_download = D2goal[["F", "MS"]]
    df_download = df_download.rename(columns={"MS": "MS (goal)"})
    df_download = df_download.join(D2baseline[["MS"]])
    df_download = df_download.rename(columns={"MS": "MS (baseline)"})

    return create_download_link(df_download, filename="MS.csv")


def PlotD2HPBW() -> HTML:
    """Plot half power beam width of DESHIMA 2.0 on ASTE.

    Returns
    -------
    html
        HTML object for download link (to be used in Jupyter notebook).

    """
    F = np.logspace(np.log10(220), np.log10(440), 349) * 1e9

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(
        F / 1e9,
        D2HPBW(F) * 180 * 60 * 60 / np.pi,
        linewidth=1,
        color="b",
        alpha=1,
        label="HPBW",
    )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("HPBW (arcsec)")
    ax.set_yscale("linear")
    ax.set_xlim(200, 460)
    ax.tick_params(direction="in", which="both")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # Create download link
    df_download = pd.DataFrame(data=F, columns=["F"])
    df_download["HPBW"] = D2HPBW(F) * 180 * 60 * 60 / np.pi

    return create_download_link(df_download, filename="HPBW.csv")


# helper functions
def create_download_link(
    df: pd.DataFrame, title: str = "Download CSV file", filename: str = "data.csv"
) -> HTML:
    """Create an HTML object for download link of a DataFrame.

    This function convert a DataFrame to a CSV file and create an HTML object
    object of anchor (a tag) in which the CSV file is embedded as Base64 format.
    The return HTML object is intended to be used in a Jupyter notebook.

    Parameters
    ----------
    df
        Pandas DataFrame to be embedded in an HTML anchor (a tag).
    title
        Text of an HTML anchor.
    filename
        Filename of download

    Returns
    -------
    html
        HTML object for download link (to be used in Jupyter notebook).

    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()

    href = f"data:text/csv;base64,{payload}"
    target = "_blank"

    return HTML(f'<a download="{filename}" href="{href}" target="{target}">{title}</a>')
