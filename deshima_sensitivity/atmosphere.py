# standard library
from pathlib import Path
from typing import Callable, List, Union


# dependent packages
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d


# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]


# main functions
def eta_atm_func(
    F: ArrayLike, pwv: float, EL: float = 60.0, R: float = 0.0
) -> ArrayLike:
    """Calculate eta_atm as a function of F by interpolation.

    If R~=0 then the function will average the atmospheric transmission
    within each spectrometer channel.

    Parameters
    ----------
    F
        Frequency of the astronomical signal.
        Units: Hz (works also for GHz, will detect).
    pwv
        Precipitable water vapour. Units: mm.
    EL
        Telescope elevation angle. Units: degrees.
    R
        Spectral resolving power in F/W_F where W_F is the 'equivalent bandwidth'.
        R is used to average the atmospheric trannsmission within one spectrometer
        channel. If R = 0, then the function will return the transmission
        at that exact frequency. Units: None.
        See also: http://www.astrosurf.com/buil/us/spe2/hresol7.htm

    Returns
    -------
    eta_atm
        Atmospheric tranmsmission. Units: None.

    """
    if np.average(F) > 10.0 ** 9:
        F = F / 10.0 ** 9

    # give F a length if it is an integer.
    if not hasattr(F, "__len__"):
        F = np.asarray([F])

    eta_atm_df = pd.read_csv(
        Path(__file__).parent / "data" / "atm.csv",
        skiprows=4,
        delim_whitespace=True,
        header=0,
    )

    eta_atm_func_zenith = eta_atm_interp(eta_atm_df)

    if R == 0:
        eta_atm = np.abs(eta_atm_func_zenith(pwv, F)) ** (
            1.0 / np.sin(EL * np.pi / 180.0)
        )
    else:
        # smooth with spectrometer resolution
        # 100.0, 100.1., ....., 1000 GHz as in the original data.
        F_highres = eta_atm_df["F"]
        eta_atm_zenith_highres = np.abs(eta_atm_func_zenith(pwv, F_highres)) ** (
            1.0 / np.sin(EL * np.pi / 180.0)
        )
        eta_atm = np.zeros(len(F))
        for i_ch in range(len(F)):
            eta_atm[i_ch] = np.mean(
                eta_atm_zenith_highres[
                    (F_highres > F[i_ch] * (1 - 0.5 / R))
                    & (F_highres < F[i_ch] * (1 + 0.5 / R))
                ]
            )
        eta_atm = [eta_atm]

    if len(eta_atm) == 1:
        return eta_atm[0]
    else:
        return eta_atm[:, 0]


# helper functions
def eta_atm_interp(eta_atm_dataframe: pd.DataFrame) -> Callable:
    """Used in the function eta_atm_func().

    Returns a function that interpolates atmospheric transmission data
    downloaded from ALMA (https://almascience.eso.org/about-alma/atmosphere-model).

    The returned function has the form of eta = func(F [GHz], pwv [mm]).
    Note telescope EL = 90 (zenith).

    Parameters
    ----------
    eta_atm_dataframe
        Pandas DataFrame of atmospheric transmission data.

    Returns
    --------
    func
        Function that returns the atmospheric transmission.

    Example
    --------
        % read csv file with pandas (in e.g., Jupyter)
        eta_atm_df = pd.read_csv("<desim-folder>/data/atm.csv",skiprows=4,
                                 delim_whitespace=True,header=0)
        % make function from pandas file
        etafun = desim.eta_atm_interp(eta_atm_df)

    """
    x = np.array(list(eta_atm_dataframe)[1:]).astype(np.float)
    y = eta_atm_dataframe["F"].values
    z = eta_atm_dataframe.iloc[:, 1:].values
    return interp2d(x, y, z, kind="cubic")
