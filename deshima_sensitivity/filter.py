# standard library
from typing import List, Union, Tuple

# dependent packages
import numpy as np
import pandas as pd
from lmfit.models import LorentzianModel
from scipy.interpolate import interp1d
from scipy.stats import cauchy

# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]


# main functions
def eta_filter_lorentzian(
    F: ArrayLike,
    FWHM: ArrayLike,
    eta_circuit: ArrayLike = 1,
    F_res: int = 30,
    overflow: int = 80,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Calculate the filter transmissions as a matrix of
        Lorentzian approximations. Also calculates approximating box filter

    Parameters
    ----------
    F
        Center frequency of the filter channels.
        Units: Hz (works also for GHz, will detect)
    FWHM
        Full width at half maximum of the filter channels.
        Units: same as F.
    eta_circuit
        Average transmission over the FWHM. Equal to pi/4 times the peak transmission
        Units: none.
    F_res
        The number of frequency bins within a FWHM
        Units: none.
    Overflow
        The number of extra FHWM's below the first and above the last channel
        Units: none.


    Returns
    -------
    eta_filter
        The filter transmission as an m x n matrix
        m: the number of integration bins.
        n: the number of filter channels.
        Units: None.
    eta_inband
        Whether a frequency is out- (false) or inband (true) as an m x n matrix
        m: the number of integration bins.
        n: the number of filter channels.
        Units: None.
    F_int
        Frequency integration bins.
        Units: Hz.
    W_F_int
        Integration bandwith bins.
        Units: Hz.
    box_height
        The transmission of the box-filter approximation.
        Units: None.
    box-width
        The bandwidth of the box-filter approximation.
        Units: Hz.
    chi-sq
        Zero. For compatibility with the .csv fit
        Units: None.
    """

    if np.average(F) < 10.0**9:
        F = F * 10.0**9
        FWHM = FWHM * 10.0**9

    # give F a length if it is an integer.
    if not hasattr(F, "__len__"):
        F = np.asarray([F])
    # give FWHM a length if it is an integer.
    if not hasattr(FWHM, "__len__"):
        FWHM = np.asarray([FWHM])

    F_int, W_F_int = expand_F(F, FWHM, F_res, overflow)

    eta_filter = (
        eta_circuit
        * 4
        / np.pi
        * (cauchy.pdf(F_int[np.newaxis].T, F, FWHM / 2) * FWHM / 2 * np.pi).T
    )

    # Equivalent in-band box filter approximation
    box_height = eta_circuit
    box_width = FWHM

    chi_sq = np.zeros(len(F))

    eta_inband = eta_inband_mask(F_int, F, box_width / 2) * eta_filter

    return eta_filter, eta_inband, F, F_int, W_F_int, box_height, box_width, chi_sq


def eta_filter_csv(
    file: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Read filter transmissionsfrom csv and return filter matrix,
    integrationbins and integration bin bandwith

    Parameters
    ----------
    file
        A string to the location of the .csv file
        .csv has headers of frequency bins (in GHz) and rows of channels

    Returns
    -------
    eta_filter
        The filter transmission as an m x n matrix
        m: the number of integration bins.
        n: the number of filter channels.
        Units: None.
    eta_inband
        Whether a frequency is out- (false) or inband (true) as an m x n matrix
        m: the number of integration bins.
        n: the number of filter channels.
        Units: None.
    F
        Center frequency of each channel
    F_int
        The integration bins. Units: Hz.
    F_W_int
        The integration bandwith. units: Hz.
    box_height
        The transmission of the box-filter approximation.
        Units: None.
    box_width
        The bandwidth of the box-filter approximation.
        Units: Hz
    chi_sqr
        The Chi Square value of the Lorentzian fit.
        Units: None.

    """
    eta_filter_df = pd.read_csv(file, header=0)

    F_int = eta_filter_df.columns.values.astype(float)

    if np.average(F_int) < 10.0**9:
        F_int = F_int * 10.0**9
        eta_filter_df.columns = F_int

    # Fit to lorentzian model
    fit = np.apply_along_axis(fit_lorentzian, 1, eta_filter_df.to_numpy(), x=F_int)
    fit_df = pd.DataFrame(fit, columns=["Center", "HWHM", "max height", "chi sq"])

    eta_filter_df = eta_filter_df.join(fit_df)

    # Sort frequency bins
    eta_filter_df = eta_filter_df.sort_values("Center", axis=0)
    eta_filter_df.set_index(np.arange(0, len(eta_filter_df)), inplace=True)

    # Extract values
    F = eta_filter_df["Center"].to_numpy(float)

    # Make filter matrix
    eta_filter = eta_filter_df.to_numpy()[:, :-4]

    # calculate integration bandwith, copy second-last bin BW to last
    W_F_int = np.copy(F_int)
    W_F_int[0:-2] = F_int[1:-1] - F_int[0:-2]
    W_F_int[-2] = W_F_int[-3]

    # Equivalent in-band box filter approximation
    box_height = np.pi / 4 * eta_filter_df["max height"].to_numpy(float)
    box_width = 2 * eta_filter_df["HWHM"].to_numpy(float)
    chi_sq = eta_filter_df["chi sq"].to_numpy(float)

    eta_inband = eta_inband_mask(F_int, F, box_width / 2) * eta_filter

    return eta_filter, eta_inband, F, F_int, W_F_int, box_height, box_width, chi_sq


def weighted_average(var: ArrayLike, eta_filter: np.ndarray) -> ArrayLike:
    """Returns the average of a variable over the filter channels

    Parameters
    ----------
    var
        A variable varying over the frequency range.
        Units: varying.
    eta_filter:
        The filter transmission as an m x n matrix
        m: the number of integration bins.
        n: the number of filter channels.
        Units: None.

    Returns
    -------
    average
        The average of var over each filter channel.
        Units: same as var.
    """
    if len(var) == 1:
        return var

    var_array = np.tile(var, (np.shape(eta_filter)[0], 1))
    average = np.average(var_array, weights=eta_filter, axis=1)
    return average


# Helper functions
def expand_F(
    F: ArrayLike,
    FHWM: ArrayLike,
    F_res: int = 30,
    overflow: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expands the given channel(s) to create frequency bins over which to integrate.

    If a single channel is given the function will expand by 2 * overflow * R
    using  2 * F_res * Overflow bins

    Parameters
    ----------
    F
        Center Frequency of filter channel
        Units: Hz (works also for GHz, will detect).
    FHWM
        The full width at half maximum, given by F/R.
        Units: same as F.
    F_res
        The number of frequency bins within a FWHM
        Units: none.
    Overflow
        The number of extra FHWM's below the first and above the last channel
        Units: none.
    Returns
    -------
    F_int
       Frequency bins for later integration.
       units: Hz.
    W_F_int
        Bandwith of each frequency bin.
        units: Hz.

    """

    if np.average(F) < 10.0**9:
        F = F * 10.0**9
        FHWM = FHWM * 10**9

    N = len(F)
    if N > 1:
        # Entered frequency array
        n = np.linspace(-overflow, N - 1 + overflow, (N + 2 * overflow) * F_res)
        F_int = interp1d(
            np.arange(N),
            F,
            bounds_error=False,
            fill_value="extrapolate",
            kind="quadratic",
        )(n)
    else:
        # Entered a single frequency
        half_spacing = overflow * FHWM
        F_int = np.linspace(F - half_spacing, F + half_spacing, 2 * overflow * F_res)

    # calculate integration bandwith
    W_F_int = np.mean([F_int[2:-1] - F_int[1:-2], F_int[1:-2] - F_int[0:-3]], axis=0)
    F_int = F_int[1:-2]

    return F_int, W_F_int


def fit_lorentzian(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fits profile to lorentzian. Used with np.apply_along_axis.

    Parameters
    ----------
    y
        y values of profile to be fitted.
        Units: none.
    x
        x values of profiles to be fitted
        Units: Hz.

    Returns
    -------
    result_params
       A numpy array of the returning parameters
       ['Center', 'HWHM', 'max height', 'chi sq'].
       Units: [same as x, same as x, None, None].
    """
    model = LorentzianModel()

    center_guess = x[y.argmax()]
    HWHM_guess = x[y.argmax()] / 1000

    params = model.make_params(
        amplitude=y.max() * np.pi * HWHM_guess, center=center_guess, sigma=HWHM_guess
    )

    result = model.fit(y, params, x=x)
    result_params = np.array(
        [
            result.params["center"].value,
            result.params["sigma"].value,
            result.params["height"].value,
            result.chisqr,
        ]
    )
    return result_params


def eta_inband_mask(F_int: np.ndarray, F: np.ndarray, HWHM: np.ndarray) -> np.ndarray:
    """Generates an array to use with weighted_average() that averages over
    in-band (F +- HWHM) frequencies

    Parameters
    ----------
    F_int
       Frequency bins for integration.
       units: Hz or GHz.
    F
        Center frequencies of channels.
        units: Same as F_int.
    HWHM
        Half width at half maximum of these channels
        units: Same as F_int.

    Returns
    -------
    eta_inband
        Whether a frequency is out- (false) or inband (true) as an m x n matrix
        m: the number of integration bins.
        n: the number of filter channels.
        Units: None.
    """
    F = F[np.newaxis].T
    HWHM = HWHM[np.newaxis].T
    F_int = F_int[np.newaxis]

    return np.logical_and(
        np.greater_equal(F_int, F - HWHM), np.less_equal(F_int, F + HWHM)
    )
