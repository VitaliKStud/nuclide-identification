import src.nuclide.api as npi
import src.measurements.api as mpi
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from uncertainties import unumpy
import warnings
from scipy.stats import norm
import matplotlib.pyplot as plt


def gaussian(x, a, mu, sigma):
    """
    Gaussian (normal distribution) function.

    Parameters:
    -----------
    x : array-like or float
        The input value(s) at which to evaluate the Gaussian function.
    a : float
        Amplitude of the peak (maximum height of the curve).
    mu : float
        Mean (center) of the peak.
    sigma : float
        Standard deviation (controls the width of the peak).

    Returns:
    --------
    array-like or float
        The value(s) of the Gaussian function at x.
    """
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fit_gaussian(data: pd.DataFrame, peaks: np.ndarray, properties: dict, polynomial: callable) -> np.ndarray:
    """
    Fits Gaussian functions to the identified peaks in the data and applies the polynomial to the fitted peaks.

    Parameters:
    -----------
    data : pd.DataFrame
        The processed data containing the counts.
    peaks : np.ndarray
        The indices of the identified peaks.
    widths : np.ndarray
        The widths of the identified peaks.
    polynomial : callable
        The polynomial function to be applied to the fitted peaks.

    Returns:
    --------
    np.ndarray
        The polynomial values at the fitted peaks with uncertainties.

    Notes:
    ------
    - The function uses the `curve_fit` method from `scipy.optimize` to fit Gaussian functions to the peaks.
    - The fitted peaks are represented as `uncertainties` arrays with mean and standard deviation.
    """

    warnings.filterwarnings("ignore", category=OptimizeWarning)

    fitted_peak_mean = []
    fitted_peak_std = []

    for peak, left_base, right_base in zip(peaks, properties['left_bases'], properties['right_bases']):
        try:
            x = np.arange(left_base, right_base)
            # fallback to +-5 if the range is too small to fit or too large to be realistic
            if len(x) <= 10:
                x = np.arange(peak - 5, peak + 5)
            elif len(x) > 30:
                x = np.arange(peak - 5, peak + 5)
            y = data['count_cleaned'].iloc[x]
            popt, *_ = curve_fit(gaussian, x, y, p0=[y.max(), x.mean(), 1], maxfev=2000)
            fitted_peak_mean.append(np.abs(popt[1]))
            fitted_peak_std.append(np.abs(popt[2]))
        except RuntimeError:
            fitted_peak_mean.append(0)
            fitted_peak_std.append(1)
    fitted_peaks = unumpy.uarray(fitted_peak_mean, fitted_peak_std)
    return polynomial(fitted_peaks)


def identify_background(data: pd.DataFrame, window: int = 5, scale: float = 1.5) -> np.ndarray:
    """
    Identify the background of a spectrum by analyzing the slopes and applying a moving average.
    Parameters:
    -----------
    data : pd.DataFrame
        The processed data containing the counts.
    window : int, optional
        The window size for the moving average (default is 5).
    order : int, optional
        The order of the moving average (default is 3).
    scale : float, optional
        The scale factor to determine the background threshold (default is 1.5).
    Returns:
    --------
    np.ndarray
        The interpolated background values.
    """
    counts = data['count'].values
    slopes = np.abs(np.diff(counts))
    moving_avg = np.convolve(slopes, np.ones(window) / window, mode='same')
    threshold = np.mean(moving_avg) * scale
    background_mask = moving_avg < threshold
    background_mask = np.append(background_mask, True)  # Ensure the mask has the same length as counts
    background = np.interp(np.arange(len(counts)), np.arange(len(counts))[background_mask], counts[background_mask])
    return background


def plot_spectrum(df: pd.DataFrame, semilogy=False, all_fitted_peaks=False):
    """Plot a gamma spectrum from a DataFrame."""
    dates = df['filename'][0]
    data = mpi.measurement([dates])
    data['background'] = identify_background(data)
    print(data.shape)
    data.plot(x='energy', y='count', kind='line', figsize=(10, 6))

    if all_fitted_peaks:
        plt.vlines(df['fitted_peaks_mean'], 0, max(data['count']), color='r', linestyles='dashed',
                   label='Fitted peaks')
    colors = plt.get_cmap("hsv")(np.linspace(0.2, 0.8, len(df['identified_peaks'][0])))
    for i, (identified_peak, isotope) in enumerate(zip(df['identified_peaks'][0], df['identified_isotopes'][0])):
        plt.vlines(identified_peak, -100, max(data['count']) / 10, color=colors[i], alpha=0.5,
                   label=f'ISOTOPE: {isotope}\nENERGY: {round(identified_peak, 2)}')

    if semilogy:
        plt.semilogy('log')

    plt.plot(data['energy'], data['background'], color='r', label='Background')
    plt.title(f'Spectrum: {df["filename"]}')
    plt.legend()
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.show()


def calculate_confidence(peak, energy, std):
    """Calculate confidence score for peak matching an energy value."""
    # Use a Gaussian probability density function
    confidence = norm.pdf(energy, loc=peak, scale=std)
    # Normalize to [0,1] range
    confidence = confidence / norm.pdf(peak, loc=peak, scale=std)
    return confidence


def calculate_intensity_cutoff(data: pd.DataFrame, snr_threshold: float = 3.0) -> float:
    """
    Calculate an intensity cutoff based on the Signal-to-Noise Ratio (SNR).

    Parameters:
    -----------
    data : pd.DataFrame
        The processed data containing the counts and background.
    snr_threshold : float, optional
        The SNR threshold to determine the intensity cutoff (default is 3.0).

    Returns:
    --------
    float
        The calculated intensity cutoff.
    """
    signal_max = data['counts'].max()
    background_mean = data['background'].mean()
    noise = signal_max - background_mean
    snr = signal_max / noise

    if snr < snr_threshold:
        return 0.0  # No significant signal detected
    else:
        return snr_threshold * noise / signal_max


def identify_isotopes(fitted_peaks: unumpy.uarray, tolerance: float = 0.5, matching_ratio: float = 1 / 5) -> tuple:
    """
        Identify isotopes based on fitted peak energies.
        This function compares the provided fitted peak energies with known isotope energies
        and identifies potential isotopes based on a confidence threshold and matching ratio.
        Parameters:
        -----------
        fitted_peaks : unumpy.uarray
            Array of fitted peak energies with uncertainties.
        tolerance : float, optional
            Confidence threshold for matching peaks to isotope energies (default is 0.5).
        matching_ratio : float, optional
            Minimum ratio of matched peaks to isotope energies required to identify an isotope (default is 0.5).
        verbose : bool, optional
            If True, prints detailed matching information (default is False).
        **kwargs : dict
            Additional keyword arguments (not used in this function).
        Returns:
        --------
        identified_isotopes : tuple
            List of identified isotopes.
        confidences : list
            List of confidence values for each matched peak.
        percentage_matched : list
            List of percentages of matched peaks for each identified isotope.
    """

    known_isotopes = npi.nuclides(["cs137",
                                   "co60",
                                   "i131",
                                   "tc99m",
                                   "ra226",
                                   "th232",
                                   "u238",
                                   "k40",
                                   "am241",
                                   "na22",
                                   "eu152",
                                   "eu154"], intensity=5)
    identified_isotopes = []
    peak_confidences = []
    isotope_confidences = []
    percentage_matched = []
    identified_peaks = []

    for peak, std in zip(unumpy.nominal_values(fitted_peaks), unumpy.std_devs(fitted_peaks)):
        for nuclide_id, nuclide_group in known_isotopes.groupby("nuclide_id"):
            matched = []
            nuclide_list = nuclide_group["energy"].to_list()
            for energy in nuclide_list:
                peak_confidence = calculate_confidence(peak, energy, std)
                if peak_confidence > tolerance:
                    matched.append(nuclide_id)
                    peak_confidences.append(peak_confidence)

            if len(matched) == 0:
                continue

            elif len(matched) / len(nuclide_list) >= matching_ratio:
                identified_isotopes.append(nuclide_id)
                isotope_confidences.append(np.mean(peak_confidences))
                percentage_matched.append(len(matched) / len(nuclide_list))
                identified_peaks.append(peak)

    return identified_isotopes, identified_peaks, isotope_confidences, percentage_matched


def process_spectrum(dates: list, prominence: int = 1000, width: int = None, rel_height: float = None,
                     tolerance: float = 0.5, **kwargs) -> pd.DataFrame:
    """
    Processes a gamma spectrum file and identifies peaks.
    This function reads a gamma spectrum data file, processes the data to identify peaks,
    fits Gaussian functions to the peaks, and returns a DataFrame with the results.

    Parameters:
    -----------
    filename : str
        The path to the gamma spectrum data file.
    prominence : int, optional
        The prominence of peaks to be identified (default is 1000).
    width : int, optional
        The width of peaks to be identified (default is None).
    rel_height : float, optional
        The relative height of peaks to be identified (default is None).
    **kwargs : dict
        Additional keyword arguments to be passed to the `find_peaks` function.
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'filename': The name of the processed file.
        - 'data': The processed data as a DataFrame.
        - 'peaks': The indices of the identified peaks.
        - 'properties': The properties of the identified peaks.
        - 'calculated_polynomial': The polynomial values at the identified peaks.
        - 'fitted_peaks': The fitted Gaussian peaks.
    Notes:
    ------
    - The input file is expected to have a header with at least 3 lines.
    - The energy values in the file should be in a column named 'energy in keV' and should use commas as decimal separators.
    - The function reads a polynomial from a specific file ('Daten/2016-11-21_09-27-54_Summenspektrum.txt') to calibrate the energy values. When the polynomial is not found or readable in the input file.
    Example:
    --------
    result = process_spectrum(prominence=1500)
    print(result)
    Time Complexity:
    ----------------
    The time complexity of this function depends on the size of the input data and the complexity of the peak finding and fitting algorithms.
    - Reading and processing the data: O(n), where n is the number of lines in the file.
    - Finding peaks: O(m), where m is the number of data points.
    - Fitting Gaussian functions: O(p), where p is the number of peaks.
    Overall, the time complexity is approximately O(n + m + p).
    """

    data = mpi.measurement(dates)
    meta = mpi.meta_data(dates)

    polynomial = lambda x: meta["coef_1"][0] + meta["coef_2"][0] * x + meta["coef_3"][0] * x ** 2 + meta["coef_4"][
        0] * x ** 3

    background = identify_background(data)
    data['count_cleaned'] = data['count'] - background
    peaks, properties = find_peaks(data['count_cleaned'], prominence=prominence, width=width, rel_height=rel_height,
                                   **kwargs)
    fitted_peaks = fit_gaussian(data, peaks, properties, polynomial=polynomial)
    identified_isotopes, identified_peaks, confidences, matched = identify_isotopes(fitted_peaks, tolerance=tolerance)
    total_confidences = [c * p for c, p in zip(confidences, matched)]
    return pd.DataFrame({
        'filename': dates,
        'data': [data],
        'peaks': [peaks],
        'properties': [properties],
        'calculated_polynomial': [polynomial(peaks)],
        'fitted_peaks': [fitted_peaks],
        'fitted_peaks_mean': [unumpy.nominal_values(fitted_peaks)],
        'fitted_peaks_std': [unumpy.std_devs(fitted_peaks)],
        'identified_isotopes': [identified_isotopes],
        'identified_peaks': [identified_peaks],
        'confidences': [confidences],
        'matched': [matched],
        'total_confidences': [total_confidences]
    })


result = process_spectrum(["2017-04-13 13:36:00"], prominence=1500)
plot_spectrum(result)
