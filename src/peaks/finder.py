import src.nuclide.api as npi
import src.measurements.api as mpi
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from uncertainties import unumpy
import warnings
from scipy.stats import norm
from config import DB
import logging
from sqlalchemy import text
from scipy.interpolate import interp1d


class PeakFinder:
    def __init__(
        self,
        selected_date,
        nuclides,
        nuclides_intensity,
        interpolate_energy=True,
        prominence=1000,
        width=None,
        rel_height=None,
        matching_ratio=1 / 15,
        tolerance=0.5,
        schema="processed_measurements",
        step_size=0.34507313512321336,
        **kwargs,
    ):
        self.selected_date = selected_date
        self.data = mpi.measurement([selected_date])
        self.meta = mpi.meta_data([selected_date])
        self.polynomial = self.__polynomial_f()
        self.isotopes = npi.nuclides(nuclides, nuclides_intensity)
        logging.warning(
            f"Isotopes: {self.isotopes['nuclide_id'].unique()}, {len(self.isotopes['nuclide_id'].unique())}"
        )
        self.identified_isotopes = []
        self.peak_confidences = []
        self.isotope_confidences = []
        self.percentage_matched = []
        self.identified_peaks = []
        self.identified_peaks_idx = []
        self.prominence = prominence
        self.width = width
        self.rel_height = rel_height
        self.matching_ratio = matching_ratio
        self.tolerance = tolerance
        self.schema = schema
        self.step_size = step_size
        self.interpolate_energy = interpolate_energy
        self.kwargs = kwargs

    def __fit_gaussian(self, peaks: np.ndarray, properties: dict) -> np.ndarray:
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

        def gaussian(x, a, x0, sigma):
            return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

        fitted_peak_mean = []
        fitted_peak_std = []

        for peak, left_base, right_base in zip(
            peaks, properties["left_bases"], properties["right_bases"]
        ):
            try:
                x = np.arange(left_base, right_base)
                # fallback to +-5 if the range is too small to fit or too large to be realistic
                if len(x) <= 10:
                    x = np.arange(peak - 5, peak + 5)
                elif len(x) > 30:
                    x = np.arange(peak - 5, peak + 5)
                y = self.data["counts_cleaned"].iloc[x]
                popt, _ = curve_fit(
                    gaussian, x, y, p0=[y.max(), x.mean(), 1], maxfev=2000
                )
                fitted_peak_mean.append(np.abs(popt[1]))
                fitted_peak_std.append(np.abs(popt[2]))
            except RuntimeError:
                fitted_peak_mean.append(0)
                fitted_peak_std.append(1)
        fitted_peaks = unumpy.uarray(fitted_peak_mean, fitted_peak_std)
        return self.polynomial(fitted_peaks)

    def __identify_background(self, wndw: int = 5, scale: float = 1.5) -> np.ndarray:
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
        self.data = self.data.sort_index()
        self.data["count"] = self.data["count"].astype(float)
        counts = self.data["count"].values
        slopes = np.abs(np.diff(counts))
        moving_avg = np.convolve(slopes, np.ones(wndw) / wndw, mode="same")
        threshold = np.mean(moving_avg) * scale
        background_mask = moving_avg < threshold
        background_mask = np.append(
            background_mask, True
        )  # Ensure the mask has the same length as counts
        background = np.interp(
            np.arange(len(counts)),
            np.arange(len(counts))[background_mask],
            counts[background_mask],
        )
        return background

    def __calculate_confidence(self, peak, energy, std):
        """Calculate confidence score for peak matching an energy value."""
        # Use a Gaussian probability density function
        confidence = norm.pdf(energy, loc=peak, scale=std)
        # Normalize to [0,1] range
        confidence = confidence / norm.pdf(peak, loc=peak, scale=std)
        return confidence

    def __identify_isotopes_matches_and_confidence(
        self,
        energies,
        peak,
        std,
        matched,
        nuclide_id,
    ):
        for energy in energies:
            peak_confidence = self.__calculate_confidence(peak, energy, std)
            if peak_confidence > self.tolerance:
                matched.append(nuclide_id)
                self.peak_confidences.append(peak_confidence)
                print(
                    f"Peak at {peak:2f} +- {std:2f} keV matched to {nuclide_id} at "
                    f"{energy} keV with confidence {peak_confidence:.2f}"
                )

    def __identify_isotopes(
        self, fitted_peaks: unumpy.uarray, peaks_idx: np.array = None
    ):
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

        for idx, (peak, std) in enumerate(
            zip(unumpy.nominal_values(fitted_peaks), unumpy.std_devs(fitted_peaks))
        ):
            for nuclide_id, nuclide_group in self.isotopes.groupby("nuclide_id"):
                matched = []
                energies = nuclide_group["energy"].to_list()
                self.__identify_isotopes_matches_and_confidence(
                    energies,
                    peak,
                    std,
                    matched,
                    nuclide_id,
                )

                if len(matched) == 0:
                    continue

                elif len(matched) / len(energies) >= self.matching_ratio:
                    peak_idx = peaks_idx[idx]
                    self.identified_isotopes.append(nuclide_id)
                    self.isotope_confidences.append(np.mean(self.peak_confidences))
                    self.percentage_matched.append(len(matched) / len(energies))
                    self.identified_peaks.append(peak)
                    self.identified_peaks_idx.append(peak_idx)

    def __polynomial_f(self):
        return (
            lambda x: self.meta["coef_1"][0]
            + self.meta["coef_2"][0] * x
            + self.meta["coef_3"][0] * x**2
            + self.meta["coef_4"][0] * x**3
        )

    def __safe_processed_spectrum(self):
        self.data[
            [
                "energy",
                "background",
                "total_confidence",
                "matched",
                "confidence",
                "identified_peak",
            ]
        ] = self.data[
            [
                "energy",
                "background",
                "total_confidence",
                "matched",
                "confidence",
                "identified_peak",
            ]
        ].round(2)
        query = text("""
                     DELETE
                     FROM measurements.processed_measurements
                     WHERE "datetime" = :selected_date;
                     """)
        with DB.ENGINE.connect() as connection:
            try:
                with connection.begin():
                    connection.execute(query, {"selected_date": self.selected_date})
                    logging.warning(self.selected_date)
            except Exception as e:
                logging.warning(f"Deletion failed: {e}")
        self.data.to_sql(
            self.schema,
            DB.ENGINE,
            if_exists="append",
            index=False,
            schema="measurements",
        )

    def __interpolate_spectrum(self, energy_original, counts_original):
        energy_max = self.step_size * 8160
        energy_axis = np.arange(0, energy_max, self.step_size)
        logging.warning(f"Max Interpolation Value for Energy: {energy_max}")
        f = interp1d(
            energy_original,
            counts_original,
            kind="nearest",
            bounds_error=False,
            fill_value=0,
        )
        return f(energy_axis), energy_axis

    def process_spectrum(
        self,
        return_detailed_view: bool = False,
    ) -> pd.DataFrame:
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

        self.data = self.data.sort_values(by="energy").reset_index(drop=True)
        self.data["background"] = self.__identify_background()
        self.data["counts_cleaned"] = self.data["count"] - self.data["background"]
        peaks_idx, properties = find_peaks(
            self.data["counts_cleaned"],
            prominence=self.prominence,
            width=self.width,
            rel_height=self.rel_height,
            **self.kwargs,
        )

        fitted_peaks = self.__fit_gaussian(peaks=peaks_idx, properties=properties)
        self.__identify_isotopes(
            fitted_peaks=fitted_peaks,
            peaks_idx=peaks_idx,
        )
        total_confidences = [
            c * p for c, p in zip(self.isotope_confidences, self.percentage_matched)
        ]

        if self.interpolate_energy is True:
            interpolated_counts, energy_axis = self.__interpolate_spectrum(
                self.data["energy"].values,
                self.data["count"].values,
            )
            self.data["energy"] = energy_axis
            self.data["count"] = interpolated_counts

        if return_detailed_view is False:
            self.data = self.data.drop(columns=["counts_cleaned"])
            self.data["peak"] = False
            self.data["interpolated"] = self.interpolate_energy
            self.data["total_confidence"] = 0.0
            self.data["matched"] = 0.0
            self.data["confidence"] = 0.0
            self.data["identified_peak"] = 0.0
            self.data["identified_isotope"] = ""

            self.data.loc[self.identified_peaks_idx, "peak"] = True
            self.data.loc[self.identified_peaks_idx, "total_confidence"] = (
                total_confidences
            )
            self.data.loc[self.identified_peaks_idx, "matched"] = (
                self.percentage_matched
            )
            self.data.loc[self.identified_peaks_idx, "confidence"] = (
                self.isotope_confidences
            )
            self.data.loc[self.identified_peaks_idx, "identified_peak"] = (
                self.identified_peaks
            )
            self.data.loc[self.identified_peaks_idx, "identified_isotope"] = (
                self.identified_isotopes
            )
            self.__safe_processed_spectrum()
            return self.data

        else:
            return pd.DataFrame(
                {
                    "datetime": self.selected_date,
                    "peaks": [peaks_idx],
                    "peaks_polynomial": [self.polynomial(peaks_idx)],
                    "fitted_peaks": [fitted_peaks],
                    "fitted_peaks_mean": [unumpy.nominal_values(fitted_peaks)],
                    "fitted_peaks_std": [unumpy.std_devs(fitted_peaks)],
                    "identified_isotopes": [self.identified_isotopes],
                    "identified_peaks": [self.identified_peaks],
                    "identified_peaks_idx": [self.identified_peaks_idx],
                    "confidences": [self.isotope_confidences],
                    "matched": [self.percentage_matched],
                    "total_confidences": [total_confidences],
                }
            )
