from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from uncertainties import unumpy
import warnings
from scipy.stats import norm
import logging
from sqlalchemy import text
from scipy.interpolate import interp1d
from config.loader import load_engine
import src.peaks.api as ppi
from config.loader import load_config


class PeakFinder(ppi.API):
    def __init__(
            self,
            selected_date,
            nuclides,
            nuclides_intensity,
            data=None,
            meta=None,
            interpolate_energy=True,
            prominence=1000,
            width=None,
            rel_height=None,
            wlen=None,
            matching_ratio=1 / 15,
            tolerance=0.5,
            schema="processed_measurements",
            step_size=0.34507313512321336,
            **kwargs,
    ):
        super().__init__()
        self.engine = load_engine()
        self.selected_date = selected_date
        self.data = data
        self.meta = meta
        self.polynomial = self.__polynomial_f()
        self.isotopes = self.npi.nuclides(nuclides, nuclides_intensity)
        logging.warning(
            f"Isotopes: {self.isotopes['nuclide_id'].unique()}, {len(self.isotopes['nuclide_id'].unique())}"
        )
        self.identified_isotopes = []
        self.peak_confidences = []
        self.isotope_confidences = []
        self.percentage_matched = []
        self.identified_peaks = []
        self.identified_peaks_idx = []
        self.compton_edge_idx = []
        self.prominence = prominence
        self.width = width
        self.wlen = wlen
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
            return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

        fitted_peak_mean = []
        fitted_peak_std = []

        for peak, left_base, right_base in zip(
                peaks, properties["left_bases"], properties["right_bases"]
        ):
            try:
                x = np.arange(left_base, right_base)
                # fallback to +-5 if the range is too small to fit or too large to be realistic
                # if len(x) <= 10:
                #     x = np.arange(peak - 5, peak + 5)
                # elif len(x) > 30:
                #     x = np.arange(peak - 5, peak + 5)
                if len(x) <= 10 or len(x) > 30:
                    x = np.arange(max(0, peak - 5), min(len(self.data), peak + 5))
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
        pols = self.polynomial(fitted_peaks)
        compton_edges = self.calculate_compton_effect(pols)
        return pols, compton_edges

    def calculate_compton_effect(self, pols):
        compton_edges = []
        for peak in pols:
            compton_edges.append(peak - (peak / (1 + (2 * peak / 511))))
        return np.array(compton_edges)


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
        std = max(std, 1.0)
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
                logging.warning(
                    f"Peak at {peak:2f} +- {std:2f} keV matched to {nuclide_id} at "
                    f"{energy} keV with confidence {peak_confidence:.2f}"
                )

    def __identify_isotopes(
            self, fitted_peaks: unumpy.uarray, peaks_idx: np.array = None, compton_edges=np.array
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
                self.peak_confidences = []
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
                    match_ratio = len(matched) / len(energies)
                    mean_confidence = np.mean(self.peak_confidences)
                    if peak_idx in self.identified_peaks_idx:
                        existing_idx = self.identified_peaks_idx.index(peak_idx)
                        existing_conf = self.isotope_confidences[existing_idx]
                        if mean_confidence > existing_conf:
                            self.identified_isotopes[existing_idx] = nuclide_id
                            self.isotope_confidences[existing_idx] = mean_confidence
                            self.percentage_matched[existing_idx] = match_ratio
                            self.identified_peaks[existing_idx] = peak
                            if (peak - compton_edges).min() <= 5 and (peak - compton_edges).min() >= -5:
                                self.compton_edge_idx.append(peak_idx)
                                logging.warning(f"Found Compton Edge for peak at: {peak} keV")
                    else:
                        self.identified_isotopes.append(nuclide_id)
                        self.isotope_confidences.append(mean_confidence)
                        self.percentage_matched.append(match_ratio)
                        self.identified_peaks.append(peak)
                        self.identified_peaks_idx.append(peak_idx)
                        if (peak - compton_edges).min() <= 5 and (peak - compton_edges).min() >= -5:
                            self.compton_edge_idx.append(peak_idx)
                            logging.warning(f"Found Compton Edge for peak at: {peak} keV")

    def __polynomial_f(self):
        if self.meta is None:
            self.data = self.data.sort_values(by="energy")
            x = self.data["energy"].to_numpy()
            coef = np.polyfit(
                np.arange(0, load_config()["measurements"]["number_of_channels"], 1),
                x,
                deg=3,
            )
            logging.warning(f"Calculated Coefficients: {coef}")
            return lambda x: coef[-1] + coef[-2] * x + coef[-3] * x ** 2 + coef[-4] * x ** 3

        else:
            return (
                lambda x: self.meta["coef_1"][0]
                          + self.meta["coef_2"][0] * x
                          + self.meta["coef_3"][0] * x ** 2
                          + self.meta["coef_4"][0] * x ** 3
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
        if self.schema == "processed_synthetics":
            query = text("""
                         DELETE
                         FROM measurements.processed_synthetics
                         WHERE "datetime" = :selected_date;
                         """)

        elif self.schema == "processed_measurements":
            query = text("""
                         DELETE
                         FROM measurements.processed_measurements
                         WHERE "datetime" = :selected_date;
                         """)
        with self.engine.connect() as connection:
            try:
                with connection.begin():
                    connection.execute(query, {"selected_date": self.selected_date})
                    logging.warning(self.selected_date)
            except Exception as e:
                logging.warning(f"Deletion failed: {e}")
        self.data.to_sql(
            self.schema,
            self.engine,
            if_exists="append",
            index=False,
            schema="measurements",
        )

    def __interpolate_spectrum(self, energy_original, counts_original):
        energy_max = (
                self.step_size * load_config()["measurements"]["number_of_channels"]
        )
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

    def __interpolation_process(self):
        interpolated_df = pd.DataFrame()
        interpolated_counts, energy_axis = self.__interpolate_spectrum(
            self.data["energy"].values,
            self.data["count"].values,
        )

        interpolated_df["datetime"] = self.data["datetime"]
        interpolated_df["energy"] = energy_axis
        interpolated_df["count"] = interpolated_counts
        self.data = pd.merge_asof(
            interpolated_df.sort_values(by="energy"),
            self.data.sort_values(by="energy"),
            on="energy",
            direction="nearest",
        )

        self.data = self.data.drop(columns=["count_y", "datetime_y"])
        self.data = self.data.rename(
            columns={"count_x": "count", "datetime_x": "datetime"}
        )

        self.data["energy"] = self.data["energy"].round(3)

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
        # self.data["count"] = (self.data["count"] - self.data["count"].min()) / (self.data["count"].max() - self.data["count"].min())

        std = self.data["count"].std()
        mean = self.data["count"].mean()
        self.data["count"] = (self.data["count"] - mean) / std
        self.data["background"] = self.__identify_background()
        self.data["counts_cleaned"] = self.data["count"] - self.data["background"]
        peaks_idx, properties = find_peaks(
            self.data["counts_cleaned"],
            prominence=self.prominence,
            width=self.width,
            rel_height=self.rel_height,
            wlen=self.wlen,
            **self.kwargs,
        )

        fitted_peaks, compton_edges = self.__fit_gaussian(peaks=peaks_idx, properties=properties)
        self.__identify_isotopes(
            fitted_peaks=fitted_peaks,
            peaks_idx=peaks_idx,
            compton_edges=compton_edges
        )
        total_confidences = [
            c * p for c, p in zip(self.isotope_confidences, self.percentage_matched)
        ]

        self.data = self.data.drop(columns=["counts_cleaned"])
        self.data["peak"] = False
        self.data["compton_edge"] = False
        self.data["interpolated"] = self.interpolate_energy
        self.data["total_confidence"] = 0.0
        self.data["matched"] = 0.0
        self.data["confidence"] = 0.0
        self.data["identified_peak"] = 0.0
        self.data["identified_isotope"] = ""
        self.data["count"] = self.data["count"] * std + mean  # RESCALE
        self.data["background"] = self.data["background"] * std + mean  # RESCALE

        self.data.loc[self.identified_peaks_idx, "peak"] = True
        self.data.loc[self.compton_edge_idx, "peak"] = False
        self.data.loc[self.compton_edge_idx, "compton_edge"] = True
        self.data.loc[self.identified_peaks_idx, "total_confidence"] = total_confidences
        self.data.loc[self.identified_peaks_idx, "matched"] = self.percentage_matched
        self.data.loc[self.identified_peaks_idx, "confidence"] = (
            self.isotope_confidences
        )
        self.data.loc[self.identified_peaks_idx, "identified_peak"] = (
            self.identified_peaks
        )
        self.data.loc[self.identified_peaks_idx, "identified_isotope"] = (
            self.identified_isotopes
        )

        if self.interpolate_energy is True:
            self.__interpolation_process()

        if return_detailed_view is False:
            self.__safe_processed_spectrum()
        else:
            return self.data
