"""Functions for analysing RIVs for resp rate estimation."""

import copy  # to prevent mutating dictionaries in function.
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
from IPython.display import display
from scipy import signal

from ppg2rr import kalman
from ppg2rr.util import closest_to, normalize, sigmoid, standardize


def psd_analysis(
    rivs: dict[str, dict[str, np.ndarray]],
    fs_riv: float,
    samples_from_peak: float = 3,
    resp_range_max_f: float = 2.3,
    resp_range_min_f: float = 0.13,
    min_peak_height: float = 0.1,
    max_lags_samples: int = 150,
    acf_min_peak_height: float = 0,
    include_combined_spectra: bool = True,
    rr_seed: float = None,
    pct_rr_change_from_seed: float = 0.6,
    show: bool = False,
):
    """Calculate RR candidates from RIVs via spectral analysis.

    Spectral analysis involves
    * Calculating the power spectral density of each RIV waveform
    * Calculating the product of all RIV waveforms
    * Calculated an RQI (respiratory quality index) for each RIV waveform based on [1]
        Here, RQI is the ratio between area under the most prominant peak to the
        PSD area under the respiratory range.
    * Calculate the autocorrelation, to help with harmonic analysis.
    * Perform harmonic analysis if necessary.

    This function currently returns two candidates:
    * "psd RR candidate", which is the candidate after performing harmonic analysis
    * "nearest psd RR candidate", which is the candidate closest to the rr seed.
    
    TODO: Try the following logic:
        For each RIV-associated candidate list:
        1. if multiple candidate peaks are found in the periodogram,
            check for harmonics and select the f0
        2. if multiple candidate peaks are found and they're not
            harmonics of some f0, then select the peak closest to the
            previous frame's RR

    Inputs:
        rivs: A nested dictionary with keys corresponding to each ppg-feature
            each dictionary contains keys "y" for the feature value, from which
            the PSD will be derived.
        fs_riv (float, hz): Sampling frequency of the RIV signals.
        samples_from_peak: number of samples surrounding the max PSD peak used in RQI
            calculation. Each sample is ~fs_riv/2048 hz, so +/- 3 samples is about 2
            breaths per minute.
        resp_range_max_f, resp_range_min_f: the upper and lower bound of respiratory
            range to search for PSD peaks, in hertz. Defaults to 2.3 hz (138 breaths
            per minute) and 0.13 hz (7.8) breaths per minute. The resp_range_min_f
            should be adjusted based on the patient population. For example, lower 
            bound of breathing for adults can be as low as 6 breaths per minute, but
            for pediatric patients it can be as high as 20 breaths per minite.
        min_peak_height: minimum relative height of PSD peaks considered.
            Defaults to 0.1.
        max_lags_samples: number of lags to compute during ACF computation.
            ACF analysis is done to inspect the signal for harmonics.
        acf_min_peak_height: Minimum peak height to be considered a RR candidate
            during ACF analysis analysis.
        include_combined_spectra: If True, compute the product of PSD of each RIV.
            This is used to "vote" for the most common peak among all RIVs.
        rr_seed: Used to estimate RR based on the previous frame's RR in
            [breaths per minute].
        pct_rr_change_from_seed: Estimate of percent-difference between this frame's RR
            and the rr_seed. Defaults to 0.6 (60%).
        show: default False.

    Returns:
        An updated `rivs` dictionary with keys:
        f - Frequency axis.
        PSD - Power spectral density.
        freq candidates - Frequencies of the top 3 PSD peaks in the respiratory range.
        freq candidates idx - Frequency indices of the top 3 frequency candidates.
        max PSD freq - Frequency of the max PSD peak in the respiratory range.
        PSD RQI - Respiratory quality index based on the PSD, as described in [1].

    [1] Khreis, S., Ge, D., Rahman, H. A., & Carrault, G. (2020). Breathing Rate
        Estimation Using Kalman Smoother with Electrocardiogram and Photoplethysmogram.
        IEEE Transactions on Biomedical Engineering, 67(3), 893–904.
        https://doi.org/10.1109/TBME.2019.2923448

    """
    # This avoids mutation
    rivs = copy.deepcopy(rivs)
    
    # adjust minimum breathing frequency based on rr_seed. We assume that breathing
    # rate can increase rapidly, but decrease relatively slowly over time.
    if rr_seed is None:
        min_f_based_on_previous = resp_range_min_f
    else:
        min_f_based_on_previous = max(
            resp_range_min_f,
            rr_seed / 60 * pct_rr_change_from_seed
            )
    
    
    # get power spectra estimate via periodogram.
    psd_list = []
    f_axis_list = []
    for key in rivs.keys():
        riv_waveform = rivs[key]["y"]
        f_axis, psd = signal.periodogram(riv_waveform, fs_riv, nfft=2048)
        rivs[key]["f"] = f_axis
        rivs[key]["PSD"] = psd
        f_axis_list.append(f_axis)
        psd_list.append(psd)

    assert np.all(f_axis_list[0] == f_axis_list[-1])

    # If multiple RIVs share the same PSD peak, the product of power spectra across 
    # RIVs would amplify that shared peak.
    if include_combined_spectra:
        # Compute product of all PSD (including original ppg)
        psd_matrix = np.array(psd_list).T
        rivs["product of all PSD"]["PSD"] = np.prod(psd_matrix, axis=1)
        # placeholder to prevent key not exist errors
        rivs["product of all PSD"]["y"] = np.zeros(2048)

    # calculate autocorrelation, for checking harmonics in the next step
    for key in rivs.keys():
        if key == "product of all PSD":
            acf = np.real(np.fft.ifft(rivs[key]["PSD"], n=2048))
            acf = normalize(acf[:max_lags_samples])
        else:
            acf = autocorr_norm(rivs[key]["y"], n_lags=max_lags_samples)
        peak_idx, _ = signal.find_peaks(acf, height=acf_min_peak_height)
        
        if len(peak_idx) > 0:
            sorted_idx = sorted(peak_idx, key=lambda p: acf[p], reverse=True)
            sorted_idx = np.array(sorted_idx)
            acf_rr_candidates = (60 * fs_riv) / sorted_idx # breaths per min
            valid_idx = acf_rr_candidates > resp_range_min_f * 60 * 0.7
            rivs[key]["acf"] = acf
            rivs[key]["acf peak idx"] = sorted_idx[valid_idx]
            rivs[key]["acf peaks"] = acf[sorted_idx][valid_idx]
            rivs[key]["acf rr candidates"] = acf_rr_candidates[valid_idx]
        else:
            rivs[key]["acf"] = []
            rivs[key]["acf peak idx"] = []
            rivs[key]["acf peaks"] = []
            rivs[key]["acf rr candidates"] = []


    # Compute a digital filter to suppress peaks outside the search range
    _, rr_max_f_idx = closest_to(f_axis, resp_range_max_f)
    _, rr_min_f_idx = closest_to(f_axis, resp_range_min_f)
    _, rr_min_f_idx_estimated = closest_to(f_axis, min_f_based_on_previous)
    highpass = sigmoid(f_axis, center=f_axis[rr_min_f_idx] * 0.9, slope=12)
    lowpass = sigmoid(f_axis, center=f_axis[rr_max_f_idx] * 1.1, slope=-12)
    digital_bp_filter = highpass * lowpass
    
    for key in rivs.keys():
        # Apply digital bandpass filter
        filtered_psd = rivs[key]["PSD"] * digital_bp_filter

        # Find the highest PSD peaks within respiratory range.
        # 2nd or 3rd highest peaks can still be good resp rate candidates.
        peak_idx, _ = signal.find_peaks(
            normalize(filtered_psd),
            height=min_peak_height,
        )
        if np.any(peak_idx):
            # if peaks are found, remove ones outside of resp range
            sorted_peaks = np.asarray(
                sorted(peak_idx, key=lambda p: rivs[key]["PSD"][p], reverse=True)
            )

            # get the f-axis indices of the highest peaks within respiratory range
            valid_peaks = sorted_peaks[
                (sorted_peaks > rr_min_f_idx_estimated) & (sorted_peaks < rr_max_f_idx)
            ]

        if ~np.any(peak_idx) or ~np.any(valid_peaks):
            # no valid peaks or no peaks found
            rivs[key]["freq candidates"] = np.array([0])
            rivs[key]["freq candidates idx"] = [0]
            rivs[key]["max PSD freq"] = 0
            rivs[key]["PSD RQI"] = 0
            rivs[key]["psd RR candidate"] = 0
            rivs[key]["freq candidates rel power"] = [0]
            rivs[key]["original psd RR candidate"] = [0]
            rivs[key]["nearest psd RR candidate"] = 0
            rivs[key]["n valid peaks"] = 0
        else:
            rivs[key]["n valid peaks"] = len(valid_peaks)
            max_psd_idx = valid_peaks[0]
            max_psd_freq = f_axis[max_psd_idx]
            rr_candidate = max_psd_freq * 60
            
            # all_rr_candidates = f_axis[sorted_peaks]*60
            # all_peaks = rivs[key]["PSD"][sorted_peaks]*60
            rivs[key]["max PSD freq"] = max_psd_freq  # used for kalman filter initialization
            rivs[key]["original psd RR candidate"] = rr_candidate
            # =============== harmonics analysis ===============
            # check whether rr_candidate is f0 or f1
            f0_candidate = None
            half_psd_freq_exist = np.any(np.abs(f_axis[sorted_peaks]*60 - rr_candidate/2) < 2)
            if (len(rivs[key]["acf rr candidates"]) >= 2) & half_psd_freq_exist:
                f0_candidate = get_f0(
                    psd_freq=rr_candidate,
                    acf_freqs=rivs[key]["acf rr candidates"],
                    acf_sqi=rivs[key]["acf peaks"],
                    min_rr=4,  # resp_range_min_f*60,
                    tolerance=rr_candidate / 10,
                )
                rivs[key]["f0 candidate"] = f0_candidate
                
            if f0_candidate is None:
                rivs[key]["psd RR candidate"] = rr_candidate
            else:
                rivs[key]["psd RR candidate"] = f0_candidate

            # frequency candidates = top 3 peaks in the resp range
            # sort the peaks by height
            rivs[key]["freq candidates"] = f_axis[valid_peaks[:3]]
            rivs[key]["freq candidates idx"] = valid_peaks[:3]

            # compute respiratory quality index using using Eq 1 in Khreis et al., 2020
            power_around_max = np.sum(
                rivs[key]["PSD"][
                    max_psd_idx - samples_from_peak : max_psd_idx + samples_from_peak
                ]
            )
            power_in_resp_range = np.sum(
                rivs[key]["PSD"][rr_min_f_idx:rr_max_f_idx]
            )
            rivs[key]["PSD RQI"] = power_around_max / power_in_resp_range

            rivs[key]["freq candidates rel power"] = (
                rivs[key]["PSD"][rivs[key]["freq candidates idx"]] / power_in_resp_range
            )
            
            # ========= find the candidate closest to the rr seed =========
            if len(rivs[key]["freq candidates"]) > 1 and (rr_seed is not None):
                val, _ = closest_to(
                    arr=rivs[key]["freq candidates"] * 60, target=rr_seed
                )
            else:
                val = rivs[key]["freq candidates"][0] * 60
            rivs[key]["nearest psd RR candidate"] = val

        # NOTE: The MATLAB version of the code also considers weighing the
        # candidate frequencies by their relative peak height
        # weighted_f_candidate = (this_peak_height/sum(all_peak_heights))

    if show:
        cmap = plt.cm.get_cmap("tab10", len(rivs))
        for idx, key in enumerate(rivs.keys()):
            candidate_peak_idx = rivs[key]["freq candidates idx"]
            normalized_psd = normalize(rivs[key]["PSD"])

            # plot PSD for each RIV
            plt.plot(f_axis * 60, normalized_psd, label=key, color=cmap(idx))
            # mark peak candidates
            print(rivs[key]["freq candidates"] * 60)
            print(normalized_psd[candidate_peak_idx])
            plt.plot(
                rivs[key]["freq candidates"] * 60,
                normalized_psd[candidate_peak_idx],
                "o",
                color=cmap(idx),
            )

        plt.xlabel("frequency [Cycles per min]")
        plt.ylabel("PSD [V**2/Hz]")
        plt.xlim(0, 120)
        plt.legend()
        plt.show()

        # ACF spectra
        plt.figure()
        for idx, key in enumerate(rivs.keys()):
            if len(rivs[key]["acf"]) > 0:
                acf = rivs[key]["acf"]
                plt.plot(np.arange(max_lags_samples), acf, label=key, color=cmap(idx))
        x = np.round(np.arange(5, max_lags_samples, 15), 2)
        plt.xticks(ticks=x, labels=np.round(600 / x, 2))
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

        max_psd_freqs = [np.round(rivs[key]["psd RR candidate"], 2) for key in rivs.keys()]
        rqis = [np.round(rivs[key]["PSD RQI"], 4) for key in rivs.keys()]
        df_display = pd.DataFrame(
            {
                "original candidate": [np.round(rivs[key]["original psd RR candidate"], 2) for key in rivs.keys()],
                "max psd freqs": max_psd_freqs,
                "RQIs": rqis,
                "acf RRs": [np.round(rivs[key]["acf rr candidates"], 2) for key in rivs.keys()],
                "acf peak": [np.round(rivs[key]["acf peaks"], 2) for key in rivs.keys()],
            },
            index=rivs.keys(),
        )
        display(df_display)
        for key in rivs.keys():
            print(
                f'{key}: {np.round(rivs[key]["freq candidates"]*60,2)}'
                f'{rivs[key]["freq candidates rel power"]}'
            )

    return rivs


def get_f0(psd_freq: float, acf_freqs, acf_sqi, tolerance=1.5, min_rr=4):
    """Checks of frequency peak in the PSD is the f0 or a multiple of f0, by comparing
    the PSD peak freq against the frequency candidates in the autocorrelation function.
    Returns f0 if able.
    
    #! assumes there exists a corresponding psd peak at psd_freq/2

    Senario 1, pure high frequency signals
    This type of signal will have a single PSD peak at f0, and repeated ACF peaks
    at f0, f0/2, f0/3... etc. i.e., f0 is the least common multiple.
    These ACF peaks typically have the same height.

    Scenario 2, pure two-toned signals:
    In this type of signal, ACF peak is strong at f0 and there may be a peak
    ACF peak at f1=f0*2. Additional peaks may appear at f0/2.

    Scenario 3, three-toned signals:
    These signals typically have f0, f1, and f2 peaks present in the af signal,
    with f0 being the greatest peak, and f0 is roughly the greatest common divisor
    across the three.


    Expects all frequencies to be in breaths per minute.
    
    See more examples in the algorithm wiki: 
    https://github.com/new-horizons/pulseox_pubalgs/wiki/Harmonic-Mitigation

    Args:
        psd_freq: a single frequency to be checked
        acf_freqs: array of frequencies found in the ACF
        acf_sqi: ACF peaks corresponding to each ACF frequency.
        tolerance: [breaths per minute] used to check agreement between RRs.
        min_rr: sets the break condition for recursive checking; breaths per minute.
    """
    acf_freqs = np.array(acf_freqs)
    acf_sqi = np.array(acf_sqi)
    f0 = None # initial value
    sqi_threshold = max(acf_sqi) * 0.4

    valid_criteria = (acf_freqs > min_rr) & (acf_sqi > sqi_threshold)
    acf_sqi = acf_sqi[valid_criteria]
    acf_freqs = acf_freqs[valid_criteria]

    # Identify the fundamental, f1, and f2 peaks in the ACF function.
    # check length again after dropping peaks outside the first 3 psd peak multiples
    if len(acf_freqs) >= 1:
        acf_0, idx_0 = closest_to(acf_freqs, psd_freq)
        acf_half, idx_half = closest_to(acf_freqs, psd_freq / 2)
        acf_third, idx_third = closest_to(acf_freqs, psd_freq / 3)

        if abs(acf_0 - psd_freq) > tolerance:
            acf_0, idx_0 = None, None
        if abs(acf_half - psd_freq / 2) > tolerance / 2:
            acf_half, idx_half = None, None
        if abs(acf_third - psd_freq / 3) > tolerance / 3:
            acf_third, idx_third = None, None

        acf_harmonics = np.array([acf_0, acf_half, acf_third])
        acf_harmonics = acf_harmonics[acf_harmonics != np.array(None)]
        acf_harmonic_sqis = np.array([idx_0, idx_half, idx_third])
        acf_harmonic_sqis = acf_harmonic_sqis[acf_harmonic_sqis != np.array(None)]
        acf_harmonic_sqis = acf_sqi[list(acf_harmonic_sqis)]

        # if peaks remain
        if len(acf_harmonics) == 0:
            f0 = None
        elif len(acf_harmonics) == 1:
            # Scenario 2: Remaining acf peak is either same or 2x the psd peak
            # Could also be 3x the psd peak as in scenario 3.
            # Take the mean frequency between acf and psd peak multiple
            h_n = round(psd_freq / acf_harmonics[0])
            f0 = (acf_harmonics[0] + psd_freq / h_n) / 2
        elif len(acf_harmonics) > 1:
            is_all_multiples = (
                (acf_harmonics / min(acf_harmonics)) % 1
            ) < 0.3  # e.g., 2.52%1=0.52
            divisor_ratios = (np.max(acf_harmonics) / acf_harmonics).astype(float)
            is_common_divisors = (
                abs(np.round(divisor_ratios) - divisor_ratios) < 0.1
            ) & np.all([x in np.round(divisor_ratios) for x in [1, 2, 3]])
            is_similar_heights = (
                np.abs(np.diff(acf_sqi)) < 0.25
            )  # NOTE: perhaps this should be a matrix of differences instead

            # Top two peaks are similar height & is 2x of each other
            # Suggests 2nd peak (lower frequency) is a repeat of the stronger peak
            top_two_acf_peak_diff = abs(acf_sqi[0] - acf_sqi[1])
            top_two_acf_freq_ratio_1 = acf_freqs[0] / acf_freqs[1]
            top_two_acf_freq_ratio_2 = acf_freqs[1] / acf_freqs[0]
            top_two_acf_freq_ratio = max(
                top_two_acf_freq_ratio_1, top_two_acf_freq_ratio_2
            )

            # One-toned condition
            if np.all(is_similar_heights) & np.all(is_common_divisors):
                f0 = max(acf_harmonics)
            # Two-toned condition; f1 is present
            elif (top_two_acf_peak_diff < 0.15) & (
                abs(2 - top_two_acf_freq_ratio) < 0.1
            ):
                f0 = max(acf_freqs[0], acf_freqs[1])
            # Three-toned condition - Scenario 3: peaks are roughly multiples of one another
            # f0, f1, and f2 are all present
            elif np.all(is_all_multiples):
                f0 = acf_harmonics[np.argmax(acf_sqi)]
            # Sometimes, unrelated peaks are erroneously selected
            else:
                f0 = psd_freq

    return f0

def autocorr_norm(signal, n_lags):
    """Computes normalized autocorrelation, based on Khreis 2020."""
    # NOTE: A nice programming challenge is to implement this operation without a
    # forloop, using clever indexing & matrix operations.
    c0 = np.var(signal)  # sample variance
    xbar = np.mean(signal)  # sample mean
    signal_norm = signal - xbar  # normalized signal
    N = len(signal)

    r = np.zeros(n_lags)
    for lg in range(min(n_lags, N)):
        r[lg] = (1 / (N - lg)) * sum((signal_norm[: N - lg]) * (signal_norm[lg:])) / c0

    return r


def peak_counting_analysis(
    rivs: dict[str, dict[str, np.ndarray]],
    fs: float,
    peak_prominence: float = 0.5,
    show: bool = False,
):
    """Estiamte RR by identifying peaks in the RIV waveforms.
    
    If breathing rate varies greatly in a time window, there may be multiple potential
    resp rate peaks in the power spectra. Nagy et al., agrees that time-domain
    analysis may be more appropriate in this case.
    
    TODO: analyze RR estimation accuracy of peak counting vs PSD analysis methods vs the 
    number of peaks found in the PSD spectra.
    
    References:
        Nagy, Á. et al., Continuous Camera-Based Premature-Infant Monitoring Algorithms 
        for NICU. Appl. Sci. 2021, 11, 7215. https://doi.org/10.3390/app11167215
    
    Args:
        rivs: Nested dictionary containing RIV waveforms, with at minimum the
            following values: rivs[RIV]['x'], rivs[RIV]['y']
        fs: sampling frequency (hz) of RIV waveforms, for calculating window size.
            (no longer used)
        peak_prominence: Passed to signal.find_peaks() for each RIV.
        show: If true, prints out the estiamted RRs. Defaults to False.

    Returns:
        An updated dictionary with keys:
            "pk_count RR num pks" - number of peaks present in the current window, 
                extrapolated to peaks per minute
            "pk_count RR median pk diff" - the median peak-to-peak interval in 
                breather per minute.
    """

    riv_copy = copy.deepcopy(rivs)
    for key in rivs.keys():
        pks, _ = signal.find_peaks(rivs[key]["y"], prominence=peak_prominence)
        n_detected_peaks = len(pks)
        if n_detected_peaks > 1:
            window_size = rivs[key]['x'][-1] - rivs[key]['x'][0]
            first_to_last_peak = rivs[key]["x"][pks][-1] - rivs[key]["x"][pks][0] 
            avg_window_size = (window_size + first_to_last_peak)/2
            peak_diffs = np.diff(rivs[key]["x"][pks])
            wavs_per_min = 60 / peak_diffs  # waves/minute
            median_wav_per_min = np.median(wavs_per_min)  # sec/wave
            riv_copy[key]["pk_count RR median pk diff"] = median_wav_per_min
            riv_copy[key]["pk_count RR num pks"] = n_detected_peaks * 60 / avg_window_size
            riv_copy[key]["std"] = np.std(wavs_per_min)
            riv_copy[key]["n detected peaks"] = n_detected_peaks
        else:
            riv_copy[key]["pk_count RR median pk diff"] = np.nan
            riv_copy[key]["pk_count RR num pks"] = np.nan
            riv_copy[key]["std"] = np.nan
            riv_copy[key]["n detected peaks"] = n_detected_peaks

    if show:
        print("prominence value:", peak_prominence)
        fig = make_subplots(
            rows=1,
            cols=1,
        )
        
        for idx, key in enumerate(rivs):
            fig.add_scatter(
                x=rivs[key]['x'],
                y=rivs[key]['y'],
                mode='lines',
                marker={"color": mpl.colors.rgb2hex(mpl.colormaps["tab10"](idx))},
                legendgrouptitle_text=key,
                legendgroup=key,
                name='',
                # visible="legendonly"
            )
            pks, _ = signal.find_peaks(rivs[key]["y"], prominence=peak_prominence)
            fig.add_scatter(
                x=rivs[key]['x'][pks],
                y=rivs[key]['y'][pks],
                mode='markers',
                marker={"color": mpl.colors.rgb2hex(mpl.colormaps["tab10"](idx))},
                legendgrouptitle_text=key,
                legendgroup=key,
                showlegend=False,
                name='',
                # visible="legendonly"
            )
            
        fig.show()
        
        rr_candidates_median_delta = [
            np.round(riv_copy[key]["pk_count RR median pk diff"], 2)
            for key in riv_copy
            if "pk_count RR num pks" in riv_copy[key].keys()
        ]
        rr_candidates_count = [
            np.round(riv_copy[key]["pk_count RR num pks"], 2)
            for key in riv_copy
            if "pk_count RR num pks" in riv_copy[key].keys()
        ]
        stds = [
            np.round(riv_copy[key]["std"], 4)
            for key in riv_copy
            if "pk_count RR num pks" in riv_copy[key].keys()
        ]
        n_peaks = [
            np.round(val["n detected peaks"], 4)
            for val in riv_copy.values()
            if "pk_count RR num pks" in val.keys()
        ]
        df_display = pd.DataFrame(
            {
                "feature": riv_copy.keys(),
                "rr candidate median delta": rr_candidates_median_delta,
                "rr_candidates_count": rr_candidates_count,
                "std": stds,
                "n peaks": n_peaks,
            }
        )
        display(df_display)

    return riv_copy


def kalman_analysis(
    rivs_psd: dict[dict],
    fs_riv: float,
    rr_seed: Optional[float] = np.nan,
    n_sig: int = 2,
    n_non_riv: int = 1,
    fmin: float = 0.13,
    show: bool = False,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Performs Kalman Filtering, then use psd_analysis to estimate respiratory rate.

    Kalman Filtering performs fusion of n_sig number of RIV signals with the highest
    PSD-based RQIs.

    See: Khreis, S., Ge, D., Rahman, H. A. & Carrault, G. Breathing
    Rate Estimation Using Kalman Smoother with Electrocardiogram and
    Photoplethysmogram. IEEE Trans. Biomed. Eng. 67, 893-904 (2020).

    Args:
        rivs_psd (dict[dict]): Nested dictionary with each RIV as keys, and each
            rivs_psd[RIV_key] has at least the following keys:
                "y" - the y-values of the interpolated and filtered RIV waveform
                "max PSD freq" - Frequency of the max PSD peak.
                "PSD RQI" - Quality index corresponding to the max PSD frequency.
                "freq candidates" - n-largest peaks from the PSD.
        fs_riv (float, hz): Sampling frequency of RIV signals.
        rr_seed (float): initial guess of the respiratory rate (breaths per min).
            Can usually use the RR estimate from the previous frame.
        n_sig (int): Number of waveforms to fuse. Defaults to 2.
        n_non_riv (int): Number of non-RIV signals in rivs_psd. Used to exlcude signals
            from consideration for kalman fusion. Default 1.
            e.g., the ppg waveform is also used to consider the RR frequency but 
            should not be used in kalman fusion.
        fmin (float): Minimum bound for respiratory frequency range to consider.
            Default 0.1 hz (6 breaths permin)
        show (bool, optional): Defaults to False.

    """
    # NOTE: if RQIs are *high*, kalman filtering may not necessary. Need further testing

    # sort feature based on RQIs & get top n_sig features with highest RQIs.
    psd_rqis = [rivs_psd[k]["PSD RQI"] for k in rivs_psd.keys()]
    features = list(rivs_psd.keys())
    sorted_features = [
        x
        for _, x in sorted(
            zip(psd_rqis[:-n_non_riv], features[:-n_non_riv]), reverse=True
        )
    ]

    # prepare parameters to estimate waveform parameters before kalman fusion
    sig = dict.fromkeys(np.arange(n_sig))
    for i in range(n_sig):
        sig[i] = dict.fromkeys(["wav", "theta", "A", "var", "b_k"])

    for i in range(n_sig):
        sig[i]["name"] = sorted_features[i]
        sig[i]["wav"] = rivs_psd[sorted_features[i]]["y"]
        sig[i]["fs"] = fs_riv
        if rr_seed is None:
            rr_seed = 0
            sig[i]["f"] = rivs_psd[sorted_features[i]]["max PSD freq"]
        else:
            # NOTE: Harmonics RRs were often selected when rr_seed isn't used
            f_estimate, nearest_idx = closest_to(
                arr=rivs_psd[sorted_features[i]]["freq candidates"] * 60,
                target=rr_seed,
            )
            sig[i]["f"] = f_estimate / 60

        (
            sig[i]["theta"],
            sig[i]["A"],
            sig[i]["var"],
            sig[i]["b_k"],
        ) = kalman.init_params(sig[i]["wav"], sig[i]["f"], sig[i]["fs"])

    y, ys = kalman.kalman_fusion(sig=sig)

    # ==================== Analyze fused waveform ====================
    # We use a combination of fused signal and its differences, which act like
    # high pass filters, because maybe the most prominant peak in low frequencies
    # is not the target. By using a combination of sequentially filtered signals,
    # we can find the peak that most strongly presists, which
    # we then use as the RR candidate.
    kalman_data = defaultdict(dict)
    kalman_data["kalman"] = {"y": normalize(y)}
    kalman_data["kalman d1"] = {"y": normalize(np.diff(y))}
    kalman_data["kalman d2"] = {"y": normalize(np.diff(np.diff(y)))}

    #! Assume this frame's RR is at most 60% away from the previous frame
    #! This is a key assumption that dramatically increases the accuracy
    #! of the estimations, and helps overcome low frequency noise in PPG signals.
    resp_freq_min = max(fmin, rr_seed / 60 * 0.6)
    kalman_data = psd_analysis(
        rivs=kalman_data,
        fs_riv=fs_riv,
        include_combined_spectra=True,
        resp_range_min_f=resp_freq_min,
        show=show,
    )

    # Clean, strong, prominant PSD peak after fusion
    if kalman_data["kalman"]["PSD RQI"] > 0.3:
        RR_kalman = kalman_data["kalman"]["psd RR candidate"]
        RR_kalman_f0 = kalman_data["kalman"]["psd RR candidate"]
        frame_quality  = kalman_data["kalman"]["PSD RQI"]
    else:
        
        # result of harmonic analysis
        RR_kalman_f0 = kalman_data["product of all PSD"]["psd RR candidate"]
        frame_quality = kalman_data["product of all PSD"]["PSD RQI"]

        # experiment 1: use the "closest to previous frame" method
        RR_kalman_nearest_psd, nearest_idx = closest_to(
            arr=kalman_data["product of all PSD"]["freq candidates"] * 60,
            target=rr_seed,
        )
        frame_quality = kalman_data["product of all PSD"]["freq candidates rel power"][
            nearest_idx
        ]

        # # experiment 2: count peaks!
        pks, _ = signal.find_peaks(standardize(y), prominence=0.5)
        wavelengths = np.diff(pks)
        if len(wavelengths) > 0:
            wavs_per_min = (60 * fs_riv) / wavelengths  # waves/minute
            median_wav_per_min = np.median(wavs_per_min)
            if show:
                print(
                    "median RR based on fused waveform pk time-delta:", median_wav_per_min
                )
                print("fused waveform peak count per min:", len(pks + 1) * 2)

        # merge the results
        if abs(RR_kalman_f0 * 2 - RR_kalman_nearest_psd) < 2:
            RR_kalman = RR_kalman_f0
        else:
            RR_kalman = kalman_data["product of all PSD"]["original psd RR candidate"]

        # In this case, the strongest "presistent" signal falls outside our estimated
        # respiratory range. We provide an RR with associated RQI=0
        if RR_kalman == 0:
            RR_kalman = np.median(
                [feat["psd RR candidate"] for feat in kalman_data.values()]
        )

    if show:
        print("kalman seed: ", rr_seed)
        print("harmonic result: ", RR_kalman_f0)
        print("RR kalman: ", RR_kalman)

        plt.plot(y, label="Kalman Filter output")
        # plt.plot(pks, y[pks], "o")
        # plt.plot(ys, label="Kalman Smoother output")
        for idx in range(n_sig):
            plt.plot(sig[idx]["wav"], label=sig[idx]["name"])
        plt.xlabel("samples")
        signals_used = [sig[i]["name"] for i in range(n_sig)]
        plt.title(f"Fusing signals: {signals_used}")
        plt.legend()

    return RR_kalman, frame_quality
