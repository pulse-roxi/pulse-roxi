"""Functions for estimating respiratory induced variations (RIVs) from a ppg signal."""
from collections import defaultdict
from typing import Literal, Tuple

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from IPython.display import display
from scipy import signal
from scipy.stats import kurtosis, skew

from ppg2rr import signal_quality as sqi
from ppg2rr import util
from ppg2rr.config import AlgorithmParams

DEFAULT_PARAMS = AlgorithmParams(dataset='kapiolani')

# NOTE: In the final production code, RIV extraction can be streamlined by doing:
# 1. Estimate heart rate
# 2. Find fiducial points
# 2.1 Estimate PPG pulse onsets
# 2.2 Extra singular PPG pulse
# 2.3 Identify fiducial points (max, min, dicrotic notch, max velocity)
# 2.4 Identify fiducial-related features (e.g., area under descending pulse)
# 3. Estimate RIVs uding fiducial points and features.

def find_extrema_HX2017(
    ppg: np.ndarray,
    fs: float,
    hr_bpm: float,
    win_tolerance_pct: float = 0.5,
    debug_flag: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrema search algorithm based on Hanyu & Xiaohui, 2017.

    Uses HR (heart rate, beats per minute) as window length. Looks for extrema within
    that window. The idea is that ppg signals should have one wave cycle per heart beat.
    This helps in distinguishing systolic peaks from and diastolic peaks.
    This may be a great option for calculating RIFVs.

    Args:
        ppg (array-like): The ppg signal to be analyzed.
        fs (float): The sampling frequency of the signal in Hz [samples/second].
        hr_bpm (float): The heart rate in [beats per minute].
        win_tolerance_pct (float): The scope of search range for the next
            peak/valley in the ppg waveform from the current peak/value.
            Defaults to 1/2. e.g.,

            next_estimated_peak =
                current_peak + window_len +/- win_tolerance_pct*window_len

            The smaller the respiratory sinus arrythmia, the smaller this percentage can
            be. [1] uses 1/3 but did not give justification. We may be able to use a
            smaller number (e.g., 1/4) since RSA is smaller in children. However, in the
            presence of motion artiact corruption, we may need to increase the search
            range due to the shifts in peaks or appearance of false peaks caused by
            motion artifacts.
        debug_flag (bool): If True, plots ppg with estimated min and max.

    Returns:
        max_idx_list, min_idx_list

    Hanyu and Xaiohui [1] estimates the HR by taking the autocorrelation of the first 3
    seconds of the window. Then:
    - estimate increment = period of the signal
    - lcoatie m1
    - search for m2 within m1+increment+/-increment/3
    - update increment = m2-m1

    [1] Hanyu, S. & Xiaohui, C. Motion artifact detection and reduction
        in PPG signals based on statistics analysis. in 2017 29th Chinese
        Control And Decision Conference (CCDC) 3114-3119 (IEEE, 2017).
        doi:10.1109/CCDC.2017.7979043

    """
    hr = hr_bpm / 60  # convert to Hz
    win_len = np.ceil(fs / hr).astype(int)  # samples per heart beat
    max_idx_list = _get_maxima_via_local_windows(
        ppg=ppg, win_len=win_len, win_tolerance_pct=win_tolerance_pct
    )
    min_idx_list = _get_maxima_via_local_windows(
        ppg=-ppg, win_len=win_len, win_tolerance_pct=win_tolerance_pct
    )

    if debug_flag:
        fig = px.line(ppg)
        fig.add_scatter(x=max_idx_list, y=ppg[max_idx_list], mode="markers")
        fig.add_scatter(x=min_idx_list, y=ppg[min_idx_list], mode="markers")
        fig.update_layout(
            title=f"Find Extrema, H&X2017, reference HR: {hr_bpm:.2f} bpm"
        )
        series_names = ["ppg", "max", "min"]
        for idx, name in enumerate(series_names):
            fig.data[idx].name = name
            fig.data[idx].hovertemplate = name
        fig.update_traces(showlegend=True)
        fig.update_yaxes(range=[-3, 3])
        # fig.update_xaxes(range=[0, 9000])

        fig.show()

    return max_idx_list, min_idx_list


def _get_maxima_via_local_windows(
    ppg: np.ndarray, win_len: int, win_tolerance_pct: float
) -> np.ndarray:
    """Wrapper function for _get_largest_local_maxima().
    
    Find the maxima within a local window within a signal.

    Args:
        ppg (np.ndarray): The entire ppg signal
        win_len (int): size of the local window to find the maxima
        win_tolerance_pct (float): local window start/end =
            next_max_idx_est + win_len +/- win_tolerance_pct*window_len

    Returns:
        array of maxima indices
    """
    max_idx_list = []

    # first max falls within one heart beat
    next_max_idx_est, max_idx = _get_largest_local_maxima(
        ppg,
        next_max_idx_est=win_len / 2,
        win_len=win_len,
        win_tolerance_pct=0.75,
    )
    max_idx_list.append(max_idx)

    # all other maxima
    while next_max_idx_est < len(ppg):
        next_max_idx_est, max_idx = _get_largest_local_maxima(
            ppg,
            next_max_idx_est,
            win_len,
            win_tolerance_pct,
        )
        max_idx_list.append(max_idx)

    max_idx_list = np.asarray(max_idx_list)
    return max_idx_list[~np.isnan(max_idx_list)].astype(int)


def _get_largest_local_maxima(
    ppg: np.ndarray, next_max_idx_est: int, win_len: int, win_tolerance_pct: float
):
    """Returns largest local maxima within a window of PPG signal.

    PPG can often have multiple peaks, corresponding to the systolic and
    diastolic peaks. Furthermore, noise and motion artifacts may create additional
    false peaks.

    Args:
        ppg (np.ndarray): The entire ppg signal
        next_max_idx_est (int): estimated index of the next maxima
        win_len (int): size of the local window to find the maxima
        win_tolerance_pct (float): local window start/end =
            next_max_idx_est + win_len +/- win_tolerance_pct*window_len

    Returns:
        next_max_idx_est, max_idx
    """
    win_start = np.ceil(next_max_idx_est - win_len * win_tolerance_pct).astype(int)
    win_end = np.ceil(next_max_idx_est + win_len * win_tolerance_pct).astype(int)

    # cap window start and end to the duration of the signal
    win_start = max(0, win_start)
    win_end = min(len(ppg), win_end)

    local_ppg_window = ppg[win_start:win_end]
    local_max_idx_arr, _ = signal.find_peaks(local_ppg_window)
    local_extrema = local_ppg_window[local_max_idx_arr]

    # find current maxima index & update next_max_idx_est
    if ~np.any(local_max_idx_arr):
        next_max_idx_est = win_start + win_len
        return next_max_idx_est, np.NaN
    else:
        largest_local_max_idx = local_max_idx_arr[np.argmax(local_extrema)]
        max_idx = win_start + largest_local_max_idx
        next_max_idx_est = max_idx + win_len
        return next_max_idx_est, max_idx


def one_x_per_y(signal: np.ndarray, x_locs: np.ndarray, y_locs: np.ndarray, keep: str):
    """Makes sure there's only one x index per y index, removing extra x-indices.

    x_locs and y_locs are indices of features X and Y of a signal (e.g., peaks and
    troughs), and X and Y should interleave with each other. For example if outliers
    are removed from feature indices then X and Y may not have a 1-to-1 correspondence.
    This function removes extra X features to make sure one feature X corresponds to
    one feature Y.

    If multiple feature X's are foud between two feature Y's, the largest feature X
    is retained and other are removed.

    Args:
        signal: signal with features X and Y
        x_locs: indices of feature X
        y_locs: indices of feature Y
        keep: a string indicating the rule for feature selection. If features X and Y
            correspond to maxima and minima, then set `keep=max`. If features X and Y
            correspond to minima and maxima, then set `keep=min`.

    Suggested usage:
        Calling one_x_per_y() twice, switching the x- and y- locs each time, will
        ensure a 1-to-1 correspondence between feature X and feature Y; for example,
        x_locs_paired, _ = one_x_per_y(
            signal=y, x_locs=x_locs, y_locs=y_locs,keep='max
            )
        y_locs_paired, _ = one_x_per_y(
            signal=y, x_locs=y_locs, y_locs=x_locs_paired,keep='min'
            )

    """
    assert len(x_locs) > 0, "length of x locs is 0"
    assert len(y_locs) > 0, "length of y locs is 0"

    feature_x = signal[x_locs]
    feature_y = signal[y_locs]

    x_locs_to_remove = np.zeros(len(feature_x), dtype=bool)

    offset = 1  # does max occur before min?
    if x_locs[0] > y_locs[0]:
        offset = 0

    # 1st fencepost; If there are multiple feature_x prior to
    # the first feature_y, only keep the largest one
    if offset:
        current_idx = y_locs[0]  # current feature_y

        # take the highest peak prior to current feature_y
        # remove any other local feature_x before the current feature_y
        if keep == "max":
            max_feat_x = np.max(feature_x[x_locs < current_idx])
            x_locs_to_remove[(x_locs < current_idx) & (feature_x < max_feat_x)] = True
        elif keep == "min":
            min_feat_x = np.min(feature_x[x_locs < current_idx])
            x_locs_to_remove[(x_locs < current_idx) & (feature_x > min_feat_x)] = True

    for i in range(1, len(feature_y)):
        prev_idx = y_locs[i - 1]
        current_idx = y_locs[i]
        # choose x_locs to keep; select the largest feature_x prior to current feature_y
        xs_between_ys = (x_locs > prev_idx) & (x_locs < current_idx)

        if ~np.any(xs_between_ys):
            continue
        if np.sum(xs_between_ys) > 1:
            # if x_features have the same values between 2 y's, keep the first one?
            x_locs_to_remove[np.where(xs_between_ys)[0][1:]] = True
        else:
            if keep == "max":
                max_feat_x = np.max(feature_x[xs_between_ys])
                x_locs_to_remove[xs_between_ys & (feature_x < max_feat_x)] = True
            elif keep == "min":
                min_feat_x = np.min(feature_x[xs_between_ys])
                x_locs_to_remove[xs_between_ys & (feature_x > min_feat_x)] = True

    # if there are peaks after the final min
    if x_locs[-1] > y_locs[-1]:
        current_idx = y_locs[-1]
        if keep == "max":
            max_feat_x = np.max(feature_x[x_locs > current_idx])
            x_locs_to_remove[(x_locs > current_idx) & (feature_x < max_feat_x)] = True
        elif keep == "min":
            min_feat_x = np.min(feature_x[x_locs > current_idx])
            x_locs_to_remove[(x_locs > current_idx) & (feature_x > min_feat_x)] = True

    # update remaining feature_x
    feature_x = feature_x[~x_locs_to_remove]
    x_locs = x_locs[~x_locs_to_remove]

    return x_locs, feature_x


def est_RIFV(max_idxs: np.ndarray, min_idxs: np.ndarray):
    """Estimates the RIFV of a ppg signal, and remove outliers.

    RIFVs are based on the time-difference between successive PPG peak occurrances and
    PPG trough occurrances.

    Charlton et al. An assessment of algorithms to estimate respiratory rate
    from the electrocardiogram and photoplethysmogram. Physiol. Meas. 37,
    610-626 (2016).

    Args:
        max_idxs (np.ndarray): The indices of the maxima of the ppg signal.
        min_idxs (np.ndarray): The indices of the minima of the ppg signal.

    Returns:
        rifv_max: The estimated RIFV based on the maxima of the ppg signal.
        rifv_max_idx: The indices corresponding to rifv_max.
        rifv_min: The estimated RIFV based on the minima of the ppg signal.
        rifv_min_idx: The indices corresponding to rifv_min.

    """
    # min deviation tolerance for RIFV, heuristically determined
    rifv_max = np.diff(max_idxs)
    rifv_min = np.diff(min_idxs)

    rifv_max_idx = max_idxs[1:]
    rifv_min_idx = min_idxs[1:]

    return rifv_max, rifv_max_idx, rifv_min, rifv_min_idx


def est_RIFV_mas(ppg: np.ndarray, min_idx: np.ndarray, max_idx: np.ndarray):
    """Calculate RIFV based on maximal ascending slope of PPG pulses.

    In detecting FM of PPG signal, the maximum slope point has been proven to be more
    reliable for measuring RF than peak or valley point which is prone to non-trivial
    error due to common artifacts in the waveforms and wave reflection interference
    (Escobar and Torres 2014)

    This calculation is done in two steps, first identify individual pulses. Then
    calculate the slope by taking the difference between consecutive points. We ignore
    the detrending procedure since it has no effect on the RIFV calculation.
    We define the a "slope" as the segment from one min to the next max

    References:
    - Liu, H., Chen, F., Hartmann, V., Khalid, S. G., Hughes, S. & Zheng, D.
        Comparison of different modulations of photoplethysmography in extracting
        respiratory rate: From a physiological perspective. Physiol. Meas. 41, (2020).
    - Escobar B and Torres R 2014 Feasibility ofnon-invasive blood pressure estimation
        based on pulse arrival time: a MIMIC database study Computing in Cardiology 
        2014 pp 1113-6

    Args:
        ppg (np.ndarray): The ppg signal.
        min_idx (np.ndarray): The indices of the minima of the ppg signal.
        max_idx (np.ndarray): The indices of the maxima of the ppg signal.

    Returns:
        rifv_mas (np.ndarray): The estimated RIFV based on the max ascending slope
        rifv_mas_idx (np.ndarray): The indices of the max ascending slopes in relation
            to the given ppg signal.

    # TODO: unit test
    """
    start_with_max = max_idx[0] < min_idx[0]  # does max occur before min?

    max_slope_idx_list = []

    for idx in range(len(max_idx) - start_with_max):
        pulse_start_idx = min_idx[idx]
        pulse_end_idx = max_idx[idx + start_with_max]
        segment = ppg[pulse_start_idx:pulse_end_idx]
        # differentiate and prepend 0, then find max slope index
        max_slope_idx = np.argmax(np.insert(np.diff(segment), 0, 0))
        max_slope_idx_list.append(max_slope_idx + pulse_start_idx)

    rifv_mas = np.diff(np.asarray(max_slope_idx_list))
    rifv_mas_idx = max_slope_idx_list[1:]

    return rifv_mas, np.asarray(rifv_mas_idx)


def est_AUDP(ppg: np.ndarray, max_idxs: np.ndarray, min_idxs: np.ndarray):
    """Calculate the area under descending ppg pulse (AUDP), described in Cernat 2015.

    Cernet, 2015 describes three features - audp area under the entire ppg pulse (AUP),
    area under ascending ppg pulse (AUAP), and area under descending ppg pulse (AUDP).

    Reference:
        Cernat et al., Recording system and data fusion algorithm for enhancing
        the estimation of the respiratory rate from photoplethysmogram. Proc.
        Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. EMBS 2015-November, 5977-5980
        (2015).

    Args:
        ppg (np.ndarray): The ppg signal.
        max_idxs (np.ndarray): The indices of the maxima of the ppg signal.
        min_idxs (np.ndarray): The indices of the minima of the ppg signal.
    """
    start_with_min = max_idxs[0] > min_idxs[0]  # does max occur before min?
    end_with_max = max_idxs[-1] > min_idxs[-1]

    audp = np.zeros(len(max_idxs) - end_with_max)
    for i in range(len(max_idxs) - end_with_max):
        audp[i] = np.sum(ppg[max_idxs[i] : min_idxs[i + start_with_min]])

    if end_with_max:
        audp_idx = max_idxs[:-1]
    else:
        audp_idx = max_idxs

    return audp, audp_idx

def est_RIAV(
    ppg: np.ndarray,
    max_idx: np.ndarray,
    min_idx: np.ndarray,
):
    """Get the RIAV for a given PPG pulse.

    RIAV is the amplitude (max -> min) or (min -> max) of a PPG pulse.
    
    Returns two sets of values and indices, corresponding to the amplitude of the
    ascending and descending ppg pulses.

    Args:
        ppg (np.ndarray): PPG signal to be analyzed
        max_idx (np.ndarray): indices of the ppg maxima
        min_idx (np.ndarray): indices of the ppg minima

    Returns:
        riav_ascending, riav_ascending_idxs, riav_descending, riav_descending_idxs
    """
    maxs = ppg[max_idx]
    mins = ppg[min_idx]

    start_with_max = int(max_idx[0] < min_idx[0])
    start_with_min = int(max_idx[0] > min_idx[0])
    end_with_max = int(max_idx[-1] > min_idx[-1])
    end_with_min = int(max_idx[-1] < min_idx[-1])

    # max to the next min
    riav_descending = maxs[: len(maxs) - end_with_max] - mins[start_with_min:]
    riav_descending_idxs = max_idx[: len(maxs) - end_with_max]
    # min to the next max
    riav_ascending = maxs[start_with_max:] - mins[: len(mins) - end_with_min]
    riav_ascending_idxs = min_idx[: len(mins) - end_with_min]

    return riav_ascending, riav_ascending_idxs, riav_descending, riav_descending_idxs


def detect_outliers(
    y: np.array, min_deviation_tolerance: float, outlier_tolerance: float = 3
) -> np.ndarray:
    """Mark outliers from a given list, based on mean absolute deviation.

    Based on Hampler filtering, where by default, an outlier is a value that
    is more than three scaled median absolute deviations (MAD) from the median.
    The scaled MAD is defined as c*median(abs(A-median(A))), where
    c=-1/(sqrt(2)*erfcinv(3/2)). Source:
    https://www.mathworks.com/help/matlab/ref/rmoutliers.html

    Args:
        y (np.array): The list of values to be filtered.
        min_deviation_tolerance (float): The minimum deviation allowed for a value away
            from the median for points to be labeled as outliers.
        outlier_tolerance (float): scales the MAD; a larger tolerance allows more values
            to be considered as non-outliers. Default value is 3.

    Returns:
        outliers: boolean array marking where outliers were found in the original
            input.
    """
    # constant for median amplitude deviation, as defined in MATLAB docs
    c_mad = 1.4826  # -1/(sqrt(2)*erfcinv(3/2))

    median_abs_deviation = max(
        min_deviation_tolerance,
        c_mad * outlier_tolerance * np.median(np.abs(y - np.median(y))),
    )

    outliers = (y > (np.median(y) + median_abs_deviation)) | (
        y < (np.median(y) - median_abs_deviation)
    )

    return outliers


def get_dicrotic_notch(pulse: npt.NDArray = None, show: bool = False):
    """Get dicrotic notch of a PPG pulse by using the acceleration peak.

    Under normal circumstances, the dicrotic notch is the 2nd highest peak in the
    acceleration ppg pulse. If PPG signal is not ideal, using the 1st peak after
    the 1st VPG peak or the PPG peak may be more reliable.
    
    If the APG doesn't have a peak after the PPG pulse peak, then try taking the 2nd
    derivative of APG and use its peak as the dicrotic notch location.

    TODO: test with stafan's signal and kapiolani waveforms.
    TODO: Discuss with stefan

    Args:
        pulse: ppg pulse
        
    Returns:
        index position of the dicrotic notch on the given pulse

    References: 
    - Fine, J., Branan, K. L., Rodriguez, A. J., Boonya-Ananta, T., Ajmal,
        Ramella-Roman, J. C., McShane, M. J., & Coté, G. L. (2021). Sources of 
        inaccuracy in photoplethysmography for continuous cardiovascular monitoring.
        In Biosensors (Vol. 11, Issue 4). https://doi.org/10.3390/bios11040126
    - Elgendi, M. (2012). On the Analysis of Fingertip Photoplethysmogram Signals. 
        Current Cardiology Reviews, 8(1), 14-25. 
        https://doi.org/10.2174/157340312801215782
    - Suboh, M. Z., Jaafar, R., Nayan, N. A., Harun, N. H., & Mohamad, M. S. F. (2022). 
        Analysis on Four Derivative Waveforms of Photoplethysmogram (PPG) for Fiducial 
        Point Detection. Frontiers in Public Health, 10(June), 1–12. 
        https://doi.org/10.3389/fpubh.2022.920946
    """
    vpg_pulse = np.diff(pulse)
    apg_pulse = np.diff(vpg_pulse)

    local_veloc_maxima = signal.argrelmax(vpg_pulse)[0]
    local_accel_maxima = signal.argrelmax(apg_pulse)[0]
    
    # acceleration peak after the PPG pulse peak
    ppg_peak_idx = np.argmax(pulse)
    notch_candidate_idx = local_accel_maxima[local_accel_maxima > ppg_peak_idx]
    if len(notch_candidate_idx) == 0:
        enhance_apg = -np.diff(np.diff(apg_pulse))
        enhance_apg_maxima = signal.argrelmax(enhance_apg)[0]
        enhance_notch_candidate_idx = enhance_apg_maxima[enhance_apg_maxima > ppg_peak_idx]
        if len(enhance_notch_candidate_idx) == 0:
            notch_idx = None
        else:
            notch_idx = enhance_notch_candidate_idx[0]+4
    else:
        notch_idx = notch_candidate_idx[0]+2
    
    if show:
        t = np.arange(len(pulse))
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[.99, 0.01],  # relative heights of each row
        )
        fig.add_scatter(x=t,y=pulse/max(pulse),name='PPG', row=1, col=1)
        fig.add_scatter(x=t[1:],y=vpg_pulse/max(vpg_pulse),name='VPG', row=1, col=1)
        fig.add_scatter(x=t[2:],y=apg_pulse/max(apg_pulse),name='APG', row=1, col=1)
        fig.add_vline(x=notch_idx, line_dash="dash", row=1, col=1)
        fig.show()
        
    return notch_idx


def analyse_pulse_features(ppg, min_idxs, show: bool=False):
    """Find the fiducial points in a ppg pulse and derive features.
    Fiducial points identified:
    * pulse start
    * pulse end
    * systolic peak
    * diastolic peak.
    * dicrotic notch amplitude - Finnegan et al., (2023). Features from the 
        photoplethysmogram and the electrocardiogram for estimating changes in blood 
        pressure. Scientific Reports, 13(1), 1–20. 
        https://doi.org/10.1038/s41598-022-27170-2
    * Slope transit time - Addison, P. S. (2016). Slope Transit Time (STT): A Pulse
        Transit Time Proxy requiring only a Single Signal Fiducial Point. IEEE 
        Transactions on Biomedical Engineering, 63(11), 2441–2444. 
        https://doi.org/10.1109/TBME.2016.2528507
    * Inflection point areas - Wang et al, (2009). Noninvasive cardiac output 
        estimation using a novel photoplethysmogram index. Proceedings of the 31st 
        Annual International Conference of the IEEE Engineering in Medicine and
        Biology Society: Engineering the Future of Biomedicine, EMBC 2009, 1746–1749. 
        https://doi.org/10.1109/IEMBS.2009.5333091

    This is only currently used to find the dicrotic notch, STT, and inflection point 
    area. However, we can probably streamline the entire RIV estimation operation
    by using this function to also find all other fiducial points.

    Args:
        ppg: signal to be analyzed
        min_idx: indices of minima are used as start and end indices of each pulse.
        
    Returns:
        (max_slope_idxs, slope_transit_times, pulse_max_idx, norm_audp), 
        (d_notch_idxs, notch_area_ratios, d_notch_rel_amp)
    TODO: unit test
        * correctly id notch if possible
        * returns None if no notch can be identified
    """
    pulse_starts = min_idxs[:-1]
    pulse_ends = min_idxs[1:]

    systolic_peak_idxs = []
    max_slope_idxs = []
    max_slopes = []
    audp = []
    norm_audp = []
    d_notch_idxs = []
    notch_area_ratios = []
    pulse_normalized_ppg = []
    slope_transit_times =[]
    # fiducial points of each pulse
    for pulse_start, pulse_end in zip(pulse_starts, pulse_ends):
        pulse = ppg[pulse_start:pulse_end]
        vpg_pulse = np.diff(pulse)

        norm_pulse = util.normalize(pulse)
        pulse_normalized_ppg.extend(norm_pulse)
        systolic_peak_idx = np.argmax(pulse)
        systolic_peak_idxs.append(systolic_peak_idx+pulse_start)

        # Max ascending slope
        max_slope_idxs.append(np.argmax(vpg_pulse)+pulse_start+1)
        max_slopes.append(np.max(vpg_pulse))

        # Slope Transit time; slope = rise/stt; normalized stt = 1/slope
        slope_transit_times.append( 1 / max_slopes[-1])
        
        # Descending pulse area
        audp.append(sum(pulse[systolic_peak_idx:pulse_end]))
        norm_audp.append(sum(norm_pulse[systolic_peak_idx:pulse_end]))

        # Dicrotic notch locations
        # NOTE: It may be impossible to identify dicrotic notch location. 
        # In that case, we will need to discard this information.
        d_notch_idx = get_dicrotic_notch(norm_pulse)
        if d_notch_idx is not None:
            d_notch_idxs.append(d_notch_idx+pulse_start)
        
            # Areas before and after dicrotic notch; inflection point area
            s1 = sum(norm_pulse[:d_notch_idx])
            s2 = sum(norm_pulse[d_notch_idx:])
            notch_area_ratios.append(s2/s1)

    
    # Relative dicrotic notch amplitudes
    pulse_normalized_ppg = np.array(pulse_normalized_ppg)
    # threshold of 10 is chosen arbitrarily
    if len(d_notch_idxs) > (len(pulse_starts)-10):
        d_notch_idxs = np.array(d_notch_idxs)-pulse_starts[0]
        d_notch_rel_amp = pulse_normalized_ppg[d_notch_idxs]
    else:
        d_notch_rel_amp = None
        notch_area_ratios = None
    
    # # Frequency Variations (i.e., Respiraotry Sinua Arrythmia)
    # np.diff(max_slope_idxs) # RIFV_mas
    # np.diff(systolic_peak_idxs) # RIFV_max
    # np.diff(min_idxs) # RIFV_min

    # # Itensity Variations (Baseline Wander)
    # riiv_min_x = min_idxs
    # riiv_min_y = ppg[riiv_min_x]
    # riiv_max_x = np.array(systolic_peak_idxs)
    # riiv_max_y = ppg[riiv_max_x]

    # # Amplitude Variations
    # systolic_peaks = riiv_max_y
    # systolic_peaks - ppg[pulse_starts]  # RIAV_ascending
    # systolic_peaks - ppg[pulse_ends]  # RIAV_descending
    
    if show:
        notch_color = "#ff5c0a"
        vpg = np.diff(ppg)
        apg = np.diff(vpg)
        t = np.arange(0,len(ppg))
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.33, 0.33, .33],  # relative heights of each row
        )
        fig.add_scatter(x=t,y=ppg/max(ppg),name='PPG', row=1, col=1)
        fig.add_scatter(x=t[1:],y=vpg/max(vpg),name='VPG', row=1, col=1)
        fig.add_scatter(x=t[2:],y=apg/max(apg),name='APG', row=1, col=1)
        fig.add_scatter(x=systolic_peak_idxs,y=ppg[systolic_peak_idxs]/max(ppg),name='peaks', mode='markers',row=1, col=1)
        fig.add_scatter(x=d_notch_idxs+pulse_starts[0], 
                        y=ppg[d_notch_idxs+pulse_starts[0]]/max(ppg),
                        name='dicrotic notch', 
                        mode='markers', 
                        line={'color':notch_color},
                        row=1, col=1)
                        
        fig.add_scatter(x=t,y=pulse_normalized_ppg,name='PPG', row=2, col=1)
        fig.add_scatter(x=pulse_starts-pulse_starts[0], y=np.zeros(len(pulse_starts)-1), mode='markers', row=2, col=1)
        
        if d_notch_rel_amp is not None:
            fig.add_scatter(x=d_notch_idxs, y=d_notch_rel_amp,
                            name='dicrotic notch', mode='markers', line={'color':notch_color}, row=2, col=1)
            fig.add_scatter(x=d_notch_idxs, y=util.normalize(d_notch_rel_amp),
                            name='dicrotic notch amplitude', mode='lines', line={'color':notch_color},row=3, col=1)
            fig.add_scatter(x=d_notch_idxs, y=util.normalize(notch_area_ratios),
                            name='notch area ratio', mode='lines',row=3, col=1)
        fig.add_scatter(x=max_slope_idxs, y=util.normalize(slope_transit_times),
                        name='slope transit time', mode='lines',row=3, col=1)
        fig.add_scatter(x=systolic_peak_idxs, y=util.normalize(audp),
                        name='AUDP', mode='lines',row=3, col=1)
        fig.add_scatter(x=systolic_peak_idxs, y=util.normalize(norm_audp),
                        name='AUDNP', mode='lines',row=3, col=1)
        fig.update_layout(height=800)
        fig.show()
        
    return (max_slope_idxs, slope_transit_times, systolic_peak_idxs, norm_audp), (d_notch_idxs, notch_area_ratios, d_notch_rel_amp)


def extract_ppg_features(
    ppg: np.ndarray[float],
    fs: float,
    hr: float,
    debug_flag: bool = False,
):
    """Extract PPG signal features.

    Features extracted:
    * Frequency variations - time difference (samples) between:
        * RIFV_max, PPG peaks
        * RIFV_min, PPG troughs
        * RIFV_mas, max ascending slope of PPG pulses
    * Area under descending pulse (AUDP)
    * Intensity variations - baseline wander calculated using:
        * RIIV_max - PPG upper envelope
        * RIIV_min - PPG lower envelope
    * Amplitude variations - PPG pulse amplitudes based on:
        * RIAV_ascending - PPG pulse min -> PPG pulse max
        * RIAV_descending - PPG pulse max -> PPG pulse min

    Args:
        ppg (np.ndarray[float]): ppg waveform, sampled at fs hz.
        fs (float): sampling frequency in hertz.
        hr (float): heart rate in beats per minute.
        debug_flag (bool, optional): _description_. Defaults to False.

    Returns:
        nested dictionary containing keys:
            ppg_feature
                x - array indices of ppg features
                y - values of ppg features
                outlier - boolean array indicating possible outlier values
    """
    # ================ Identify PPG pulses ================
    max_idx_raw, min_idx_raw = find_extrema_HX2017(ppg, fs, hr, debug_flag=debug_flag)

    # ================ Remove ripples from filtering ================
    # remove peaks too close to 0 - filtering will cause zero-signal segments
    # to fluctuate about 0 amplitude. These are artificial peaks
    max_idx_raw = max_idx_raw[np.abs(ppg[max_idx_raw]) > 1e-10]
    min_idx_raw = min_idx_raw[np.abs(ppg[min_idx_raw]) > 1e-10]

    # In order to identify various PPG pulses, there needs to be one maximum
    # that corresponds to each minimum.
    # NOTE: the following operations can be ignored by doing all analysis based on 
    # individual pulses
    (max_idx, _) = one_x_per_y(
        signal=ppg, x_locs=max_idx_raw, y_locs=min_idx_raw, keep="max"
    )
    (min_idx, _) = one_x_per_y(
        signal=ppg, x_locs=min_idx_raw, y_locs=max_idx, keep="min"
    )

    # ================== PPG pulse features ==================
    ppg_feat = defaultdict(dict)

    # RIFV are not as susceptible to ppg min/max outliers
    # Estimate RIFV - frequency modulation of PPG pulse
    (
        ppg_feat["RIFV_max"]["y"],
        ppg_feat["RIFV_max"]["x"],
        ppg_feat["RIFV_min"]["y"],
        ppg_feat["RIFV_min"]["x"],
    ) = est_RIFV(max_idx, min_idx)

    # Estimate RIFV with max ascending slope of PPG pulse
    ppg_feat["RIFV_mas"]["y"], ppg_feat["RIFV_mas"]["x"] = est_RIFV_mas(
        ppg=ppg,
        min_idx=min_idx,
        max_idx=max_idx,
    )

    # AUDP - area under descending PPG pulse
    ppg_feat["AUDP"]["y"], ppg_feat["AUDP"]["x"] = est_AUDP(
        ppg=ppg,
        max_idxs=max_idx,
        min_idxs=min_idx,
    )

    # RIIV - upper and lower envelopes of PPG (baseline wander)
    ppg_feat["RIIV_upper"]["y"] = ppg[max_idx_raw]
    ppg_feat["RIIV_upper"]["x"] = max_idx_raw
    ppg_feat["RIIV_lower"]["y"] = ppg[min_idx_raw]
    ppg_feat["RIIV_lower"]["x"] = min_idx_raw

    # RIAV - peak-trough amplitudes of PPG
    # amp_decending, amp_ascending, bad_max, bad_min = est_RIAV_old(y=ppg, max_idx_all=max_idx_raw, min_idx_all=min_idx_raw) # noqa E501
    (
        ppg_feat["RIAV_ascending"]["y"],
        ppg_feat["RIAV_ascending"]["x"],
        ppg_feat["RIAV_descending"]["y"],
        ppg_feat["RIAV_descending"]["x"],
    ) = est_RIAV(
        ppg=ppg,
        max_idx=max_idx,
        min_idx=min_idx,
    )
    
    # experimental function to identify PPG pulse features
    # NOTE: notch-related features may not be identifiable. Excluding these features
    # minimally impacts the algorithm performance.
    ((max_slope_idxs, slope_transit_times, pulse_max_idx, norm_audp), 
    (d_notch_idxs, notch_area_ratios, d_notch_rel_amp)) = analyse_pulse_features(ppg, min_idx)
    ppg_feat['STT']['x'] = np.array(max_slope_idxs)
    ppg_feat['STT']['y'] = np.array(slope_transit_times)
    # ppg_feat['norm AUDP']['x'] = np.array(pulse_max_idx)
    # ppg_feat['norm AUDP']['y'] = np.array(norm_audp)
    if (d_notch_idxs is not None) & (notch_area_ratios is not None):
        ppg_feat['notch area ratio']['x'] = np.array(d_notch_idxs)
        ppg_feat['notch area ratio']['y'] = np.array(notch_area_ratios)
        ppg_feat['notch rel amp']['x'] = np.array(d_notch_idxs)
        ppg_feat['notch rel amp']['y'] = np.array(d_notch_rel_amp)

    return ppg_feat


def resample_riv(t: np.ndarray, y: np.ndarray, fs: float, fs_riv: float):
    """Wrapper around scipy's signal.resample function.

    Args:
        t: Indices of the uniformly sampled signal.
        y: A uniformly sampled signal.
        fs: Sampling frequency of the uniformly sampled signal.
        fs_riv: Desired RIV sampling frequency.

    Returns:
        t_resampled - Time indices of the resampled signal
        y_resampled - Resampled signal
    """
    num_samples = int(len(y) / fs * fs_riv)
    y_resampled = signal.resample(y, num_samples)
    # y_resampled = resample_by_interpolation(y, input_fs=fs, output_fs=fs_riv)
    t_resampled = np.linspace(
        t[0] / fs, (len(y) + t[0]) / fs, len(y_resampled), endpoint=False
    )
    return t_resampled, y_resampled


def resample_by_interpolation(signal, input_fs, output_fs):
    """Resample a signal by interpolation to address the edge effects.

    Edge effect example - https://stackoverflow.com/q/51420923

    DISCLAIMER: This function is copied from
    https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, which was released
    under LGPL. - https://stackoverflow.com/a/52702937

    """
    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal


def cut_riv_outliers(
    riv_x: np.ndarray[int],
    riv_y: np.ndarray[float],
    signal_len: int,
    min_keep_pct: float,
    outlier_groups: list[tuple[int, int]],
    min_group_duration: int = 0,
):
    """Auxiliary function to cut individual RIV signals based on all RIV outliers.

    Possible scenarios:
    1. artifacts at the front
    2. artifacts at the back
    3. artifacts at front and back
    4. artifacts in the middle
    5. artifacts all over.

    Approach:
    * Identify segments of the given RIV where there are not outliers
    * Only keep segments where segment length > min_keep_pct*signal_len

    Args:
        riv_x: array of x-indices of a signal
        riv_y: array of y-indices of a signal
        signal_len: overall length of the signal
        min_keep_pct: minimum percentage of frame to be kept. Segments shorter than 
            this are not kept. Between 0 and 1.
        outlier_groups: list of tuples containing (outlier_start, outlier_end) indices
        min_group_duration: [samples] If artifact groups have less than this duration,
            then they are not considered for removal
    """
    if min_keep_pct > 1:
        raise ValueError("min_keep_pct: Please provide a percentage between 0 and 1")

    artifact_in_center = np.any((riv_x >= signal_len/4) & (riv_x <= 3*signal_len/4))

    # exclude groups that are too small
    valid_outlier_groups = []
    for outlier_start, outlier_end in outlier_groups:
        if outlier_end - outlier_start > min_group_duration:
            valid_outlier_groups.append((outlier_start, outlier_end))

    # from artifact groups -> get "good" ppg segments
    non_outlier_segment_starts = [0]
    non_outlier_segment_ends = []
    for group in valid_outlier_groups:
        non_outlier_segment_ends.append(group[0])
        non_outlier_segment_starts.append(group[1] + 1)
    non_outlier_segment_ends.append(signal_len)

    # find indices that fall in non-outlier segments
    keep_these_x, keep_these_y = [], []
    min_keep_len = min_keep_pct * signal_len
    for segment_start, segment_end in zip(
        non_outlier_segment_starts, non_outlier_segment_ends, strict=True
    ):
        segment_dur = segment_end - segment_start
        if segment_dur >= min_keep_len:
            in_non_outlier_segment = (riv_x >= segment_start) & (riv_x <= segment_end)
            keep_these_x.extend(riv_x[in_non_outlier_segment])
            keep_these_y.extend(riv_y[in_non_outlier_segment])

    if not keep_these_x:
        artifact_in_center = True
        return riv_x, riv_y, artifact_in_center
    else:
        return keep_these_x, keep_these_y, artifact_in_center

def is_outliers_everywhere(ppg_len, outlier_idxs):
    """Returns true if outliers are found in all 4 quarters of the PPG signal."""
    q_dur = ppg_len/4
    outliers_in_quartile = []
    for q in [0,1,2,3]:
        outliers_in_quartile.append(
            (
                (outlier_idxs > q_dur*q) &
                (outlier_idxs < q_dur*(q+1))
            ).sum()
        )
    if np.all(np.array(outliers_in_quartile) > 0): 
        outliers_everywhere = True
    else:
        outliers_everywhere = False
        
    return outliers_everywhere

def extract_rivs(
    ppg: np.ndarray[float],
    fs: float,
    hr: float,
    fs_riv: float = 10,
    rr_min: float = 5,
    rr_max: float = 78,
    outlier_tolerance: float = 3,
    min_deviation_tolerance: dict[str,float] = DEFAULT_PARAMS.min_deviation_tolerance,
    min_keep_pct: float = 0.5,
    remove_outliers: Literal["segment-wise", "point-wise"] = "segment-wise",
    show: bool = False,
):
    """Extract respiratory induced variations (RIVs) from the given PPG signal.

    RIVs are estimated from PPG signal features, which are:
        RIFV_max - frequency modulations, calculated using PPG peaks
        RIFV_min - frequency modulations, calculated using PPG troughs
        RIFV_mas - frequency modulations, based on max ascending slope of PPG pulses
        AUDP - area under descending ppg pulse
        RIIV_upper - baseline wander based on upper PPG envelope
        RIIV_lower - baseline wander based on lower PPG envelope
        RIAV_ascending - amplitude modulation based on ascending PPG pulse amplitude
        RIAV_descending - amplitude modulation based on descending PPG pulse amplitude

    The returned RIVs are resampled from unevenly sampled signals into signals sampled
    at fs_riv (hz).

    Args:
        ppg: An array representing the PPG signal
        fs: Sampling frequency of the PPG signal in hertz.
        hr: Estimated heart rate of the PPG signal in beats per minute.
        fs_riv: Desired sampling frequency of the RIVs. Defaults to 10.
        rr_min: Lower bound of possible resp rate, used for filtering the RIVs.
            Defaults to 5.
        rr_max: Lower bound of possible resp rate, used for filtering the RIVs.
            Defaults to 78.
        outlier_tolerance: Points are marked as outliers if they're larger than
            abs(outlier_tolance * median). Defaults to 3.
        min_deviation_tolerance (dict[str, float]): threshold used for outlier
            detection. Deviations smaller than this value are not considered outliers.
        min_keep_pct: Minimum percentage of frame to be kept. Segments shorter than 
            this are not kept. Between 0 and 1. Default: 0.5 (50%)
        remove_outliers (str, optional): Removes RIV outliers prior to interpolation
            and resampling. Either "point-wise" or "segment-wise" (default).
        show: Defaults to False.

    Returns:
        riv: a nested dictionary with one key for each RIV, interpolated and resampled,
            and their corresponding x and y values.
        ppg_feat: raw ppg features with their x (index) and y values.
    """
    # extract ppg features
    ppg_feat = extract_ppg_features(
        ppg=ppg,
        fs=fs,
        hr=hr,
        debug_flag=show,
    )

    # ============= Outlier Analysis: Detect outliers in each feature ============
    for key in ppg_feat.keys():
        if ppg_feat[key]["y"] is None:
            print("huh? there's no y value?")
        ppg_feat[key]["is_outlier"] = detect_outliers(
            ppg_feat[key]["y"],
            min_deviation_tolerance[key],
            outlier_tolerance=outlier_tolerance,
        )

    # combine outliers across features for "grouped analysis"
    all_outliers = []
    outlier_group_padding = 62 # somewhat arbitrary
    min_gap_len = (fs / (hr / 60)) * 2
    for feat in ppg_feat.values():
        if np.any(feat["is_outlier"]):
            all_outliers.extend(list(feat["x"][feat["is_outlier"]]))
    all_outliers = np.unique(np.sort(all_outliers))

    non_isolated_outliers = sqi.remove_isolated_outliers(all_outliers, min_gap_len)
    artif_groups, artifact_durations = sqi.get_artifact_groups(
        artifact_locations=non_isolated_outliers,
        threshold_distance=min_gap_len,
        signal_len=len(ppg),
        group_padding=outlier_group_padding,
    )
    outliers_everywhere = is_outliers_everywhere(
        ppg_len=len(ppg), 
        outlier_idxs=all_outliers,
        )

    # ppg itself is a feature. This is excluded from outlier analysis.
    ppg_index = np.arange(0, len(ppg))
    ppg_feat["ppg"]["x"] = ppg_index
    ppg_feat["ppg"]["y"] = ppg
    ppg_feat["ppg"]["is_outlier"] = [False] * len(ppg_index)

    # ================== interpolate & resample ==================
    # Turn PPG features into RIV waveforms by resampling & filtering
    # We resample RIVs to 10Hz (Pimental et al., 2017). Interpolation from sampling may
    # cause ripples, so we filter the resampled signals to remove large ripples
    f_low = rr_min / 60
    f_high = rr_max / 60
    filter_order = 5
    sos = signal.butter(
        filter_order, [f_low, f_high], "bandpass", fs=fs_riv, output="sos"
    )
    riv = defaultdict(dict)
    for key, feat in ppg_feat.items():
        feat_x = feat["x"].copy()
        feat_y = feat["y"].copy()

        # If current frame has artifacts (outliers) in the center of frame, then it is
        # likely contaminated with low frequency noise.
        artifact_in_center = False

        if (
            (len(non_isolated_outliers) > 1) & 
            (remove_outliers is not None) & 
            (~outliers_everywhere)
            ):
            # if sum(feat["is_outlier"]) > 1:
            if remove_outliers == "segment-wise":
                # # remove signal segments based on outliers of specific RIV signals
                # non_isolated_outliers = sqi.remove_isolated_outliers(
                #     feat_x[feat["is_outlier"]], min_gap_len
                # )
                # artif_groups, _ = sqi.get_artifact_groups(
                #     artifact_locations=non_isolated_outliers, threshold_distance=min_gap_len
                # )
                feat_x, feat_y, artifact_in_center = cut_riv_outliers(
                    feat_x,
                    feat_y,
                    signal_len=len(ppg),
                    min_keep_pct=min_keep_pct,
                    outlier_groups=artif_groups,
                    min_group_duration=min_gap_len + outlier_group_padding * 2,
                )
            elif remove_outliers == "point-wise":
                feat_x = feat_x[~feat["is_outlier"]]
                feat_y = feat_y[~feat["is_outlier"]]
            elif remove_outliers == "experiment":
                feat_x_group_cut, _, artifact_in_center = cut_riv_outliers(
                    feat_x,
                    feat_y,
                    signal_len=len(ppg),
                    min_keep_pct=min_keep_pct,
                    outlier_groups=artif_groups,
                    min_group_duration=min_gap_len + outlier_group_padding * 2,
                )
                feat_x_point_cut = feat_x[~feat["is_outlier"]]

                retained_x = list(set(feat_x_group_cut) & set(feat_x_point_cut))
                retained_x_idx = [True if x in retained_x else False for x in feat_x]
                feat_x = feat_x[retained_x_idx]
                feat_y = feat_y[retained_x_idx]
            else:
                raise ValueError(
                    """
                    remove_outliers must be one of Literal['segment-wise', 'point-wise',
                    'experiment'] or None
                    """
                )

        if key == "ppg":
            # ppg doesn't need to be interpolated before resampled
            # we add data to the dictionary anyway to be consistent
            riv[key]["x_interp"] = ppg_index
            riv[key]["y_interp"] = ppg
            x_resampled, y_resampled = resample_riv(ppg_index, ppg, fs, fs_riv)
        else:
            # stop point for debugging
            # if len(feat_y) < 10:
            #     print('huh? no feat_y?')
            x_interp, y_interp = util.interpolate(x=feat_x, y=feat_y)
            y_interp = util.standardize(y_interp)
            riv[key]["x_interp"] = x_interp / fs
            riv[key]["y_interp"] = y_interp
            x_resampled, y_resampled = resample_riv(x_interp, y_interp, fs, fs_riv)

        riv[key]["x_resampled"] = x_resampled
        riv[key]["y_resampled"] = y_resampled

        # Using the reflection trick can help reduce edge effects
        filtered_y = util.filter_signal_with_reflection(sos, y_resampled)
        riv[key]["x_filtered"] = x_resampled
        riv[key]["y_filtered"] = util.standardize(filtered_y)

        riv[key]["x"] = riv[key]["x_filtered"]
        riv[key]["y"] = riv[key]["y_filtered"]

        # quality indicators, could act as features for ML during RR candidate selection
        # differencial coefficient of variation (DCV) is used by 
        # Baker, Xiang, and Atkinson, 2021. PLOS One. DOI: 10.1371/journal.pone.0249843
        riv[key]["dcv"] = 1 - np.std(ppg_feat[key]["y"])/np.mean(ppg_feat[key]["y"])
        riv[key]["kurtosis"] = kurtosis(riv[key]["y_filtered"])
        riv[key]["skew"] = skew(riv[key]["y_filtered"])
        riv[key]["artifact in frame center"] = artifact_in_center

    if show:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.5, 0.5],  # relative heights of each row
        )

        # ppg-related elements
        # fmt: off
        fig.add_scatter(
            x=ppg_index/fs, y=ppg, name="raw", mode="lines", row=1, col=1,
            legendgroup="ppg", legendgrouptitle_text="PPG",
        )
        fig.add_scatter(
            x=ppg_feat["RIIV_upper"]["x"]/fs,
            y=ppg_feat["RIIV_upper"]["y"],
            name="peaks", mode="markers", row=1, col=1,
            legendgroup="ppg",
        )
        fig.add_scatter(
            x=ppg_feat["RIIV_lower"]["x"]/fs,
            y=ppg_feat["RIIV_lower"]["y"],
            name="troughs", mode="markers", row=1, col=1,
            legendgroup="ppg",
        )
        fig.add_scatter(
            x=ppg_feat["RIFV_mas"]["x"]/fs,
            y=ppg[ppg_feat["RIFV_mas"]["x"]],
            name="max slope", mode="markers", row=1, col=1,
            legendgroup="ppg",
        )
        if 'notch area ratio' in ppg_feat.keys():
            fig.add_scatter(
                x=ppg_feat['notch area ratio']['x']/fs,
                y=ppg[ppg_feat['notch area ratio']['x']],
                name="d-notch", mode="markers", row=1, col=1,
                legendgroup="ppg",
            )
        fig.add_scatter(
            x=riv["ppg"]["x_filtered"],
            y=riv["ppg"]["y_filtered"],
            mode="lines",
            line={
                "color": "#0064d3",
                "dash": "solid",
            },
            row=2,
            col=1,
            legendgroup="ppg",
            legendgrouptitle_text="ppg",
            name="filtered ppg",
        )
        # fmt: on

        # RIV-related elements - outliers
        for idx, (key, feat) in enumerate(ppg_feat.items()):
            outlier_x = feat["x"][feat["is_outlier"]] / fs
            fig.add_scatter(
                x=outlier_x,
                y=np.ones(len(outlier_x)) * np.max(ppg),
                mode="markers",
                name="outlier",
                marker={"color": mpl.colors.rgb2hex(mpl.colormaps["tab10"](idx))},
                row=1,
                col=1,
                showlegend=False,
                legendgroup=key,
                legendgrouptitle_text=key,
            )
        steps = ["interp", "resampled", "filtered"]
        labels = ["1-interpolated", "2-resampled", "3-filtered"]
        line_styles = ["dash", "dot", "solid"]
        # display RIV waveforms except ppg
        for idx, key in enumerate(list(riv.keys())[:-1]):
            for step, label, style in zip(steps, labels, line_styles):
                fig.add_scatter(
                    x=riv[key][f"x_{step}"],
                    y=riv[key][f"y_{step}"],
                    mode="lines",
                    line={
                        "color": mpl.colors.rgb2hex(mpl.colormaps["tab10"](idx)),
                        "dash": style,
                    },
                    row=2,
                    col=1,
                    legendgroup=key,
                    legendgrouptitle_text=key,
                    name=label,
                )

        fig.update_traces(showlegend=True)
        fig.update_layout(height=800)
        fig.show()

        # display some stats
        keys = riv.keys()
        skews = [riv[key]["skew"] for key in keys]
        kurt = [riv[key]["kurtosis"] for key in keys]
        debug_df = pd.DataFrame(
            {
                "feature": keys,
                "skew": skews,
                "kurtosis": kurt,
            }
        )
        display(debug_df)

    return riv, ppg_feat
