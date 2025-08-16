"""Signal quality related calculations."""

from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import numpy.typing as npt
from IPython.display import display
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from scipy.stats import iqr, kurtosis, skew
from scipy.signal import welch
import plotly.express as px
import plotly.graph_objects as go

from ppg2rr import util


def group_outlier_analysis(
    ppg_feat: dict,
    ppg_len: int,
    min_gap_len: float,
):
    """Outlier analysis as as a whole - grouping all RIV outliers together.

    Calculate percent of the frame containing outliers, and whether outliers are
    grouped in the center of the frame.

    If a section of PPG has many outliers, then we mark that section as having poor
    quality. On the other hand, single isolated occurrences of outliers are considered
    false positives. Here we consider occurances larger than one heart beat to be
    "isolated".

    Args:
        ppg_feat: output from extract_rivs()
        ppg_len: length of ppg window
        min_gap_len: mininum duration between outlier groups.
    """
    # initialize return values
    data = {}
    data["pct_frame_with_outliers"] = 0
    data["artifact in center"] = False

    # combine outliers across ppg features
    outliers = []
    for feat in ppg_feat.values():
        if np.any(feat["is_outlier"]):
            outliers.extend(list(feat["x"][feat["is_outlier"]]))
    outliers = np.unique(np.sort(outliers))

    # Do nothing if there's only a single occurance of outlier or no outliers
    if len(outliers) < 2:
        return data

    # ===================== remove false positives ======================
    # False positives are isolated instances of outlier occurance
    # if all we have are individual instances of spikes, do nothing

    # two heartbeats
    true_pos_outliers = remove_isolated_outliers(outliers, min_gap_len)

    if len(true_pos_outliers) < 2:
        return data

    # =========== Calculate proporation of frame with outliers ============
    # Outliers may be grouped into clusters. We assume clusters are separated by
    # some minimum gap length.
    gaps = np.diff(true_pos_outliers)
    total_artif_duration = gaps[gaps < min_gap_len].sum()
    data["pct_frame_with_outliers"] = total_artif_duration / ppg_len

    # =========== Are there artifacts in the middle of the frame? =======
    # Artifacts in the beginning or end of the signal is considered
    # acceptable and are trimmed off. If artifacts are in the middle, then the
    # signal tends to be corrupted by low frequency information. In this case
    # we'll need to bias our RR search toward the high RRs.
    # We'll define "center of the frame" to be between 25%~75% of the frame and
    # artifacts "occupy" the center of the frame if 50% of the artifacts fall in that
    # center range.

    rel_artif_positions = true_pos_outliers / ppg_len
    artif_in_center_maybe = (rel_artif_positions > 0.25) & (rel_artif_positions < 0.75)
    num_artif_in_center = np.count_nonzero(artif_in_center_maybe) / len(
        true_pos_outliers
    )
    if (total_artif_duration > 0.15) & (num_artif_in_center >= 0.5):
        data["artifact in center"] = True

    return data


def remove_isolated_outliers(outliers: Iterable, min_gap_len: float) -> np.ndarray[int]:
    """Remove isolated outliers.

    Remove isolated instances of outlier occurance, characterized by
    single outliers further than min_gap_len apart from other outliers.

    Args:
        outliers: numpy array wtih indices of outliers
        min_gap_len(float): If gaps (diff) between outliers < min_gap_len,
            then the two outliers are considered in the same group. Isolated
            outliers are considered to be false positives.

    """
    if len(outliers) < 2:
        return np.array([])

    outliers = np.asarray(outliers)
    gaps = np.diff(outliers)

    # true positives
    true_pos_gaps = gaps < min_gap_len
    left_post = set(outliers[1:][true_pos_gaps])
    right_post = set(outliers[:-1][true_pos_gaps])
    true_pos_outliers = np.array(sorted(left_post.union(right_post)))

    return true_pos_outliers


def get_artifact_groups(
    artifact_locations: Union[np.ndarray, list],
    threshold_distance: int,
    signal_len: int = 10e6,
    group_padding: float = 62,
):
    """Identify groups of artifacts based on the threshold distance.

    Args:
        artifact_locations: Indices of detected artifacts
        threshold_distance: Minimum distance between artifact groups
        signal_len: Length of the signal, so we don't add padding outside of it.
            Default 10e6.
        group_padding: Padding [samples] before and after an artifact group.
            Default is 62 = floor(0.5 seconds @ 125 hz)

    Returns:
        artifact_groups: [(artifact_start_index, artifact_end_index), ...]
        artifact_durations: [artifact_duration_1, ...]
    """
    artifact_groups = []
    group_start = None
    for i, loc in enumerate(artifact_locations):
        if group_start is None:
            group_start = loc - group_padding
        elif loc - artifact_locations[i - 1] > threshold_distance:
            group_end = min(artifact_locations[i - 1] + group_padding, signal_len)
            artifact_groups.append((group_start, group_end))
            group_start = loc - group_padding
    if group_start is not None:
        group_end = min(artifact_locations[-1] + group_padding, signal_len)
        artifact_groups.append((group_start, group_end))

    # Compute the duration of each artifact group
    artifact_durations = [group[1] - group[0] for group in artifact_groups]

    return artifact_groups, artifact_durations

def template_match_similarity(
    y: np.ndarray[float],
    peak_locs: np.ndarray[float],
    correlation_threshold: float = 0.99,
    show: bool = False,
):
    """Calculates similarity between PPG pulses and the average PPG pulse shape.

    Inspired by [1], which calculates the similarity between PPG pulses and the
    average PPG pulse shape using correlation coefficient. Note that correlation
    coefficient is equivalent to the zero-normalized cross correlation at time 0.
    We also use deviation from the average pulse shape standard deviation as another
    distance metric to identify "non-conforming" PPG pulses. Here, we report
    1 - (percentage of nonconforming pulses) and calls it the "percentage of
    diagnostic quality pulses" to be consistent with literature [2].

    Cross correlation threshold for non-conforance is set to 0.99,
    and deviation from the average pulse shape standard deviation is set to 0.3.

    Args:
        y: A waveform to calculate similarity on
        peak_locs: maxima indices of each pulse in the waveform
        correlation_threshold: minimum correlation between each individual pulse with
            the average pulse to be considered diagnostic quality.
        show: Default False
    Returns:
        avg_dist (dict): average distances using different measures
        distance_matrix (dict): distance matrix from which avg distances are calculated
            It has the following keys:
            - "pct diagnostic quality pulses" considers both pulse shapes and heights
            - "pct good pulse shapes" only considers pulse shape via cross correlation

    Example useage: template_match_similarity(ppg_window,peak_indices_samples,show=True)

    [1] Orphanidou et al., 2015, "Signal-quality indices for the electrocardiogram
    and photoplethysmogram: derivation and applications to wireless monitoring".
    [2] Huthart S, Elgendi M, Zheng D, Stansby G, Allen J. Advancing PPG Signal 
    Quality and Know-How Through Knowledge Translation-From Experts to Student 
    and Researcher. Front Digit Health. 2020 Dec 21; doi: 10.3389/fdgth.2020.619692. 

    NOTE: The MATLAB version of this function also returns the quality metric
    for a "lookback" period, where the quality of the most recent 5 or 10 seconds of 
    the signal is assessed. This is used to display "instantaneous" quality. This
    is not implemented in the python code since we can simply make two separate
    calls to this function, and are not constrained by computational resources.

    @ Kenny Chou <kchou@nhgh.org>
    New Horizons, GHLabs
    """
    win_len = round(np.median(np.diff(peak_locs)))  # median beat-to-beat interval
    half_win_len = np.floor(win_len / 2).astype(int)

    # remove peaks too close to the edge, which correspond to "incomplete" pulses
    edge_peaks_bool = (peak_locs <= half_win_len) | (peak_locs + half_win_len > len(y))
    center_peaks = peak_locs[~edge_peaks_bool]

    # extract individual PPG pulse_matrix, store in columns
    # avoid for-loop by using 2d array indices, where each row is a slice of the ppg
    # [:, np.newaxis] turns each element of the pulse_starts array into row arrays, and
    # + np.arange turns each of those row arrays into [pulse_start: pulse_end]
    pulse_starts = center_peaks - half_win_len
    indices = pulse_starts[:, np.newaxis] + np.arange(win_len) - 1
    pulse_matrix = y[indices.astype(int).T]

    # Concatenate average pulse shape to pulse matrix
    mean_pulse_shape = np.mean(pulse_matrix, axis=1)[:, np.newaxis]
    pulse_matrix_with_avg = np.concatenate((pulse_matrix, mean_pulse_shape), 1)

    distance_mtx = {}
    avg_dist = {}
    # Similarity metric - correlation coefficient, output between -1 and 1
    distance_mtx["corrcoef"] = np.corrcoef(pulse_matrix_with_avg, rowvar=False)
    bad_pulse_shapes = np.array(distance_mtx["corrcoef"][:-1, -1]) < correlation_threshold

    # outliers based on pulse amplitude value ranges
    # see https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm#MAD
    pulse_stds = np.std(pulse_matrix, axis=0)
    m = 3
    d = np.abs(pulse_stds - np.median(pulse_stds))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d)) #scaled distance to median
    std_outliers = s > m
    
    # inspect
    bad_apples = bad_pulse_shapes | std_outliers
    avg_dist["pct poor pulse shapes"] = bad_pulse_shapes.sum() / len(bad_pulse_shapes)
    avg_dist["pct amplitude outliers"] = std_outliers.sum() / len(std_outliers)
    avg_dist["pct good pulse shapes"] = 1 - (bad_pulse_shapes.sum() / len(bad_apples))
    avg_dist["pct diagnostic quality pulses"] = 1 - (bad_apples.sum() / len(bad_apples))
    if len(np.where(bad_apples)[0]) == 0:
        avg_dist["mean loc nonconforming pulses"] = np.array([])
    else:
        avg_dist["mean loc nonconforming pulses"] = np.mean(
            np.where(bad_apples)[0] / len(bad_apples)
        )
    if show:
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [8, 2]})
        ax1.set_title(f"Quality: {avg_dist['pct diagnostic quality pulses']:.2f}")
        ax2.set_xlabel("Samples")
        ax2.set_ylabel("Amplitude")
        ax1.plot(pulse_matrix[:, ~bad_apples], color="#aec5c9")
        ax1.plot(pulse_matrix[:, bad_pulse_shapes], color="#ce4c4abd")
        ax1.plot(pulse_matrix[:, std_outliers], linestyle="--", color="#cf9f37bd")
        ax1.plot(mean_pulse_shape, linewidth=3, color="#1f77b4")
        # turn off the top, bottom, left and right spines
        for spine in ax1.spines.values():
            spine.set_visible(False)
        # create a simplified legend (there are too many separate series for an auto legend)
        line_bad_pulse_shapes = mlines.Line2D([], [], label="bad_pulse_shapes", color="#ce4c4abd")
        line_std_outliers     = mlines.Line2D([], [], label="std_outliers", linestyle="--", color="#cf9f37bd")
        fig.legend(handles=[line_bad_pulse_shapes, line_std_outliers])

        # unstacked fiugre
        ax2.plot(y)
        for idx in range(pulse_matrix.shape[1]):
            if bad_pulse_shapes[idx]:
                ax2.plot(indices[idx,:],pulse_matrix[:,idx], color="#ce4c4abd")
            if std_outliers[idx]:
                ax2.plot(indices[idx,:],pulse_matrix[:,idx], color="#cf9f37bd")
                
        plt.show()

    return avg_dist, distance_mtx

def moving_window_stats(
    signal,
    window_length: int,
    n_overlap: int,
    fs: Optional[float] = None,
    difference: bool = False,
    standardize: bool = True,
    threshold_level: Union[float, npt.NDArray] = 0.3,
    show: bool = False,
) -> Tuple[npt.NDArray, dict]:
    """Calculate mean, variance, skew, kurtosis, and IQR using a moving window.

    Args:
        signal: signal array to be analyzed.
        window_length: samples
        n_overlap: samples
        fs: For converting samples to time during visualization.
        difference: If true, take the difference in stats between windows. Default
            is False.
        threshold_level: Default is 0.4. If an array is provided, then it is assumed
            that the array contains specific threshold levels for each statistic.
        standardize: If True, standardize the signal first.
        show: If True, display signal and first order stat differences (derivatives).

    Returns:
        A dictionary with keys being each statistic in the window, and values being
        the corresponding statistic array.

    Example usage:
    ```
        window_len = 3*fs # 3 second window
        n_overlap = 2*fs # 2 second overlap. Window increment is 3-2=1 second
        show = True
        moving_window_stats(ppg, window_len, n_overlap, fs, show)
    ```
    """
    if standardize:
        signal = util.standardize(signal)

    num_windows = int((len(signal) - window_length) // (window_length - n_overlap)) + 1
    statistics = np.zeros((num_windows, 5))  # Array to store the statistics
    start_idx = []

    for i in range(num_windows):
        start = i * (window_length - n_overlap)
        end = start + window_length
        window = signal[start:end]  # Extract the current window
        start_idx.append(start)

        # Calculate the statistics using the window
        if np.sum(np.abs(np.diff(window))) < 0.1:
            statistics[i, 0] = np.mean(window)
            statistics[i, 1] = np.var(window, ddof=1)
            statistics[i, 2] = np.nan
            statistics[i, 3] = np.nan
            statistics[i, 4] = iqr(window)
        else:
            statistics[i, 0] = np.mean(window)
            statistics[i, 1] = np.var(window, ddof=1)
            statistics[i, 2] = skew(window)
            statistics[i, 3] = kurtosis(window)
            statistics[i, 4] = iqr(window)

    # make start index at the end of the window
    start_idx_with_offset = np.array(start_idx) + window_length
    if not difference:
        out_dict = {
            "mean": statistics[:, 0],
            "std": statistics[:, 1],
            "skew": statistics[:, 2],
            "kurtosis": statistics[:, 3],
            "iqr": statistics[:, 4],
        }
    else:
        statistics_diff = np.diff(statistics, axis=0)
        out_dict = {
            "delta mean": statistics_diff[:, 0],
            "delta std": statistics_diff[:, 1],
            "delta skew": statistics_diff[:, 2],
            "delta kurtosis": statistics_diff[:, 3],
            "delta iqr": statistics_diff[:, 4],
        }

    # stat_difference = np.diff(statistics, axis=0)

    # pct_windows_over_thresh = (
    #     np.sum(stat_difference > threshold_level, axis=0) / num_windows
    # )
    # pct_windows_over_thresh = np.mean(stat_difference, axis=0)
    # pct_windows_over_thresh_dict = {
    #     "mean": pct_windows_over_thresh[0],
    #     "std": pct_windows_over_thresh[1],
    #     "skew": pct_windows_over_thresh[2],
    #     "kurtosis": pct_windows_over_thresh[3],
    # }

    if show:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.5],  # relative heights of each row
        )

        ppg_t = np.arange(0, len(signal) / fs, 1 / fs)
        fig.add_scatter(
            x=ppg_t,
            y=signal,
            mode="lines",
            name="standardized signal",
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=np.array(start_idx_with_offset) / fs,
            y=statistics[:, 0],
            mode="lines+markers",
            name="mean",
            row=1,
            col=1,
        )

        # for idx, label in zip([0, 1, 2, 3], ["mean", "variance", "skew", "kurotsis"]):
        for key, value in out_dict.items():
            fig.add_scatter(
                x=start_idx_with_offset / fs,
                y=value,
                mode="lines+markers",
                name=f"{key}",
                row=2,
                col=1,
            )

        fig.add_hline(
            y=threshold_level,
            line_dash="dot",
            annotation_text="quality threshold",
            annotation_position="top right",
            row=2,
            col=1,
        )
        fig.add_hline(y=-threshold_level, line_dash="dot", row=2, col=1)
        fig.update_layout(
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        fig.show()

    out_dict["time index"] = start_idx_with_offset[: num_windows - int(difference)] / fs

    return out_dict


def ppg_mains_noise(
    ppg: np.ndarray,
    sampling_rate: float,
    min_freq: float = 30,
    show: bool = False,
    save_fig_as: str = False,
) -> Tuple[float, float, float]:
    """Measure the mains noise in the PPG signal.

    Args:
        ppg (np.ndarray): The raw PPG signal.
        sampling_rate (float): The sampling rate of the ppg signal [Hz].
        min_freq (float): The minimum frequency to consider for noise [Hz]. Default 30, for 50 Hz
            mains and side bands.
        show (bool): If True, displays the result of Welch's PSD estimate.
        save_fig_as (str): If non-blank, the figure is saved to this path.

    Returns:
        max_noise_freq (float): The frequency above min_freq with max Pxx
        Pxx_max_over_median (float): The ratio of Pxx max divided by Pxx median
        Pxx_mean (float): The mean Pxx above min_freq.
    """

    nperseg = min([1024*4, len(ppg)])
    nfft = max(nperseg, 8192*4)
    f, Pxx = welch(
        ppg,
        fs=sampling_rate,
        nperseg=nperseg,
        nfft=nfft,
    )

    Pxx_above_noise_freq = Pxx[f > min_freq]
    f_above_noise_freq = f[f > min_freq]
    sum_mains_noise_Pxx  = np.sum(Pxx_above_noise_freq)
    median_noise_Pxx = np.median(Pxx_above_noise_freq)
    Pxx_mean = np.mean(Pxx_above_noise_freq)
    max_noise_idx = np.argmax(Pxx_above_noise_freq)
    max_noise_freq = f_above_noise_freq[max_noise_idx]
    max_noise_Pxx = Pxx_above_noise_freq[max_noise_idx]
    Pxx_max_over_median = max_noise_Pxx / median_noise_Pxx

    # Compare to the filtered signal that we use elsewhere
    # ppg_filt_old = util.lowpass_butter(ppg, signal_fs=sampling_rate, order=4, f_max=20, show=False)
    # ppg_filt_new = util.lowpass_butter(ppg, signal_fs=sampling_rate, order=10, f_max=8, show=False)
    # f_filt_old, Pxx_filt_old = welch(
    #     ppg_filt_old,
    #     fs=sampling_rate,
    #     nperseg=nperseg,
    #     nfft=nfft,
    # )
    # f_filt_new, Pxx_filt_new = welch(
    #     ppg_filt_new,
    #     fs=sampling_rate,
    #     nperseg=nperseg,
    #     nfft=nfft,
    # )
    # # High-pass filtering to show just the noise.
    # # By shortcut, which should be valid given the phase-correct filtfilt filtering.
    # ppg_minus_filt_old = ppg - ppg_filt_old
    # ppg_minus_filt_new = ppg - ppg_filt_new

    if show or save_fig_as:

        # PPG
        # fig_width  = 2800
        # fig_height = 1000
        # t_axis = np.arange(0, len(ppg)/sampling_rate, 1/sampling_rate)
        # fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

        # fig.add_trace(go.Scatter(x=t_axis, y=ppg, mode="lines+markers", line={"width":0.8}, name="unfiltered"), 
        #               row=1, col=1)
        # fig.add_trace(go.Scatter(x=t_axis, y=ppg_filt_old, line={"width":1.2}, name="ppg_filt_old"),
        #               row=1, col=1)
        # fig.add_trace(go.Scatter(x=t_axis, y=ppg_filt_new, line={"width":1.2}, name="ppg_filt_new"),
        #               row=1, col=1)
        # fig.add_trace(go.Scatter(x=t_axis, y=ppg_minus_filt_old, line={"width":1}, name="ppg_minus_filt_old"),
        #               row=2, col=1)
        # fig.add_trace(go.Scatter(x=t_axis, y=ppg_minus_filt_new, line={"width":1}, name="ppg_minus_filt_new"),
        #               row=2, col=1)
        # fig.update_layout(
        #     xaxis2_title="time within frame (s)", 
        #     yaxis1_title="PPG", 
        #     yaxis2_title="High-pass",
        #     width=fig_width, 
        #     height=fig_height
        # )
        # if show:
        #     fig.show()

        # PSD
        fig_width = 800
        fig_height = 450
        fig = go.Figure()
        # fig.add_trace(go.Scatter(x=f_filt_new, y=Pxx_filt_new, line={"width":1}, name="ppg_filt_new"))
        # fig.add_trace(go.Scatter(x=f_filt_old, y=Pxx_filt_old, line={"width":1}, name="ppg_filt_old"))
        fig.add_trace(go.Scatter(x=f, y=Pxx, line={"width":1}, name="LEDx_DELTA"))

        fig.update_layout(
            title=f"For f > {min_freq} Hz, peak @ {max_noise_freq:5.1f} Hz. Power max:median = {Pxx_max_over_median:8,.0f}, mean = {Pxx_mean:.1e}.",
            xaxis_title="frequency (Hz)", 
            yaxis_type="log", 
            yaxis_title="Power", 
            width=fig_width, 
            height=fig_height,
        )
        fig.update_yaxes(
            range=[0,10],      # in logs
            dtick=1,
        )
        if show:
            fig.show()
        if save_fig_as:
            fig.write_image(save_fig_as)
    
    return max_noise_freq, Pxx_max_over_median, Pxx_mean
        

