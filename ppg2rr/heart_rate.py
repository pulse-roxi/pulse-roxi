"""Detect individual heartbeat-related pulses from PPG.

Heart rate is the true physiological rate of heart beats per time;
pulse rate is the estimation of heart rate measured at another body location
based on changes in blood pressure.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import (
    butter,
    sosfiltfilt,  # for envelope calculation
    find_peaks,
    welch,
)

from ppg2rr import util

# TODO: memoize function to improve performance
def butter_bandpass(data, lowcut, highcut, fs, order=5):
    """Zero phase bandpass filter."""
    sos = butter(order, [lowcut, highcut], btype="bandpass", analog=False, output='sos', fs=fs)
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data


def estimate_instantaneous_heart_rate_with_resample(
    ppg: np.array,
    fs: float,
    smooth: bool = False,
    median_filter_dur: float = 25,
    hr_min=35,
    hr_max=240,
):
    """Estimates the instantaneous pulse rate for a given ppg signal.

    This function is typically only used if estimate_avg_heart_rate() fails to find a 
    heart rate. In this case, the heart rate PSD peak is weak, implying the signal is
    corrupted with high frequency noise. So we assume each peak with 
    prominance > 0.5 to correspond to a heart beat.
    
    Don't rely on this function to identify each PPG pulse. Instead, use 
    riv_est.find_extrema_HX2017() to find the max and min corresponding to each
    ppg pulse based on the estimated heart rate.
    
    General procedure:
    1. Find peaks
    2. Make an initial estimation of HR
    3. interpolate & resample
    4. median filter with a 15-sample long kernel

    Default values assumes heart rate ranges between 0.583 - 4hz (35-240 beats
    per minute).
    

    Performance note:
        with smooth = False, this function is on average 2x faster than
        estimate_avg_heart_rate. But the two functions will return slightly different
        heart rate estimates, depending on the variability of the instantaneous heart
        rate.

    Args:
        ppg: PPG signal from which a heart rate is estimated.
        fs: Sampling freqeuncy of the PPG signal.
        smooth: If True, smooths the instantaneous heart rate with a median filter.
        median_filter_dur: Samples are resampled to 1hz before smoothing with median
            filter. Defaults to 25 seconds.
        hr_min: Minimum heart rate (beats per minute) of the sample. Used in bandpass
            filter. Default to 35, for extreme atheletes.
        hr_max: Maximum heart rate (beats per minute) of the sample. Used in bandpass
            Filter. Default to 240, for newborns.

    Return:
        Array of heart rate values
        Array of times [s] of those heart rate values
    """
    fmin = hr_min / 60  # hertz
    fmax = hr_max / 60  # hertz

    # ================== pre-process ==================
    ppg_filtered = butter_bandpass(ppg, lowcut=fmin, highcut=fmax, fs=fs, order=5)
    ppg_standard = util.standardize(ppg_filtered)

    # find the beginning and end of pulses
    peak_idx, _ = find_peaks(-ppg_standard, prominence=0.5)
    instantaneous_hr = fs * 60 / np.diff(peak_idx)

    if len(peak_idx) == 0:
        util.newprint("estimate_instantaneous_heart_rate_with_resample(): No peaks found.", on_a_new_line=True)

    # Interpolate with "next" per Berger 1986 so that this stepwise signal can be averaged accurately
    x_berger, y_berger = util.interpolate(
        peak_idx[1:] / fs,
        instantaneous_hr,
        kind="next",
        step=0.01
    )
    # fig = px.line(x=x_berger, y=y_berger)
    # fig.show()

    if smooth:
        # ===== use a median filter to smooth out the instantaneous heart rate =====
        # NOTE: Patient monitor PR calculation lags the instantaneous PR by 2 seconds
        # Therefore the calculated PR requires t_offset to match the patient monitor
        fs_resampled = 1            # TODO: This is surely too slow, but I don't think we're using this anyway
        _, y_interp = util.interpolate(peak_idx[1:], y=instantaneous_hr)
        y_resampled = util.resample_signal(y_interp, fs, fs_resampled)

        # median filter; round duration to nearest odd number
        med_filt_len = round(median_filter_dur * fs_resampled)
        if med_filt_len % 2 == 0:
            med_filt_len += 1
        smoothed_hr = util.medfilt_with_reflections(
            y_resampled, kernel_size=med_filt_len
        )
        return smoothed_hr, peak_idx / fs
    else:
        return y_berger, x_berger


def estimate_avg_heart_rate(
    ppg: np.ndarray,
    sampling_rate: float,
    min_heart_rate_hz: float = 1.2,
    show: bool = False,
    save_Hz_fig_as: str = "",
) -> float:
    """Estimate the average heart rate via Welch's power spectral estimate.

    This is a very simple implementation of a heart rate estimator. It first detects spectral peaks
    in the heart rate region (between min_heart_rate and 4 hz), then compares those peaks against
    all other PSD peaks below 40 hz (to stay away from the shoulders of 50 Hz noise). If the peak in
    the heart rate region is the greatest peak, as defined by the `peak_prominance_ratio` variable,
    the signal is assumed to be "clean" with the strongest AC component corresponding to the heart
    rate. The exception is that this function also allows the presence of the 1st harmonic peak with
    a peak value greater than the peak_prominance_ratio. 
    
    Heart rates can be identified in noisier signals by specifying prominance values 
    during find_peaks (e.g., tests/resources/ppg_waveforms/ppg_kapiolanip1-8-3.csv), 
    this is purposely left out for now.
    
    Three possible conditions are marked in the function:
    1. No prominant peaks are present within the heart rate region 
        -> np.nan is returned
    2. One prominant peak or prominant peak + 1st harmonic are found 
        -> assume prominant peak is pulse rate to be pulse rate
    3. More than one large peak found in the heart rate and breathing rate region, 
        indicating noisy signal with low signal quality -> np.nan is returned

    The frequency resolution of welche's method is sampling_rate/N, where N is the
    number of samples in the ppg signal. e.g., if given 10 seconds of ppg,
    then the frequency resolution is sampling_frequency/10*sampling_frequency = 0.01.
    We set nfft = 2048 to get the frequency resolution to be 0.0004hz

    The default nperseg (segment length) in the Welch's PSD spectral estimation is
    256 samples, which is too short to accurately measure heart rate frequency on the
    data tested. Setting this to 1024 with nfft = 8192 on a signal sampled at 250 hz
    seems to be the sweet spot between accuracy and resolution.

    Args:
        ppg (np.ndarray): The raw PPG signal.
        sampling_rate (float): The sampling rate of the ppg signal [hz].
        min_heart_rate_hz (float): The minimum pulse rate.
            The default is set to 1.2 hz (72 bpm) for infants, but may be as low as
            0.5 hz (30 bpm) for adults.
        show: If True, displays the result of Welch's PSD estimate.
        save_Hz_fig_as: If not blank, saves the Hz PSD plot using this filename including path.

    Returns:
        float: heart rate in beats per minute, or np.nan is none found.
    """
    max_heart_rate_hz = 4
    max_noise_freq = 40
    peak_prominance_ratio = 1.5 # the most prominant HR peak is this much larger than 
        # other peaks
    
    
    ppg_standardized = util.standardize(ppg)
    nperseg = np.min([1024, len(ppg)])
    f, Pxx = welch(
        ppg_standardized,
        fs=sampling_rate,
        nperseg=nperseg,
        nfft=8192,
    )
    peak_idxs, _ = find_peaks(Pxx, height=0.2)
    peak_freqs = f[peak_idxs]
    peak_vals = Pxx[peak_idxs]
    peak_idx_in_hr_region = (
        (peak_freqs >= min_heart_rate_hz) & (peak_freqs <= max_heart_rate_hz)
        )
    
    peak_idx_others_below_max_noise_f = []
    # No peak found in hr region
    if peak_idx_in_hr_region.sum() == 0:
        condition = 1
        hr_estimate = np.nan
        
    else:
        # check that largest peak in the hr regions is x*ratio greater than all other 
        # peaks. If true, we have confidence that this is the most likely heart rate.
        # otherwise, we have very low confidence that the actual heart rate is found.
        largest_peak_idx_in_hr_region = peak_idxs[peak_idx_in_hr_region][
        np.argmax(peak_vals[peak_idx_in_hr_region])
        ]
        # boolean indexing
        peak_idx_others_below_max_noise_f = (
            (peak_freqs < max_noise_freq) 
            # include this condition if we want to estimate HR regardless of signal quality
            # exclude it if we want np.nan if signal quality is low
            # & (peak_freqs > min_heart_rate_hz) 
            & (peak_idxs != largest_peak_idx_in_hr_region)
        )
        one_prominent_peak = np.all(
            np.max(peak_vals[peak_idx_in_hr_region])
            > peak_prominance_ratio * peak_vals[peak_idx_others_below_max_noise_f]
        )
        
        # there could be a second strong peak that is harmonic of the first.
        acceptable_second_peak = False
        if ~one_prominent_peak:
            # find the second largest peak, and assess its ratio to the largest peak
            peak_freq_ratios = (
                np.max(peak_freqs[peak_idx_others_below_max_noise_f])
                /f[largest_peak_idx_in_hr_region]
                )
            acceptable_second_peak = 1.9 < peak_freq_ratios < 2.1
        
        if one_prominent_peak or acceptable_second_peak:
            condition = 2
            hr_estimate = f[largest_peak_idx_in_hr_region] * 60
        else:
            condition = 3
            hr_estimate = np.nan

    if show or save_Hz_fig_as:
        fig_width = 900
        fig_height = 350
        
        # Per minute
        rate_per_minute = f * 60
        fig = px.line(x=rate_per_minute, y=Pxx)
        fig.add_vline(x=min_heart_rate_hz*60, line_dash="dash")
        fig.add_vline(x=max_heart_rate_hz*60, line_dash="dash")
        fig.add_vline(x=max_noise_freq*60, line_dash="dash")
        fig.update_layout(xaxis_title="frequency (cycles per minute)", yaxis_title="Pxx", width=fig_width, height=fig_height)
        if show:
            fig.show()

            if condition == 1:
                print("No peaks found in heart rate region")
            if condition == 3:
                print("No prominant peaks found")
            print(
                "peaks in hr region: ",
                peak_freqs[peak_idx_in_hr_region] * 60,
                peak_vals[peak_idx_in_hr_region],
            )
            print("other peaks below 50 Hz: ", peak_freqs[peak_idx_others_below_max_noise_f] * 60)
            print("x ratio peak values below 50 Hz: ", peak_prominance_ratio * peak_vals[peak_idx_others_below_max_noise_f])
            print(f"estimated hr: {hr_estimate} beats per min")

    return hr_estimate


def compare_heart_rate_estimations(
    ppg: np.array,
    fs: float,
    median_filter_dur: float = 10,
    reference_pr: np.array = None,
):
    """Compares the result of two heart rate estimation approaches.

    Compares the functions
        1. estimate_instantaneous_heart_rate_with_resample()
        2. estimate_avg_heart_rate()

    Performance note:
        estimate_instantaneous_heart_rate_with_resample(smooth=False) -> 1.78 ms
        estimate_instantaneous_heart_rate_with_resample(smooth=True) -> 5.77 ms
        estimate_avg_heart_rate() -> 3.93 ms

    Args:
        ppg: Signal to estimate HR from.
        fs: Sampling frequency of the ppg signal.
        median_filter_dur: Defaults to 10 seconds. This is converted to samples
            using the given fs.
        reference_pr: Reference values to compare against

    Return:
        Array noting start of each heart pulse in the given PPG.
    """
    (
        hr_instantaneous_with_smooth,
        peak_idx,
    ) = estimate_instantaneous_heart_rate_with_resample(
        ppg, fs, smooth=True, median_filter_dur=median_filter_dur
    )
    (
        hr_instantaneous_no_smooth,
        peak_idx,
    ) = estimate_instantaneous_heart_rate_with_resample(
        ppg, fs, smooth=False, median_filter_dur=median_filter_dur
    )
    hr_welch_psd = estimate_avg_heart_rate(ppg, sampling_rate=fs, show=False)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.5],  # relative heights of each row
    )
    ppg_standard = util.standardize(ppg)
    ppg_t = np.arange(0, len(ppg_standard) / fs, 1 / fs)
    fig.add_trace(
        go.Scatter(x=ppg_t, y=ppg_standard, mode="lines", name="Filtered PPG"),
        row=1,
        col=1,
    )

    fig.add_scatter(
        x=peak_idx / fs,
        y=ppg_standard[peak_idx],
        mode="markers",
        name="detected troughs",
        row=1,
        col=1,
    )

    # plot heart rates
    fig.add_scatter(
        x=peak_idx[1:] / fs,
        y=hr_instantaneous_no_smooth,
        mode="lines",
        name="instantaneous HR",
        visible="legendonly",
        row=2,
        col=1,
    )
    t_resampled = np.arange(len(ppg) / fs)
    fig.add_scatter(
        x=t_resampled,
        y=hr_instantaneous_with_smooth,
        mode="lines",
        name="instantaneous HR with smoothing",
        row=2,
        col=1,
    )
    fig.add_scatter(
        x=[0, len(ppg) / fs],
        y=[hr_welch_psd, hr_welch_psd],
        name="PSD estimated HR",
        mode="lines",
        row=2,
        col=1,
    )
    if reference_pr is not None:
        fig.add_scatter(
            y=reference_pr[2:],
            mode="lines",
            name="reference HR",
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="PPG", row=1, col=1)
    fig.update_yaxes(title_text="HR Estimates", row=2, col=1)
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=700,
        width=800,
    )
    fig.show()
