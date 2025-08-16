"""General functions for signal manupilation and other utilities."""
from collections import Counter
from collections import defaultdict
from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import interpolate as interp
from scipy import signal


def cpm2hz(cpm: float) -> float:
    """Cycles per minute to Hertz [cycles per second] conversion."""
    return cpm / 60


def hz2cpm(hz: float) -> float:
    """Hertz to cycles per minute conversion."""
    return hz * 60


def cpm2wavelength(cpm: float) -> float:
    """Cycles per minute to wavelength [seconds] conversion."""
    return 60 / cpm


def wavelength2cpm(wavelength: float, fs: float) -> float:
    """Wavelength [samples] to cycles per minute conversion."""
    # 60 seconds/minute * samples/second * 1/samples = 1 / minute
    return 60 * fs / wavelength


def interpolate(
    x: np.ndarray,
    y: np.ndarray,
    kind: str = "cubic",
    t_range: Optional[tuple[float, float]] = None,
    step: float = 1,
    fill_value: str = "extrapolate",
):
    """Interpolate the non-uniformly sampled signal, treating sampling rate = 1.

    If the arrays have only one element, no interpolation can be performed but the arrays are
    returned without error.

    Args:
        x: indices of non-uniformly sampled signal
        y: a non-uniformly sampled signal
        kind: One of 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
            'cubic' (default), 'previous', or 'next'. See [1] for more info.
        t_range: If given, the returned t-axis ranges from (t_start, t_end)
        step: Time increment for the return t-axis. Default: 1 s, but that is too slow for accurate
            interpolation of HR and even RR for children. 0.01 s is safe.
        fill_value: "None" or "extrapolate" (default). See [1] for more info.

    Returns:
        t_uniform: indices the interpolated signal
        y_uniform: interpolated signal

    [1] docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    """
    
    if t_range is None:
        t_uniform = np.arange(round(x[0]/step), np.floor(x[-1]/step)) * step
    else:
        t_uniform = np.arange(t_range[0]/step, (t_range[1]/step) + 1) * step

    if (len(x) <= 1) or (len(y) <= 1):
        return t_uniform, y
    else:

        # The following avoids some errors but probably only delays them till a later step
        #
        # try:
        #     f = interp.interp1d(x, y, kind=kind, fill_value=fill_value)
        # except:
        #     next_kind = 'quadratic'
        #     newprint(f"interpolate(): '{kind}' interpolation of len(x) == {len(x)} failed. Using '{next_kind}'.", on_a_new_line=True)
        #     kind = next_kind
        #     try:
        #         f = interp.interp1d(x, y, kind=kind, fill_value=fill_value)
        #     except:
        #         next_kind = 'slinear'
        #         newprint(f"               '{kind}' interpolation of len(x) == {len(x)} failed. Using '{next_kind}'.", on_a_new_line=True)
        #         kind = next_kind
        #         f = interp.interp1d(x, y, kind=kind, fill_value=fill_value)

        f = interp.interp1d(x, y, kind=kind, fill_value=fill_value)
        
        y_uniform = f(t_uniform)

        return t_uniform, y_uniform


def resample_signal(y, original_rate, target_rate):
    """Resamples a signal from the original sample rate to the target sample rate.

    Args:
        y (ndarray): 1D array of the original signal.
        original_rate (float): Original sample rate in Hz.
        target_rate (float): Target sample rate in Hz.

    Returns:
        y_resampled (ndarray): 1D array of the resampled signal.
    """
    original_length = len(y)
    original_duration = original_length / original_rate
    target_length = int(original_duration * target_rate)

    y_resampled = signal.resample(y, target_length)

    return y_resampled


# For references on scaling the signal, see "method" under MATLAB's documentation:
# https://www.mathworks.com/help/matlab/ref/double.normalize.html
def normalize(x: np.ndarray, range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """Normalize x to range[0] and range[1]. If range is None, normalize to 0 and 1."""
    a, b = range
    return a + (x - np.min(x)) / (np.max(x) - np.min(x)) * (b - a)


def standardize(signal: np.ndarray[float]):
    """Standardize a signal by doing (signal-mean)/stdev."""
    warnings.filterwarnings("error", category=RuntimeWarning)
    range = np.max(signal) - np.min(signal)
    if range == 0:
        # warnings.warn("util.standardize(): The signal's range is 0, so it won't be standardized.")
        return signal
    else:
        signal = (signal - np.mean(signal)) / range
        try: 
            sig_sd = np.std(signal)
            # Prior to the above downscaling, this sometimes caused an overflow warning. It shouldn't anymore.
        except:
            warnings.warn("util.standardize(): np.std(signal) caused an error, so we will not divide by the standard deviation.")
            return 3 * signal
        else:
            if sig_sd == 0:
                sig_sd = 0.001  # avoid div/0 error
            return signal / sig_sd

def moving_avg(y: np.ndarray, window_len: int = 5):
    """Calculate moving average on given signal, using window size window_len."""
    b = (1 / window_len) * np.ones(window_len)
    a = np.ones(1)

    y_temp = np.copy(y)

    # Interpolate NaN values using linear interpolation, arbitrary choice
    nan_indices = np.isnan(y_temp)
    if np.any(nan_indices):
        x = np.arange(len(y_temp))
        y_temp[nan_indices] = np.interp(
            x[nan_indices], x[~nan_indices], y_temp[~nan_indices]
        )

    # Interpolate infinite values using nearest neighbor interpolation, arbitrary choice
    inf_indices = np.isinf(y_temp)
    if np.any(inf_indices):
        y_temp[inf_indices] = np.nan
        y_temp = np.nan_to_num(y_temp, nan=np.nanmedian(y_temp))

    return signal.lfilter(b, a, y_temp)[window_len - 1 :]


def isin_tolerance(A, B, tol):
    # Like numpy.isin() but with a tolerance instead of exact.
    # Source: https://stackoverflow.com/a/51747164

    A = np.asarray(A)
    B = np.asarray(B)

    Bs = np.sort(B) # skip if already sorted
    idx = np.searchsorted(Bs, A)

    linvalid_mask = idx==len(B)
    idx[linvalid_mask] = len(B)-1
    lval = Bs[idx] - A
    lval[linvalid_mask] *=-1

    rinvalid_mask = idx==0
    idx1 = idx-1
    idx1[rinvalid_mask] = 0
    rval = A - Bs[idx1]
    rval[rinvalid_mask] *=-1
    return np.minimum(lval, rval) <= tol


def subset(xsubset: np.ndarray, xfull: np.ndarray, yfull: np.ndarray, sampling_rate: float):
    # Return an ndarray of the ysubset that corresponds to xsubset in xfull (within a tolerance) and yfull. 
    # This is useful to find the y values of a signal that correspond to breath markings or other events.

    if len(xsubset) == 0:
        return None
    
    if len(xsubset) != len(np.unique(xsubset)):
        print("util.subset(): Warning: xsubset contains repeated values, which will likely cause an error. Run np.unique() on your xsubset.")

    if len(xfull) != len(yfull):
        print(f"util.subset(): Warning: Array lengths do not match, which will likely cause an error.")
        print(f"               len(xfull): {len(xfull)}")
        print(f"               len(yfull): {len(yfull)}")

    # We start with the faster test of perfect matching. If that doesn't yield the expected number
    # of matches, we try a looser test. If that fails too, we raise an error.
    mask = np.isin(xfull, xsubset)
    if np.count_nonzero(mask) < (len(xsubset)):         
        mask = isin_tolerance(xfull, xsubset, 1/(sampling_rate * 4))
        if np.count_nonzero(mask) < (len(xsubset)):     # Recently I used len() - 1 to solve a problem I didn't understand, but I don't think that's necessary now
            print("")   # blank line
            print("util.subset(): Beginning of error log. Investigating why too few matches were found.")
            print("")
            for x in xsubset:
                if (x in xfull):
                    print (f"Good: xsubset value {x} found in xfull.")
                else:
                    print (f"PROBLEM: xsubset value {x} not found in xfull!!!")
            raise ValueError(f"Of {len(xsubset)} values in xsubset, {len(np.unique(xsubset))} of which are unique, only {np.count_nonzero(mask)} exact matches were found in xfull.")
    return yfull[mask]


def closest_to(arr: np.ndarray, target: float):
    """Returns the value, idx of the value in array cloest to the given target."""
    assert isinstance(target, float) | isinstance(
        target, int
    ), f"target {target} must be a float or int"

    idx = np.argmin(np.abs(arr - target))
    return arr[idx], idx

def round_to_multiple(x: np.array, base, precision):
    # Round to the nearest multiple of the base, such as 0.04 to align with a 25 Hz timebase
    # Sources: https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python,
    # https://stackoverflow.com/questions/28425705/python-round-a-float-to-nearest-0-05-or-to-multiple-of-another-float 
    
    return np.around(base * np.around(x / base), precision)


def lowpass_butter(
    signal_raw: np.ndarray,
    signal_fs: float,
    f_max: float = 4,
    order: int = 5,
    use_reflections: bool = False,
    show: bool = False,
):
    """Low pass zero-phase filtering with a butterworth filter.

    This function is retained since it has been widely used in this project, but the Bessel filter,
    defined in lowpass_bessel() below, seems to be more appropriate.

    [b,a]=butter(order,[ppg_fMin, ppg_fMax]/(fs/2))

    For pre-processing PPGs, this code does not filter out very-low frequencies as
    recommended in literature due to difficulties when the cutoff frequency is
    close to zero.

    Args:
        signal_raw: Signal to be filtered
        signal_fs: Signal's sampling freq [hz]
        f_max: Cutoff frequency for the filter [hz]
        order: Filter order; default is 5
        use_reflections: If true, the signal is wrapped in its reflections to reduce
            edge effects.
        show: If True, displays the signal before and after filtering.

    Returns:
        Filtered signal (numpy array)

    It is recommend to bandpass filter between 0.05 Hz and 4 Hz with 6th
    order butterworth filter. For more information, see:
    Charlton et al., An assessment of algorithms to estimate respiratory rate
    from the electrocardiogram and photoplethysmogram. Physiol. Meas. 37,
    610-626 (2016).
    """
    sos = signal.butter(order, f_max, "lowpass", fs=signal_fs, output="sos")
    
    if use_reflections:
        signal_filt = filter_signal_with_reflection(sos, signal_raw)
    else:
        signal_filt = signal.sosfiltfilt(sos, signal_raw)

    # ================================================================
    # remove peaks too close to 0 - filtering will cause zero-signal segments
    # to fluctuate about 0 amplitude. These are artificial peaks.
    signal_filt[np.abs(signal_filt) < 1e-14] = 0

    if show:
        import plotly.graph_objects as go

        # Add raw signal trace
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=signal_raw, name="Raw Signal", line={"color": "firebrick", "width": 0.6}
            )
        )

        # Add filtered signal trace
        fig.add_trace(
            go.Scatter(
                y=signal_filt,
                name="Filtered Signal",
                line={"color": "royalblue", "width": 2},
            )
        )

        fig.update_layout(
            title=f"Low-pass Butterworth filter, order {order}, at {f_max:.2f} Hz"
        )
        fig.show()

    return signal_filt


def lowpass_bessel(
    signal_raw: np.ndarray,
    signal_fs: float,
    f_max: float = 20,
    order: int = 20,
    use_reflections: bool = False,
    show: bool = False,
):
    """Low pass zero-phase filtering with a Bessel filter, which is maximally flat in phase,
    preserving shape of the PPG better than the Butterworth.

    This function was adapted from lowpass_butter(). 
    
    For pre-processing PPGs, this code does not filter out very-low frequencies as
    recommended in literature due to difficulties when the cutoff frequency is
    close to zero.

    Args:
        signal_raw: Signal to be filtered
        signal_fs: Signal's sampling freq [Hz]
        f_max: Cutoff frequency for the filter [Hz]; default is 20 Hz
        order: Filter order; default is 20
        use_reflections: If true, the signal is wrapped in its reflections to reduce
            edge effects.
        show: If True, displays the signal before and after filtering.

    Returns:
        Filtered signal (numpy array)

    It is recommend to bandpass filter between 0.05 Hz and 4 Hz with 6th
    order butterworth filter. For more information, see:
    Charlton et al., An assessment of algorithms to estimate respiratory rate
    from the electrocardiogram and photoplethysmogram. Physiol. Meas. 37,
    610-626 (2016). [Note from Wallace: I don't see these frequencies in the article, 
    but the authors did high-pass at 5 bpm [0.08 Hz] and low-pass at 35 Hz, without 
    details of the filters.]
    """
    sos = signal.bessel(order, f_max, "lowpass", fs=signal_fs, output="sos")

    if use_reflections:
        signal_filt = filter_signal_with_reflection(sos, signal_raw)
    else:
        signal_filt = signal.sosfiltfilt(sos, signal_raw)

    # ================================================================
    # remove peaks too close to 0 - filtering will cause zero-signal segments
    # to fluctuate about 0 amplitude. These are artificial peaks.
    signal_filt[np.abs(signal_filt) < 1e-14] = 0

    if show:
        import plotly.graph_objects as go

        # Add raw signal trace
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=signal_raw, name="Raw Signal", line={"color": "firebrick", "width": 0.6}
            )
        )

        # Add filtered signal trace
        fig.add_trace(
            go.Scatter(
                y=signal_filt,
                name="Filtered Signal",
                line={"color": "royalblue", "width": 2},
            )
        )

        fig.update_layout(
            title=f"Low-pass Bessel filter, order {order}, at {f_max:.2f} Hz"
        )
        fig.show()

    return signal_filt


def extend_with_reflections(reflected_proportion: float = 0.6):
    def decorator(func):
        def wrapper(raw_signal, *args, **kwargs):
            """Wrap the signal with reflections before the function call.

            Extending the signal with reflections before filtering helps reduce edge
            effects (i.e., ripples).

            Reflections are then trimmed after the function call.
            """
            # Length of the input signal
            signal_length = len(raw_signal)
            reflection_size = int(signal_length * reflected_proportion)
            mirror_offset = signal_length - reflection_size

            # Extend the signal by mirroring with only the necessary reflection size
            mirrored_signal = np.concatenate(
                (
                    raw_signal[-mirror_offset::-1],
                    raw_signal,
                    raw_signal[:mirror_offset:-1],
                )
            )

            # Call the wrapped function
            filtered_signal = func(mirrored_signal, *args, **kwargs)

            # Crop the result to the original signal length
            crop_start = reflection_size + 1
            cropped_signal = filtered_signal[crop_start : crop_start + signal_length]
            return cropped_signal

        return wrapper

    return decorator


@extend_with_reflections(reflected_proportion=0.2)
def medfilt_with_reflections(y: np.ndarray, kernel_size: int):
    """Wrap the signal "y" with its own reflections to reduce edge effects.

    Args:
        y: Signal to process
        kernel_size: Size of the filter. Expects an odd number.

    Returns:
        Filtered signal.
    """
    return signal.medfilt(y, kernel_size=kernel_size)


def filter_signal_with_reflection(
    sos_coeffs, raw_signal, reflected_proportion: float = 0.6
):
    """Filter a signal using the signal reflection trick and sosfiltfilt.

    Reflections of the signal are concatenated to the ends of the original signal to
    minimize the ripping edge effects from filtering.

    Args:
        sos_coeffs (numpy.ndarray): Second-order section (SOS) filter coefficients.
        raw_signal (numpy.ndarray): The input signal.
        reflected_proportion (float): The proportion of signal to be reflected.
            Default to 0.6.

    Returns:
        numpy.ndarray: The filtered signal with edge effects mitigated.
    """
    # Length of the input signal
    signal_length = len(raw_signal)
    reflection_size = int(signal_length * reflected_proportion)
    mirror_offset = signal_length - reflection_size

    # Extend the signal by mirroring with only the necessary reflection size
    mirrored_signal = np.concatenate(
        (raw_signal[-mirror_offset::-1], raw_signal, raw_signal[:mirror_offset:-1])
    )

    # Apply zero-phase filtering using sosfiltfilt
    filtered_signal = signal.sosfiltfilt(sos_coeffs, mirrored_signal)

    # Crop the result to the original signal length
    crop_start = reflection_size + 1
    cropped_signal = filtered_signal[crop_start : crop_start + signal_length]

    return cropped_signal


def normalized_xcorr(a: np.ndarray, b: np.ndarray):
    """Normalized cross-correlation, designed to be equal to the matlab version.

    Ensures the output to be between -1 and 1

    Sources:
    * https://stackoverflow.com/a/71005798/12985130
    * https://www.mathworks.com/help/matlab/ref/xcorr.html
    """
    norm_a = np.linalg.norm(a)
    a = a / norm_a
    norm_b = np.linalg.norm(b)
    b = b / norm_b
    return np.correlate(a, b, mode="full")


def sigmoid(x, center: float = 0, slope: float = 1, L: float = 1):
    """Sigmoidal function with shape control.

    see: https://en.wikipedia.org/wiki/Logistic_function

    Args:
        x (np.array): x-axis
        center: center of sigmoid
        slope: 1 is default; 0 makes a flat line; larger = steeper.
            can be negative (flips the shape along the center so that sigmoid goes from
            1 to 0)
        L: The supermum, to which the sigmoid approaches as x -> infinity
    """
    return L / (1 + np.exp(slope * (center - x)))


def mode_within_tolerance(arr: np.array, tol: float = None, percent_tol: float = None):
    """Find the most common number in an array within a certain percentage tolerance.

    Args:
    arr (np.array): The input array.
    tol (float): Tolerance within which to consider the numbers as equal.
    percent_tol (float): The percentage tolerance within which to consider numbers as equal.

    Returns:
    The number that occurs most frequently within the specified percentage tolerance.
    """
    arr = np.array(arr)

    if tol is not None:
        # Round the array elements to the nearest multiple of their respective tolerances
        rounded_arr = np.round(arr / tol) * tol

        counts = Counter(rounded_arr)

        # Find the most common element
        mode = counts.most_common(1)[0][0]
        return mode

    elif percent_tol is not None:
        # Create an array to store the counts for each element
        counts = np.zeros_like(arr)

        # For each element in the array, count how many elements fall within the tolerance range
        lower_bound = arr - arr * percent_tol / 100
        upper_bound = arr + arr * percent_tol / 100

        for i, x in enumerate(arr):
            lower_bound = x - x * percent_tol / 100
            upper_bound = x + x * percent_tol / 100
            counts[i] = np.sum((arr >= lower_bound) & (arr <= upper_bound))

        idx_with_max_counts = np.argwhere(counts == np.amax(counts))

        # Average of the elements with the highest counts
        return np.mean(arr[idx_with_max_counts])


def pivot_and_merge(references, qualities, candidates):
    """Merge the three given dataframes into a format used by ML models.

    - combines trial number and frame number
    - combines method and feature into "method-feature"
    - combines kalman estimation to features dataframe (X)
    - includes rows that have no qualities or estimates

    Args:
        references: the df dataframe output from evaluate_dataset()
        qualities: df containing quailty metrics
        candidates: df containing candidates

    Usage:
        X = pivot_and_merge(
            references=df,
            qualities=quality_indices,
            candidates=rr_candidates
            )

    """
    if type(qualities) is not pd.core.frame.DataFrame:
        qualities = pd.DataFrame(qualities)
    if type(candidates) is not pd.core.frame.DataFrame:
        candidates = pd.DataFrame(candidates)

    # fmt: off
    qualities['trial-frame'] = qualities['trial'].astype(str) + '-' + qualities['frame index'].astype(str)
    qualities['method-feature'] = 'quality - ' + qualities['method'] + '-' + qualities['feature']
    qualities_p = qualities.pivot(index='trial-frame', columns='method-feature', values='value')

    if not candidates.empty:
        candidates['trial-frame'] = candidates['trial'].astype(str) + '-' + candidates['frame index'].astype(str)
        candidates['method-feature'] = 'candidate - ' + candidates['method'] + '-' + candidates['feature']
        candidates_p = candidates.pivot(index='trial-frame', columns='method-feature', values='RR candidate')

    references['trial-frame'] = references['trial'].astype(int).astype(str) + '-' + references['frame index'].astype(str)
    # fmt: on

    if candidates.empty:
        all_data = references.merge(
            right=qualities_p, how="left", on="trial-frame"
        )
    else:
        all_data = references.merge(right=candidates_p, how="left", on="trial-frame").merge(
            right=qualities_p, how="left", on="trial-frame"
        )
    all_data = all_data.set_index("trial-frame")

    return all_data


# Alternative to print() that allows control of whether each call should begin on a new line
_last_print_ended_with_newline = True

def newprint(*args, on_a_new_line=False, **kwargs):
    global _last_print_ended_with_newline

    if on_a_new_line and not _last_print_ended_with_newline:
        print()  # Print a newline

    # Now print the actual content
    print(*args, **kwargs)

    # Capture the ending character (default is '\n')
    end_char = kwargs.get('end', '\n')
    
    # Update the flag for the next call
    _last_print_ended_with_newline = (end_char == '\n')


# As an alternative to defaultdict(dict), use this, which allows direct assignment at any level. 
# Example usage: ref = nested_dict()
#
# Warning: Reading a non-existent key will create that key rather than throw an error.

nested_dict = lambda: defaultdict(nested_dict)  


def key_exists(data, keys):
    """Check if a nested key exists in a dictionary.

    Args:
        data: The dictionary to check.
        keys: A list of keys representing the nested path.

    Returns:
        True if the nested key exists, False otherwise.
    """

    current = data
    for key in keys:
        if key not in current:
            return False
        current = current[key]
    return True