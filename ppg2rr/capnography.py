"""Functions for handling Capnography, including calculating derived values."""

import numpy as np
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from scipy import signal

from ppg2rr import util


def rr_from_capnography(
    co2: np.ndarray[float],
    fs_co2: float,
    f_max: float = 4,
    show: bool = False,
):
    """Estimate respiration rate from capnography signal.

    Algo steps:
    1. Low-pass filter the input co2 signal.
    2. Find local minima, hopefully corresponding to the start of each breath in the
    capnography trace.
        * Minimal distance between peaks is set to 15 samples. Assuming
        fs_co2 is 62.5 samples/second, this corresponds to maximum breathing rate of
        250 breaths per minute, which is much higher than physiological "normal".
        * Peak minimum prominence is set to 5% of capnography value range (or maybe it was, once upon
        a time; see the actual value below).
    3. Difference in local minima = breathing rate

    Args:
        co2 (np.ndarray[float]): capnography trace values.
        fs_co2 (float): Sampling frequency of capnography trace.
        f_max (float): Corner frequency for filter.
        show (bool, optional): Defaults to False.

    Returns:
        rr_x (np.ndarray) - Times where rates were calculated.
        rr_y (np.ndarray) - Estimated instantaneous respiratory rate over time.
        breaths_x (np.ndarray)  - Times of breath starts (the first breath start is not included in the
            above array of rates).
        breaths_y (np.ndarray) - co2 values at each breath start (for plotting).
    """
    # parameters for peak finding
    co2_range = max(co2) - min(co2)
    peak_prominence = co2_range * 0.2

    # rate finding algorithm
    co2_filtered = util.lowpass_butter(co2, fs_co2, f_max)
    co2_min_idx, _ = signal.find_peaks(
        -co2_filtered + min(-co2_filtered),
        # TODO: Make the distance proportional to the sample rate
        distance=15,
        prominence=peak_prominence,
    )
    samples_per_breath = np.diff(co2_min_idx)

    rr_y = fs_co2 * 60 / samples_per_breath
    rr_x = (co2_min_idx[1:]) / fs_co2

    assert len(rr_x) == len(rr_y)

    breaths_y = co2_filtered[co2_min_idx]
    breaths_x = co2_min_idx / fs_co2

    if show:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.7, 0.3],  # relative heights of each row
            x_title='Time (seconds)',
        )
        xco2 = np.arange(len(co2))/fs_co2
        fig.add_trace(
            go.Scatter(
                x = xco2,
                y=co2,
                marker={"color": "grey", "size": 15},
                name="co2 trace",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x = np.arange(len(co2_filtered))/fs_co2,
                y=co2_filtered,
                marker={"color": "#90A4AE", "size": 15},
                name="co2 trace (filtered)",
            ),
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=co2_min_idx/fs_co2,
            y=co2_filtered[co2_min_idx],
            mode="markers",
            name="capno breath start",
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=co2_min_idx[1:]/fs_co2,
                y=rr_y,
                mode="lines",
                name="capnography resp rate",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title="RR calculation from Capnography",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        fig.update_xaxes(range=[min(xco2), max(xco2)])
        fig.show()

    return rr_x, rr_y, breaths_x, breaths_y


def rolling_window_count(times: np.array, window_length: int):
    """Count the number of breaths in a rolling window.
    
    If the given window length is shorter than the largest breathing time, 
    the length of breath times is returned.
    
    Args:
        times (list-like): A list of timestamps (seconds) of breaths. 
        window_length (int): The size of the rolling window (seconds).
    
    returns:
        Numpy array of time stamps from 0 up to the final breathing time stamp.
        Numpy array containing the number of breaths in the past window_length seconds.
    """    
    num_breaths = []
    t_ends = range(np.ceil(times[-1]).astype('int'))
    for t_end in t_ends:
        t_start = np.max([0,t_end-window_length])
        num_breaths.append(
            np.sum((t_start <= times) & (times <= t_end)) * (60/window_length)
            )
        
    return np.array(t_ends), np.array(num_breaths)