"""Unit and functional tests for signa_quality.py."""

import neurokit2 as nk
import numpy as np
import pandas as pd
import pytest

from ppg2rr import riv_est
from ppg2rr import signal_quality as sqi
from ppg2rr.util import lowpass_butter


@pytest.fixture
def simulated_ppg():
    fs = 125
    hr = 70
    ppg = nk.ppg_simulate(
        duration=30,
        heart_rate=hr,
        sampling_rate=fs,
        frequency_modulation=0.5,  # strength of modulation, from 0 to 1
        random_state=1234,
    )

    return ppg


@pytest.mark.parametrize(
    "ppg_file, pulse_rate, expected_consistency",
    [
        ("tests/resources/ppg_waveforms/ppg_kapiolanip1-39-11.csv", 102.54, 0.03773),
        ("tests/resources/ppg_waveforms/ppg_kapiolanip1-91-3.csv", 76.90, 0.25641),
        ("tests/resources/ppg_waveforms/ppg_kapiolanip1-8-29.csv", 73.24, 0.44444),
        ("tests/resources/ppg_waveforms/ppg_kapiolanip1-46-3.csv", 131.83, 0.52272),
        ("tests/resources/ppg_waveforms/ppg_kapiolanip1-62-19.csv", 131.83, 0.67692),
        ("tests/resources/ppg_waveforms/ppg_kapiolanip1-23-21.csv", 102.53, 0.77966),
        ("tests/resources/ppg_waveforms/ppg_kapiolanip1-65-24.csv", 157.47, 0.85897),
        ("tests/resources/ppg_waveforms/ppg_kapiolanip1-22-32.csv", 98.87, 0.96667),
    ],
)
def test_template_match_similarity(ppg_file, pulse_rate, expected_consistency):
    """Ensures that the percentage of diagnostic quality pulses is as expected.

    Tests are now ran using real ppg waveforms.
    """
    ppg = pd.read_csv(ppg_file)
    ppg = ppg["0"].values

    pulse_rate_estimate = pulse_rate
    sampling_rate = 125

    ppg_filtered = lowpass_butter(ppg, signal_fs=sampling_rate, f_max=4, show=False)
    max_idx_raw, _ = riv_est.find_extrema_HX2017(
        ppg_filtered,
        fs=sampling_rate,
        hr_bpm=pulse_rate_estimate,
        debug_flag=False,
    )

    avg_dist, _ = sqi.template_match_similarity(ppg_filtered, max_idx_raw, show=False)

    assert np.isclose(
        avg_dist["pct diagnostic quality pulses"], expected_consistency, 0.001
    )


@pytest.mark.parametrize(
    "outlier_locs, ppg_len, pct_outlier, center_artifact",
    [
        ([], 10, 0, False),
        ([0, 6], 10, 0, False),
        ([12, 14], 15, 2 / 15, False),
        ([4, 8], 10, 0.4, True),
        ([2, 4, 6, 8], 10, 0.6, True),
        ([0, 2, 7, 9], 10, 0.4, False),
        ([50, 52, 100, 101, 104, 300, 301, 800, 801, 802], 1000, 0.009, False),
    ],
)
def test_group_outlier_analysis(outlier_locs, ppg_len, pct_outlier, center_artifact):
    """Test whether outliers are correctly characterized as a whole.

    Case 1: no outliers
    Case 2: two single outlier spike (false positives)
    Case 3: true positive outliers present at end of frame
    Case 4: true positive outliers present in center of frame
    Case 5: more true positive outliers present in center of frame
    Case 6: two cluster of true positive outliers at the end of the frame
    Case 7: longer signal, small groups at the ends of the signal.
    """
    ppg_feat = {
        "feature 1": {
            "x": np.arange(10),
            "y": np.random.rand(10),
            "is_outlier": np.full(10, False),
        },
        "feature 2": {
            "x": np.array(outlier_locs),
            "y": np.random.rand(10),
            "is_outlier": np.full(len(outlier_locs), True),
        },
    }

    min_gap_len = 5
    actual = sqi.group_outlier_analysis(
        ppg_feat, ppg_len=ppg_len, min_gap_len=min_gap_len
    )
    assert np.isclose(actual["pct_frame_with_outliers"], pct_outlier)
    assert actual["artifact in center"] == center_artifact


def test_get_artifact_groups():
    # Generate a waveform with artifacts at known locations
    artifact_locations = [50, 52, 100, 101, 104, 200, 201, 800, 801, 802]

    # Define a threshold distance for grouping artifacts
    threshold = 5

    artifact_groups, artifact_durations = sqi.get_artifact_groups(
        artifact_locations=artifact_locations,
        threshold_distance=threshold,
        group_padding=0,
    )

    gaps = np.diff(artifact_locations)
    assert artifact_groups == [(50, 52), (100, 104), (200, 201), (800, 802)]
    assert artifact_durations == [2, 4, 1, 2]
    assert np.sum(artifact_durations) == gaps[gaps < threshold].sum()


@pytest.mark.parametrize(
    "outlier_locs,min_gap_len, expected",
    [
        ([], 5, []),
        ([6], 5, []),
        ([6, 30], 5, []),
        ([6, 30], 25, [6, 30]),
        ([6, 30, 31], 5, [30, 31]),
    ],
)
def test_remove_isolated_outliers(outlier_locs, min_gap_len, expected):
    """Ensure false positive outliers are removed.

    Test cases
    * Empty
    * Single outlier
    * Two false positives
    * Two outliers now belong in a "group" because min_gap_len is increased.
    * One outlier & a group.
    """
    observed = sqi.remove_isolated_outliers(outlier_locs, min_gap_len)

    assert expected == list(observed)
