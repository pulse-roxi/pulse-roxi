"""unit and function tests for functions in riv.py."""

import neurokit2 as nk
import numpy as np
import pandas as pd
import pytest

from ppg2rr import riv_est, util


@pytest.fixture
def load_synthetic_ppgs():
    return pd.read_parquet("data/Synthetic/synthetic_ppg.parquet")


def generate_outliers(median=5, std=2, outlier_std=10, array_len=80, num_outliers=5):
    """Generate data points with outliers to test outlier removal."""
    np.random.seed(0)  # repeatable
    errs = std * np.random.rand(array_len) * np.random.choice((-1, 1), array_len)
    data = median + errs

    lower_errs = outlier_std * np.random.rand(num_outliers)
    lower_outliers = median - std - lower_errs

    upper_errs = outlier_std * np.random.rand(num_outliers)
    upper_outliers = median + std + upper_errs

    data = np.concatenate((data, lower_outliers, upper_outliers))
    np.random.shuffle(data)

    return data


@pytest.mark.parametrize(
    "zero_start, zero_end, expected_maxs, expected_mins, win_tolerance_pct",
    [
        (
            1250,
            -1,
            [19, 129, 237, 354, 466, 584, 707, 826, 937, 1047, 1159],
            [108, 216, 333, 444, 563, 687, 804, 914, 1027, 1138],
            0.5,
        ),
        (
            500,
            1350,
            [19, 129, 237, 354, 466, 1376, 1479, 1578, 1681, 1777],
            [108, 216, 333, 444, 1359, 1461, 1561, 1663, 1759],
            0.5,
        ),
        (
            0,
            800,
            [826, 937, 1047, 1159, 1274, 1376, 1479, 1578, 1681, 1777],
            [804, 914, 1027, 1138, 1254, 1359, 1461, 1561, 1663, 1759],
            0.5,
        ),
        (
            1250,
            -1,
            [19, 129, 237, 354, 466, 965, 1080, 1192],
            [108, 216, 333, 444, 851, 962, 1072, 1185],
            0.1,
        ),
    ],
)
def test_find_extrema_HX2017(
    zero_start, zero_end, expected_maxs, expected_mins, win_tolerance_pct
):
    """Test the implementation of the HX2017 algorithm using simulated PPG signal.

    Tested using:
    * A regular ppg signal
    * PPG signal with zeros in the center
    * PPG signal with zeros in the beginning
    * Small win_tolerance_pct causing wrong peaks to be detected
    """
    ppg = nk.ppg_simulate(
        duration=15,
        heart_rate=70,
        sampling_rate=125,
        random_state=1234,
    )
    ppg[zero_start:zero_end] = 0
    maxs, mins = riv_est.find_extrema_HX2017(
        ppg,
        fs=125,
        hr_bpm=70,
        win_tolerance_pct=win_tolerance_pct,
    )

    assert np.all(mins == expected_mins)
    assert np.all(maxs == expected_maxs)


def test_detect_outliers():
    """Ensures outliers are detected correctly, test values of function parameters."""
    test_arr = generate_outliers()

    outliers3 = riv_est.detect_outliers(test_arr, 0.5)
    outliers5 = riv_est.detect_outliers(test_arr, 0.5, 5)
    outliers10 = riv_est.detect_outliers(test_arr, 10)

    assert list(np.where(outliers3 == 1)[0]) == [6, 13, 22, 33, 44, 49, 68, 70, 84]
    assert list(np.where(outliers5 == 1)[0]) == [13, 44, 49, 68, 70]
    assert list(np.where(outliers10 == 1)[0]) == [13]


def test_one_max_per_min():
    """Ensures one max per min is detected correctly.

    Dummy signal & min/max indices contains
    * start with more than one maxima
    * middle with 2 consecutive maxima
    * end with a maxima
    * middle with 2 consecutive minima
    * flips the signal so the opposite is true
    """
    y = np.tile([-1, 0, 0, 1, 0, 0], reps=6)
    max_idxs = np.array([1, 3, 9, 21, 27, 29, 33, 34])
    min_idxs = np.array([6, 12, 18, 24, 30])

    max_idxs_paired, _ = riv_est.one_x_per_y(
        signal=y, x_locs=max_idxs, y_locs=min_idxs, keep="max"
    )
    assert list(max_idxs_paired) == [3, 9, 21, 27, 33]

    min_idxs_paired, _ = riv_est.one_x_per_y(
        signal=y, y_locs=max_idxs_paired, x_locs=min_idxs, keep="min"
    )
    assert list(min_idxs_paired) == [6, 12, 24, 30]

    # now test inverted signal, which starts and ends with minima
    y_inv = -y
    max_idxs_inv = min_idxs
    min_idxs_inv = max_idxs
    max_idxs_paired, _ = riv_est.one_x_per_y(
        signal=y_inv, x_locs=max_idxs_inv, y_locs=min_idxs_inv, keep="max"
    )
    assert list(max_idxs_paired) == [6, 12, 24, 30]

    min_idxs_paired, _ = riv_est.one_x_per_y(
        signal=y_inv, y_locs=max_idxs_paired, x_locs=min_idxs_inv, keep="min"
    )
    assert list(min_idxs_paired) == [3, 9, 21, 27, 33]


def test_audp():
    """Tests the area under descending peak logic."""
    y = np.tile([-1, 0, 1, 0], reps=6) + 1
    max_idx = np.array([2, 6, 10, 14, 18, 22])
    min_idx = np.array([4, 8, 12, 16, 20])

    # signal start and end with maxima
    audp, audp_loc = riv_est.est_AUDP(ppg=y, max_idxs=max_idx, min_idxs=min_idx)
    assert list(audp) == [3.0, 3.0, 3.0, 3.0, 3.0]
    assert list(audp_loc) == [2, 6, 10, 14, 18]

    # signal start with minima
    audp, audp_loc = riv_est.est_AUDP(ppg=y, max_idxs=max_idx[1:], min_idxs=min_idx)
    assert list(audp) == [3.0, 3.0, 3.0, 3.0]
    assert list(audp_loc) == [6, 10, 14, 18]

    # signal end with minima
    audp, audp_loc = riv_est.est_AUDP(ppg=y, max_idxs=max_idx[:-1], min_idxs=min_idx)
    assert list(audp) == [3.0, 3.0, 3.0, 3.0, 3.0]
    assert list(audp_loc) == [2, 6, 10, 14, 18]


@pytest.mark.parametrize(
    "maxs, mins, num_ascending, num_descending",
    [
        (np.array([2, 6, 10, 14, 18, 22]), np.array([4, 8, 12, 16, 20]), 5, 5),
        (np.array([6, 10, 14, 18, 22]), np.array([4, 8, 12, 16, 20]), 5, 4),
        (np.array([2, 6, 10, 14, 18]), np.array([4, 8, 12, 16, 20]), 4, 5),
        (np.array([6, 10, 14, 18]), np.array([4, 8, 12, 16, 20]), 4, 4),
    ],
)
def test_est_RIAV(maxs, mins, num_ascending, num_descending):
    """Unit test for RIAV (min-max amplitude) extraction.

    Checks that lenths of RIAV derived from the ascending and descending PPG pulse
    are as expected, using signals that starts/ends with min/max peaks (4 conditions).
    """
    ppg = np.tile([-1, 0, 1, 0], reps=6)

    ascending, _, descending, _ = riv_est.est_RIAV(
        ppg=ppg,
        max_idx=maxs,
        min_idx=mins,
    )

    assert len(ascending) == num_ascending
    assert len(descending) == num_descending


def test_interpolate():
    """Given a nonuniformly sampled signal, expect a uniformly sampled signal."""
    x = np.array([0, 1, 2, 6, 10])
    y = np.array([10, 5, 2.5, 0.5, 0.1])
    x_int, y_int = util.interpolate(x, y, kind="cubic")

    assert np.all(x_int == np.arange(0, 10))
    assert np.all(
        np.isclose(
            y_int,
            np.array(
                [
                    10.0,
                    5.0,
                    2.5,
                    1.14125,
                    0.49,
                    0.34375,
                    0.5,
                    0.75625,
                    0.91,
                    0.75875,
                ]
            ),
        )
    )


def test_resample_riv():
    """Tests the resample_riv function using a simple linear array.

    Check that the time-axis is correctly resampled.
    """
    t = np.arange(0, 60)
    y = np.linspace(0, 10, 60)
    fs1 = 1
    fs2 = 0.5
    t_resampled, _ = riv_est.resample_riv(t, y, fs1, fs2)

    assert np.all(t_resampled == np.arange(0, 60, 2))


def test_extract_rivs(load_synthetic_ppgs):
    """Function tests for riv extraction from simulated ppg.

    Asserts:
    * each RIV has a corresponding x and y value of equal length
    * ppg_feat exist for each RIV
    * each raw ppg_feat has x, y, and is_outlier arrays of equal length
    """
    ppgs = load_synthetic_ppgs
    ppg = ppgs["ppg_v"][0]
    hr = ppgs["hr_beatsPerMin"].iloc[0]
    fs = ppgs["ppg_fs"].iloc[0]

    rivs, ppg_feat = riv_est.extract_rivs(
        ppg=ppg,
        fs=fs,
        hr=hr,
    )

    assert list(rivs.keys()) == list(ppg_feat.keys())

    for key in rivs.keys():
        assert {"x", "y"}.issubset(rivs[key].keys())
        assert len(rivs[key]["y"]) == len(rivs[key]["x"])

    for feat in ppg_feat.values():
        assert len(feat["x"]) == len(feat["y"])
        assert len(feat["x"]) == len(feat["is_outlier"])


@pytest.mark.parametrize(
    "min_keep_pct,outlier_groups, expected_len",
    [
        (0.5, [(450, 499)], 451),
        (0.5, [(200, 499)], 500),
        (0.5, [(10, 50)], 449),
        (0.001, [(10, 50)], 460),
    ],
)
def test_cut_riv_outliers(min_keep_pct, outlier_groups, expected_len):
    """Tests cut_riv_outliers().

    Scenarios tested:
    * short segment of outliers at the end
    * looong segment of outliers
    * short segment of outliers at the front
    * short segment, but unrealistically small min_keep pct.
    """
    signal_len = 500
    test_x = np.arange(signal_len)
    test_y = np.random.rand(signal_len)

    cut_x, _, _ = riv_est.cut_riv_outliers(
        test_x, test_y, signal_len, min_keep_pct, outlier_groups
    )

    assert expected_len == len(cut_x)
