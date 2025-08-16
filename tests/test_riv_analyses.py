"""Functional tests for ppg2rr.py."""

import neurokit2 as nk
import numpy as np
import pandas as pd
import pytest

from ppg2rr import riv_analyses, riv_est


# fmt: off
@pytest.mark.parametrize(
    "trial_num", 
    [16, 62, 73, 32, 41, 86, 0, 9, 59],
)
# fmt: on
def test_psd_analysis_synthetic(trial_num):
    """Does the RR estimated via PSD analysis approximate the expected value?
    
    Uses the synthetic dataset to do the testing, which manipulates one of bw, fm, or am
    in a waveform to impart "breathing" information. Three waveforms from each 
    group are chosen.
    
    Pass:
        psd analysis returns RR within 1 breath per minute of the expected RR.
    """
    synth = pd.read_parquet("data/Synthetic/synthetic_ppg.parquet")
    
    expected_rr = synth['rr_breathsPerMin'].iloc[trial_num]
    
    ppg = synth["ppg_v"][trial_num]
    hr = synth.iloc[trial_num]["hr_beatsPerMin"]
    fs = synth.iloc[trial_num]["ppg_fs"]

    rivs, _ = riv_est.extract_rivs(
        ppg=ppg,
        fs=fs,
        hr=hr,
    )

    rivs = riv_analyses.psd_analysis(
        rivs=rivs, 
        fs_riv=10,
        show=False,
        )

    rr_ests = [round(value["max PSD freq"] * 60, 2) for value in rivs.values()]

    assert np.isclose(np.median(rr_ests), expected_rr, atol=1)


def test_autocorr_norm():
    """Tests normalized autocorrelation.

    Case:
    * a perfectly repeating pattern
    """
    data = np.tile([3, 2, 1, 0], 6)
    acfn = riv_analyses.autocorr_norm(data, n_lags=7)

    # fmt: off
    assert np.all(np.isclose(
        acfn,
        np.array([1.0, -0.13043478, -0.6, -0.27619048, 1.0, -0.11578947, -0.6,])
        ))
    # fmt: on


def peak_counting_analysis():
    # case where no peaks are found
    test_data = {"key1", {"x": np.arange(50), "y": np.arange(50)}}
    observed = riv_analyses.peak_counting_analysis(test_data)
    assert np.isnan(observed["key1"]["pk_count RR num pks"])


def test_kalman_analysis():
    """Tests the kalman analysis code with a simple simulaged PPG signal."""
    ppg = nk.ppg_simulate(
        duration=15,
        heart_rate=70,
        sampling_rate=125,
        random_state=1234,
    )

    rivs, _ = riv_est.extract_rivs(
        ppg=ppg,
        fs=125,
        hr=70,
        rr_max=150,
        remove_outliers="None",
        show=False,
    )

    rivs_psd = riv_analyses.psd_analysis(
        rivs, 
        fs_riv=10,
        show=False
        )
    
    del rivs_psd["ppg"]  # remove ppg from list of candidates
    rr, quality = riv_analyses.kalman_analysis(
        rivs_psd=rivs_psd, 
        n_sig=2, 
        fs_riv=10, 
        fmin=0.13, show=False
    )

    assert np.isclose(rr, 15.52734375, atol=0.5)
    assert np.isclose(quality, 0.10996439, atol=0.5)

@pytest.mark.parametrize("acf_freqs, sqis, psd_freq, expected_result", [
    ([9.84, 7.59, 4.76, 5.56],[0.6, 0.16, 0.08, 0.0], 20, 9.92),  # 2 tones, PSD is f1
    ([10, 5],[0.66, 0.4], 20, 10), # 2 tones, PSD is f1
    ([10.71, 5.26],[0.56, 0.44],30,10.355), # 2 tones, PSD is f2
    ([5, 20],[0.4, 0.25], 20, 20), # 2 tones, PSD is f0
    ([10, 5, 20],[0.66, 0.4, 0.25], 10, 10), # 2 tones, PSD is f0
    ([10, 5, 20],[0.66, 0.4, 0.25],25,None), # 2 tones, weird PSD peak
    ([5.17, 10.0, 4.08, 20.0, 6.82],[0.57, 0.55, 0.45, 0.33, 0.21],20,10), # 2 tones
    ([9.84, 4.8, 27.27, 15.0, 7.06, 5.94],[0.68, 0.52, 0.17, 0.16, 0.13, 0.1],30,9.92), # 3 toned
    ([9.68, 4.72, 5.88, 26.09, 15.38],[0.59, 0.45, 0.08, 0.05, 0.02],30,9.84),  # 3 toned
    ([9.84, 4.96, 30.0, 4.26, 7.41, 15.0],[0.89, 0.86, 0.84, 0.82, 0.79, 0.78],30,30), # 1 puretone
    ([4.92, 9.84, 7.32, 28.57, 5.88, 14.63],[0.82, 0.77, 0.75, 0.69, 0.63, 0.63],30,28.57), # 1 puretone
    ([7.23, 9.84, 28.57, 4.92, 5.83, 14.63],[0.76, 0.74, 0.59, 0.59, 0.38, 0.35],30,28.57), # 1 puretone
    ([6.38, 4.72, 17.14, 10.0],[0.53, 0.46, 0.43, 0.3],18.75,18.75)
])
def test_get_f0(acf_freqs, sqis, psd_freq, expected_result):
    assert riv_analyses.get_f0(
        psd_freq, 
        acf_freqs=acf_freqs, 
        acf_sqi=sqis,
        tolerance=psd_freq/10,
        ) == expected_result
    
    
    	