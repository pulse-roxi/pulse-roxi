import numpy as np
import pandas as pd
import pytest

from ppg2rr import heart_rate as hr

resource_dir = "tests/resources/ppg_waveforms"

@pytest.mark.parametrize(
    "ppg_file, expected_pulse_rate",
    [
        (f"{resource_dir}/ppg_kapiolanip1-91-3.csv", 69.58),
        (f"{resource_dir}/ppg_kapiolanip1-8-29.csv", 75.98),
        (f"{resource_dir}/ppg_kapiolanip1-46-3.csv", 86.97),
        (f"{resource_dir}/ppg_kapiolanip1-62-19.csv", 131.83),
        (f"{resource_dir}/ppg_kapiolanip1-23-21.csv", 135.49),
        (f"{resource_dir}/ppg_kapiolanip1-65-24.csv", 158.38),
        (f"{resource_dir}/ppg_kapiolanip1-22-32.csv", 97.96),
        (f"{resource_dir}/ppg_kapiolanip1-39-11.csv", np.nan), #condition 3
        (f"{resource_dir}/ppg_kapiolanip1-8-3.csv", np.nan), # condition 3
        (f"{resource_dir}/ppg_kapiolanip1-67-15.csv", np.nan), # condition 3
        (f"{resource_dir}/BMA400_2024-01-17_16-26-59_nellcor_off_finger.csv", np.nan), # condition 1
        (f"{resource_dir}/BMA400_2024-01-17_16-24-40_Masimo_off_finger.csv", np.nan), # condition 1
    ],
)
def test_avg_heart_rate_cond2(ppg_file, expected_pulse_rate):
    """Ensures estimated pulse rates are as expected.
    
    Tests the following:
    Condition 1: No PSD peak in the heart rate region -> Device potentially off finger
    Condition 2: Prominant PSD peak in the heart rate region -> Heart rate detected
    Condition 3: Multiple PSD peaks in the heart rate region or 
        dominant peak below heart rate region -> Noisy
    """
    ppg_df = pd.read_csv(ppg_file)
    if "BMA400" in ppg_file:
        ppg = ppg_df['ppg1'].values
        sampling_rate = 250
    else:
        ppg = ppg_df["0"].values
        sampling_rate = 125

    
    estimated_hr = hr.estimate_avg_heart_rate(
        ppg=ppg,
        sampling_rate=sampling_rate,
        min_heart_rate_hz=60/60,
        show=False
    )
    
    if expected_pulse_rate is np.nan:
        assert estimated_hr is np.nan
    else:
        assert np.isclose(
            expected_pulse_rate, estimated_hr, atol=1
        )