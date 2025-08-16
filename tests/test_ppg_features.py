import pandas as pd
import numpy as np
from ppg2rr import riv_est

def _test_dicrotic_notch():
    """Are dicrotic notch locations correctly identified?
    
    NOTE: This test needs further development; unsure if the functionality is actually 
    going to be used.
    """
    ppg = pd.read_csv('tests/resources/short_segment_mimic1.csv')
    ppg = ppg['0'].values
    
    # fmt: off
    minima = [6,87,168,255,332,414,497,576,660,743,829,884,990,1068,1156,1237,1316]   
    expected_idxs = [57,139,221,301,384,466,549,632,713,793,881,959,1042,1127,1209,1291]
    # fmt: on
    
    notch_idxs = _loop_through_each_pulse(ppg,minima)
    
    assert notch_idxs == expected_idxs

def _loop_through_each_pulse(ppg, min_idxs):
    """Loop through each individual ppg pulse"""
    pulse_starts = min_idxs[:-1]
    pulse_ends = min_idxs[1:]
    
    notch_idxs = []
    for pulse_start, pulse_end in zip(pulse_starts, pulse_ends):
        pulse = ppg[pulse_start-2:pulse_end+2]
        notch_idx = riv_est.get_dicrotic_notch(pulse=pulse)
        if notch_idx is not None:
            # add 2 to account for taking the difference twice
            notch_idxs.append(notch_idx+pulse_start+2)
        
    return notch_idxs