import numpy as np
from scipy.signal import detrend
import math

def msptd_beat_detector(sig, fs):
    """
    MSPTD_BEAT_DETECTOR  MSPTD PPG beat detector.

        MSPTD_BEAT_DETECTOR detects beats in a photoplethysmogram (PPG) signal
    using the 'Multi-Scale Peak and Trough Detection' beat detector

    # Inputs

    * sig : a vector of PPG values
    * fs  : the sampling frequency of the PPG in Hz

    # Outputs
    * peaks : indices of detected pulse peaks
    * onsets : indices of detected pulse troughs (i.e. onsets)

    # Reference
    S. M. Bishop and A. Ercole, 'Multi-scale peak and trough detection optimised for periodic and quasi-periodic neuroscience data,' in Intracranial Pressure and Neuromonitoring XVI. Acta Neurochirurgica Supplement, T. Heldt, Ed. Springer, 2018, vol. 126, pp. 189-195. <https://doi.org/10.1007/978-3-319-65798-1_39>

    # Author
    Peter H. Charlton

    # Documentation
    <https://ppg-beats.readthedocs.io/>

    # Version
    1.0
    Downloaded 2025-04-11 and converted by machine from Matlab to Python
    
    # License - MIT
        Copyright (c) 2022 Peter H. Charlton
        Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    """
    
    # Window signal
    # split into overlapping 6 s windows
    win_len = 6  # in secs
    overlap = 0.2  # proportion of overlap between consecutive windows
    no_samps_in_win = int(win_len * fs)
    if len(sig) <= no_samps_in_win:
        win_starts = [0]
        win_ends = [len(sig)]
    else:
        win_offset = round(no_samps_in_win * (1 - overlap))
        win_starts = list(range(0, len(sig) - no_samps_in_win + 1, win_offset))
        win_ends = [s + no_samps_in_win for s in win_starts]
        # Ensure that the windows include the entire signal duration
        if win_ends[-1] < len(sig):
            win_starts.append(len(sig) - no_samps_in_win)
            win_ends.append(len(sig))
    
    # Downsample signal
    # Set up downsampling if the sampling frequency is particularly high
    do_ds = False
    min_fs = 40
    min_fs = 500
    if do_ds:
        if fs > min_fs:
            ds_factor = fs // min_fs  # integer division factor
            ds_fs = fs / ds_factor
        else:
            do_ds = False

    # detect peaks and onsets in each window
    peaks = []
    onsets = []
    for win_no in range(len(win_starts)):
    
        # - extract this window's data
        win_sig = sig[win_starts[win_no]:win_ends[win_no]]
        
        # - downsample signal
        if do_ds:
            rel_sig = win_sig[::ds_factor]  # simple downsampling by taking every ds_factor-th sample
        else:
            rel_sig = win_sig
        
        # - detect peaks and onsets
        # (the function below returns lists of indices)
        p, t = detect_peaks_and_onsets_using_msptd(win_sig)
        
        # - resample peaks if downsampling was applied
        if do_ds:
            p = [val * ds_factor for val in p]
            t = [val * ds_factor for val in t]
        
        # - correct peak indices by finding highest point within tolerance either side of detected peaks
        tol = math.ceil(fs * 0.02)
        p_corrected = []
        for pk in p:
            curr_peak = pk
            tol_start = max(curr_peak - tol, 0)
            tol_end = min(curr_peak + tol + 1, len(rel_sig))
            if tol_end <= tol_start:
                p_corrected.append(curr_peak)
            else:
                # find the index of the maximum point in rel_sig[tol_start:tol_end]
                local_max_index = np.argmax(rel_sig[tol_start:tol_end])
                # adjust current peak index (preserving the original structure)
                corrected_peak = curr_peak - tol + local_max_index
                p_corrected.append(corrected_peak)
        p = p_corrected
        
        # - correct onset indices by finding lowest point within tolerance either side of detected onsets
        t_corrected = []
        for onset in t:
            curr_onset = onset
            tol_start = max(curr_onset - tol, 0)
            tol_end = min(curr_onset + tol + 1, len(rel_sig))
            if tol_end <= tol_start:
                t_corrected.append(curr_onset)
            else:
                local_min_index = np.argmin(rel_sig[tol_start:tol_end])
                corrected_onset = curr_onset - tol + local_min_index
                t_corrected.append(corrected_onset)
        t = t_corrected
        
        # - store peaks and onsets (adjust indices by adding the window start)
        win_peaks = [val + win_starts[win_no] for val in p]
        peaks.extend(win_peaks)
        win_onsets = [val + win_starts[win_no] for val in t]
        onsets.extend(win_onsets)
    
    # tidy up detected peaks and onsets (ordering and retaining unique indices)
    peaks = np.unique(peaks)
    onsets = np.unique(onsets)
    
    # correct peak and onset indices
    # (i) to account for downsampling
    if do_ds:
        # - peaks
        peaks = peaks * ds_factor
        corrected_peaks = []
        for pk in peaks:
            tol_start = max(pk - ds_factor, 0)
            tol_end = min(pk + ds_factor + 1, len(sig))
            local_max_index = np.argmax(sig[tol_start:tol_end])
            corrected_peaks.append(pk - ds_factor + local_max_index)
        peaks = np.array(corrected_peaks)
        
        # - onsets
        onsets = onsets * ds_factor
        corrected_onsets = []
        for onset in onsets:
            tol_start = max(onset - ds_factor, 0)
            tol_end = min(onset + ds_factor + 1, len(sig))
            local_min_index = np.argmin(sig[tol_start:tol_end])
            corrected_onsets.append(onset - ds_factor + local_min_index)
        onsets = np.array(corrected_onsets)
    
    # (ii) tidy up detected peaks and onsets (by ordering them and only retaining unique ones)
    peaks  = np.unique(np.array(peaks))
    onsets = np.unique(np.array(onsets))
    
    # check (optional plotting of detected peaks and onsets)
    do_check = False
    if do_check:
        import matplotlib.pyplot as plt
        plt.plot(sig)
        plt.plot(peaks, sig[peaks], '*r')
        plt.plot(onsets, sig[onsets], '*k')
        plt.show()
    
    return peaks, onsets


def detect_peaks_and_onsets_using_msptd(x):
    """
    This function detects peaks and onsets using a multi-scale peak detection approach.
    
    Input:
      x : a signal vector
      
    Outputs:
      p : indices of detected peaks
      t : indices of detected onsets
    """
    N = len(x)  # length of signal
    L = math.ceil(N / 2) - 1  # maximum window length
    
    # Step 1: calculate local maxima and local minima scalograms
    # - detrend: removes the best-fit straight line
    x_detrended = detrend(x)
    
    # - initialise LMS matrices
    m_max = np.zeros((L, N), dtype=bool)
    m_min = np.zeros((L, N), dtype=bool)
    
    # - populate LMS matrices
    # Note: MATLAB indices start at 1 so adjustments are made for Python (0-indexed)
    for k in range(1, L + 1):
        # In MATLAB: for i = (k+2):(N-k+1) then use x(i-1)
        # Convert to Python indices: j = i - 1 runs from k+1 to N - k - 1 (inclusive)
        for j in range(k + 1, N - k):
            if x_detrended[j] > x_detrended[j - k] and x_detrended[j] > x_detrended[j + k]:
                m_max[k - 1, j] = True
            if x_detrended[j] < x_detrended[j - k] and x_detrended[j] < x_detrended[j + k]:
                m_min[k - 1, j] = True
    
    # Step 2: find the scale with the most local maxima (or local minima)
    gamma_max = np.sum(m_max, axis=1)  # row-wise summation
    gamma_min = np.sum(m_min, axis=1)
    lambda_max = np.argmax(gamma_max) + 1  # add one to convert back to MATLAB-style index
    lambda_min = np.argmax(gamma_min) + 1
    
    # Step 3: use lambda to remove all elements for which k > lambda
    m_max = m_max[:lambda_max, :]
    m_min = m_min[:lambda_min, :]
    
    # Step 4: Find peaks
    # - column-wise summation on the logical negation
    m_max_sum = np.sum(~m_max, axis=0)
    m_min_sum = np.sum(~m_min, axis=0)
    # In MATLAB: find(~m_max_sum) finds indices where m_max_sum == 0
    p = np.where(m_max_sum == 0)[0]
    t = np.where(m_min_sum == 0)[0]
    
    return list(p), list(t)


if __name__ == '__main__':
    # Example usage:
    # For demonstration, a dummy sine wave is created; replace this with your actual PPG signal.
    import matplotlib.pyplot as plt
    
    fs = 100  # sampling frequency in Hz
    t_vals = np.linspace(0, 10, int(10 * fs))
    # a sine wave at 1 Hz
    sig = np.sin(2 * np.pi * 1 * t_vals) + np.sin(2 * np.pi * 0.2 * t_vals) * 2.9 + np.cos(2 * np.pi * 0.05 * t_vals) * 5
    
    peaks, onsets = msptd_beat_detector(sig, fs)
    print("Peaks indices:", peaks)
    print("Onsets indices:", onsets)
    
    # Optional: visualize the signal with detected peaks and onsets
    plt.figure()
    plt.plot(sig, label='Signal')
    plt.plot(peaks, sig[peaks], '*r', label='Peaks')
    plt.plot(onsets, sig[onsets], '*k', label='Onsets')
    plt.legend()
    plt.title("Detected Peaks and Onsets")
    plt.show()
