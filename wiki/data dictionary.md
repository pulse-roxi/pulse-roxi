The result of `rr_est.estimate_rr_dataset()` creates a `.csv` file containing the following columns:

| variable | description |
| --- | --- |
|'trial-frame'| Trial number and frame number corresponding to this row. |
|'index'| Same as 'frame index' |
|'frame index'| Index of the time window within a trial. E.g., If a window starts at time 0, it would have index 0. It a window starts at time 10s, and the window increment was 5s, then the index is 2 |
|'time start'| Start of the window, in seconds |
|'time end'| End of the window, in seconds |
|'avg rr ref'| Average of the refernce RR signals, as specified in `config.reference_rr_signals` |
|'RR ref (median)'| Median breath per minute over the time window, based on `config.reference_rr_target` |
|'RR capno count'| Breath per minute based on counting the number of breaths observed in the capnography trace |
|'RR capno p25'| The 25 percentile of the instantaneous RR, calculated with capnography |
|'RR capno p75'| The 75 percentile of the instantaneous RR, calculated with capnography |
|'RR ref disagreement (bpm)'| The range (max-min) of `config.reference_rr_signals`. Literature often discards datapoints if the reference RR disagreement is large |
|'HR ref (median)'| Reference heart rate, in beats per minute. |
|'no rr ref'| Whether reference RR is missing in this window. This happens if, for example, the capnography trace is flat, or is Nan. |
|'aliased'| Whether the heart rate is less than 2x the breathing rate. If true, we expect accurate RR estimation to be impossible. These datapoints are excluded during analyses. |
|'no ppg'| Whether the PPG signal is missing. Excluded during analyses. |
|'clipping'| Whether the PPG signal meets or exceeds `config.ppg_max` or `config.ppg_min` |
|'RR ref co2'| RR reference calculated using capnography |
|'co2 std iqr'| The interquartile range of the standard deviation of the capnography trace. Only calculated if capnography is available. If this value is high, the breathing depth may not be stable. |
|'co2 mean iqr'| The interquartile range of the mean of the capnography trace. Only calculated if capnography is available. If this value is high, the capnography trace may not be stable.|
|'co2 std mean'| Standard deviation of the mean of the capnography trace. Only calculated if capnography is available. If this value is high, the capnography trace may not be stable.
|'co2 rr std/median'| Standard deviation of the respiratory rate normalized by the median respiratory rate. Measures breathing rate consistency. |
|'HR estimated'| Estimated Heart rate in beats per minute. |
|'HR est reliable'| Whether `heart_rate.estimate_avg_heart_rate()` was able to estimate a heart rate. If False, the estimated HR is based on `heart_rate.estimate_instantaneous_heart_rate_with_resample()` |
|'PPG quality'| Quality metric of the PPG waveform for this frame. Calculated using `signal_quality.template_match_similarity()`. The 'pct diagnostic quality pulses' in the dictionary is used as the PPG quality. |
|'trial'| Index of the trial number within the dataset |
|'subject id'| Subject ID corresponds to the file name |
|'subject age'| Subject age, if available |
|'notes'| In kapiolani dataset, the metadata includes notes gathered by the research technician. |
|'PSD, closest to prev RR'| The aggregated RR estimate using PSD analysis. The PSD peak closest to the RR seed is selected |
|'PSD median'| The aggregated RR estimate based on the median of the PSD RR estimates. These RR estimates have gone through harmonic analysis as well. |
|'Counting, median # peaks'| Aggregated RR based on the median number of peaks detected in the RIV waveforms. See more in `riv_analyses.peak_counting_analysis()` |
|'Counting, median pk delta rqi cutoff'| Aggregated RR based on the median of the RR estimates based on the peak-to-peak timing differences in the RIV waveforms. Only RIVs with some minimum respiratory quality index (RQI) are included. |
|'Counting, median pk delta std cutoff'| Aggregated RR based on the median of the RR estimates based on the peak-to-peak timing differences in the RIV waveforms. Only RIVs with peak-to-peak standard deviations below some threshold are included. See `rr_est.merge_counting_candidates()` for more. |
|'kalman'| RR estimate from `riv_analyses.kalman_analysis()` |
|'mean of fused candidates'| Mean aggregate of the `fusion_candidates` defined in `rr_est.estimate_rr_single_frame()` |
|'median of fused candidates'| Median aggregate of the `fusion_candidates` defined in `rr_est.estimate_rr_single_frame()` |
|'mode of fused candidates'| Mode aggregate of the `fusion_candidates` defined in `rr_est.estimate_rr_single_frame()` |
|'simple median'| Median aggregate of all RR individual candidates |
|'buffered_display'| Buffers `mean of fused candidates` over time. Takes a weighted average based on PPG quality. See `rr_est.rr_display_buffer()` for more. |
|'candidate - # riv peaks-{RIxV}'| RR estimate candidate from `riv_analyses.peak_counting_analysis()` based on the number of peaks detected in the RIV waveform. |
|'candidate - riv peak median delta-{RIxV}'| RR estimate candidate from `riv_analyses.peak_counting_analysis()` based on the time-difference between peaks detected in the RIV waveform. |
|'candidate - harmonic analysis-{RIxV}'| RR estimate candidate from `riv_analyses.psd_analysis()` based on PSD peak estimation followed by harmonic analysis. |
|'candidate - psd-{RIxV}'|  RR estimate candidate from `riv_analyses.psd_analysis()` based on PSD peak estimation and selecting the peak closest to the RR estiamte of the previous time frame. |
|'candidate - kalman-kalman'| RR estimate candidate from `riv_analyses.kalman_analysis()` |
|'quality - RIV kurtosis-{RIxV}'| Statsitical kurtosis of the RIV waveforms. Measures how "spread out" the waveform is along the y-axis. |
|'quality - RIV n detected peaks-{RIxV}'| Number of detected peaks in the RIV waveforms. See `riv_analyses.peak_counting_analysis()` |
|'quality - RIV peak-diff std-{RIxV}'| The standard deviation between the detected peaks in the RIV waveforms. See `riv_analyses.peak_counting_analysis()` |
|'quality - RIV skew-{RIxV}'| Statistical skew of the RIV waveforms |
|'quality - psd RQI-{RIxV}'| Respiratory quality index (RQI) based on the power spectra of the RIV waveforms. See `riv_analyses.psd_analysis()` |
|'quality - psd n valid peaks-{RIxV}'| Number of peaks in the respiratory rate range found in the RIV power spectra. See `riv_analyses.psd_analysis()`|
|'quality - fusion candidate quality-std'| Standard deviation across the RR fusion candidates as defined in `rr_est.estimate_rr_single_frame().` |
|'quality - heart rate-estimated'| Same as the Estimated heart rate from above. Can probably be removed |
|'quality - heart rate-reliable'| Same as the heart rate reliable from above. Can probably be removed |
|'quality - ppg stats-kurtosis'| Kurtosis of the PPG waveform |
|'quality - ppg stats-outliers in frame center'| Whether statistical outliers are found in the "center" of the PPG window. See `riv_est.detect_outliers()` and `riv_est.cut_riv_outliers()` |
|'quality - ppg stats-skew'| Skew of the PPG waveform |
|'quality - template matching-mean loc nonconforming pulses'| Relative positions of the outlier PPG pulses. See `signal_quality.template_match_similarity()` |
|'quality - template matching-pct amplitude outliers'| Percentage of PPG pulses with "outlier" amplitudes. see `signal_quality.template_match_similarity()` |
|'quality - template matching-pct diagnostic quality pulses'| Current preferred quality metric. Includes "pct good puls shapes" and "pct amplitude outliers". See `signal_quality.template_match_similarity()` |
|'quality - template matching-pct good pulse shapes'| Literature preferred quality metric. Percentage of PPG pulses highly correlated with the average PPG wave shape. See `signal_quality.template_match_similarity()` |
|'quality - template matching-pct poor pulse shapes'| 1 - "pct good pulse shapes". See  `signal_quality.template_match_similarity()` |