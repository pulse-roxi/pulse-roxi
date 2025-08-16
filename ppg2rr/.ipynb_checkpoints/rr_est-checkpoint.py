import textwrap
from collections import defaultdict
from datetime import datetime
from typing import Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kurtosis, skew

from ppg2rr import heart_rate, import_ppg, riv_est, util
from ppg2rr import signal_quality as sqi
from ppg2rr.riv_analyses import (
    kalman_analysis,
    peak_counting_analysis,
    psd_analysis,
)
from ppg2rr.config import AlgorithmParams

# ! Some function calls use this DEFAULT_PARAMS; which may ignore user desired params.
DEFAULT_PARAMS = AlgorithmParams(dataset='kapiolani')

# ! Potential issues:
# 1. Semi stable and accurate heart rate estimation is needed for accurate
# PPG pulse detection.
# 2. Algorithm requires a semi-accurate RR estimation as "seed".

def estimate_rr_single_frame(
    ppg: npt.NDArray[np.float16],
    fs_ppg: float,
    hr_est: Optional[float] = None,
    rr_est: Optional[float] = None,
    hr_min_bpm: float =  DEFAULT_PARAMS.hr_min_bpm,
    min_resp_freq: float = 0.13,
    lowpass_cutoff_ppg: Union[float,Literal["dynamic"]] = "dynamic",
    remove_riv_outliers: Literal["segment-wise", "point-wise"] = "segment-wise",
    fs_riv: float = DEFAULT_PARAMS.fs_riv,
    peak_counting_prominence: float = DEFAULT_PARAMS.peak_counting_prominence,
    n_sig: int = DEFAULT_PARAMS.n_kalman_fusion,
    min_deviation_tolerance: dict[str,float] = DEFAULT_PARAMS.min_deviation_tolerance,
    ppg_quality_corr_threshold: float = DEFAULT_PARAMS.ppg_quality_corr_threshold, # noqa E501
    show=False,
) -> Tuple[dict, list[dict], list[dict], dict]:
    """Estimates RR on a single time frame, typically 30 seconds.

    Args:
        ppg: Single frame of PPG signal from which RR is estimated
        fs_ppg: Sampling frequency of the PPG signal
        hr_est: Estimated heart rate in beats per minute. Assist in correctly 
            identifying each PPG pulse peaks, important for extracting the ppg features.
            If None (default), then this function calculates a heart rate estimation.
        rr_est: Estimated RR based on the previous frame. We would expect the current
            RR to be within 50% of the previous frame's RR. Default None.
        min_resp_freq: Minimum respiratory frequency in hertz, used in PSD analysis.
            Default is 0.13hz (7.2 bpm).
        remove_riv_outliers: Either "segment-wise" (default), or "point-wise".
            If "segment-wise", entire segments of the artifact-corrupted PPG signal
            are removed. If "point-wise", individual points containing the detected
            outliers are removed from the PPG signal.
        lowpass_cutoff_ppg [hz]: The cutoff frequency used to pre-process raw ppg.
            If "dynamic" (default), the cutoff frequency is 2.1*hr_est.
        fs_riv: Sampling frequency of RIV waveforms. Default is 12 hz, empirically 
            determined. Literature uses anywhere from 10 hz [1], 4 hz [2].
        n_sig: Number of RIV samples to use for Kalman Fusion. Default is 2.
        ppg_quality_corr_threshold: minimum correlation between each 
            individual pulse with the average pulse to be considered diagnostic quality.
        show: if True, displays plotly figures. Default false.


    Returns:
        rr_candidate_dict (dict): Merged RR estimates
        all_rr_candidates (list[dict]): All RR candidates, prior to merging.
            Dictionary contains "method", "feature", and "RR candidate".
            Where method is the method of derivation.
            Feature is the PPG feature from which the RR candidate is derived.
        quality_indices (list[dict]): of quality metrics
        hr: Dictionary with keys "estimated" and "reliable".

    References:
        [1] Khreis, S., Ge, D., Rahman, H. A., & Carrault, G. (2020). Breathing Rate
            Estimation Using Kalman Smoother with Electrocardiogram and
            Photoplethysmogram. IEEE Transactions on Biomedical Engineering, 67(3),
            893–904. https://doi.org/10.1109/TBME.2019.2923448
        [2] Karlen, W., Raman, S., Ansermino, J. M., & Dumont, G. A. (2013).
            Multiparameter respiratory rate estimation from the photoplethysmogram.
            IEEE Transactions on Biomedical Engineering, 60(7), 1946–1953.
            https://doi.org/10.1109/TBME.2013.2246160
    """
    # When device is off-finger, the signal is typically flat.
    flatline_ppg = (np.abs(ppg)<0.01).sum() > len(ppg)/2
    if flatline_ppg:
        remove_riv_outliers = None
        
    # =============== heart rate calculation ===============
    reliable_hr = False
    if hr_est is None:
        # estimate_avg_heart_rate returns np.nan if no reliable heart rate is found,
        # indicating noisy signal.
        hr_est = heart_rate.estimate_avg_heart_rate(
            ppg = ppg,
            sampling_rate = fs_ppg,
            min_heart_rate_hz = hr_min_bpm/60,
            show=show,
        )
        hr_est_inst, _ = heart_rate.estimate_instantaneous_heart_rate_with_resample(
            ppg = ppg,
            fs = fs_ppg,
            smooth = False,
            hr_min = 60,
        )
        if hr_est is np.nan and len(hr_est_inst)==0:
            np.savetxt('heart rate failed example.csv',ppg,delimiter=",")
            raise ValueError("heart rate is np.nan")
        
        if hr_est is not np.nan:
            reliable_hr = True
        else:
            hr_est = round(np.median(hr_est_inst),1)
            
    hr = {
        'estimate': hr_est,
        'reliable': reliable_hr,
    }
    
    
    # ================= HR-dependent parameters =================
    # assume heart rate is at least 1.75x resp rate. Any ratio higher risks aliasing,
    # and we won't be able to correctly determine the heart rate anyway.
    rr_max = hr_est / 1.75
    
    # dynamically calculate low pass filter cutoff
    if type(lowpass_cutoff_ppg) is str:
        if lowpass_cutoff_ppg == "dynamic":
            lowpass_cutoff = round(hr_est/60*2.1,2)
        else:
            raise KeyError("undefined lowpass_cutoff_ppg")
    else:
        lowpass_cutoff = lowpass_cutoff_ppg
        
    # pre-process ppg
    ppg = util.standardize(ppg)
    ppg_filt = util.lowpass_butter(
        ppg, signal_fs=fs_ppg, f_max=lowpass_cutoff, show=show
        )

    # ===================== RIV extraction =================
    # Extract respiratory induced variations from PPG signal
    # This function adds ppg as a candidate, which is used to aid RR
    # candidate extraction later.
    rivs, ppg_feat = riv_est.extract_rivs(
        ppg=ppg_filt,
        fs_riv=fs_riv,
        fs=fs_ppg,
        hr=hr_est,
        rr_max=rr_max,
        remove_outliers=remove_riv_outliers,
        min_deviation_tolerance=min_deviation_tolerance,
        show=show,
    )

    # ================= PPG quality ===================
    # The indices of PPG pulse peaks are found during RIV extraction step
    
    ppg_quality, _ = sqi.template_match_similarity(
        ppg_filt, 
        peak_locs = ppg_feat["RIIV_upper"]["x"], 
        correlation_threshold = ppg_quality_corr_threshold,
        show=show
    )

    if rivs["RIIV_upper"]["artifact in frame center"]:
        artifact_in_frame_center = True
    else:
        artifact_in_frame_center = False

    # ================= RR Candidate Extraction =================
    # time domain method - peak detection
    rivs_counting = peak_counting_analysis(
        rivs, 
        fs=fs_riv,
        peak_prominence=peak_counting_prominence,
        show=show)
    # del rivs_counting["ppg"]

    # spectral methods - power spectral density
    rivs_psd = psd_analysis(
        rivs, 
        fs_riv=fs_riv, 
        rr_seed=rr_est, 
        resp_range_min_f=min_resp_freq,
        min_peak_height=0.1, 
        show=show
    )
    # del rivs_psd["ppg"]  # remove ppg from list of candidates

    #  ============ sensor fusion based on power spectral density results  ============
    # seed with merged PSD candidates
    # kalman_seed = np.median(nearest_psd_RR_candidates)
    # print(kalman_seed)
    kalman_seed = rr_est
    rr_kalman, rqi_kalman = kalman_analysis(
        rivs_psd=rivs_psd,
        fs_riv=fs_riv,
        n_sig=n_sig,
        rr_seed=kalman_seed,
        fmin=min_resp_freq,
        show=show,
    )
    # ================= Pack RR Candidates =================
    # Build list of dictionarys containing RR candidates and associated info.
    # fmt: off
    all_rr_candidates = []
    all_rr_candidates.extend([{
                "method": "# riv peaks",
                "feature": key,
                "RR candidate": np.round(rivs_counting[key]["pk_count RR num pks"], 2),
            } for key in rivs_counting.keys()])
    all_rr_candidates.extend([{
                "method": "riv peak median delta",
                "feature": key,
                "RR candidate": np.round(rivs_counting[key]["pk_count RR median pk diff"], 2),
            } for key in rivs_counting.keys()])
    # all_rr_candidates.extend([{
    #             "method": "autocorr",
    #             "feature": key,
    #             "RR candidate": np.round(rivs_autocorr[key]["acf RR candidate"], 2),
    #         } for key in rivs_autocorr.keys()])
    all_rr_candidates.extend([{
                "method": "psd",
                "feature": key,
                "RR candidate": np.round(rivs_psd[key]["nearest psd RR candidate"], 2),
            } for key in rivs_psd.keys()])
    all_rr_candidates.extend([{
                "method": "harmonic analysis",
                "feature": key,
                "RR candidate": np.round(rivs_psd[key]["psd RR candidate"], 2),
            } for key in rivs_psd.keys()])
    all_rr_candidates.extend([{
        "method": "kalman",
        "feature": "kalman",
        "RR candidate": rr_kalman,
    }])

    # ================= Pack Quality Indices =================
    # There's not necessarily a one-to-one correspondence between between RR candidate
    # and quality index. i.e., there may be more or less than one quality index per RR
    # candidate.
    quality_indices = []
    quality_indices.extend([{
            "method": "psd RQI",
            "feature": key,
            "value": rivs_psd[key]["PSD RQI"],
        } for key in rivs_psd.keys()])
    quality_indices.extend([{
            "method": "psd RQI",
            "feature": "kalman fused",
            "value": rqi_kalman,
        }])
    quality_indices.extend([{
            "method": "psd n valid peaks",
            "feature": key,
            "value": rivs_psd[key]["n valid peaks"],
        } for key in rivs_psd.keys()])
    quality_indices.extend([{
            "method": "RIV peak-diff std",
            "feature": key,
            "value": rivs_counting[key]["std"],
        } for key in rivs_counting.keys()])
    quality_indices.extend([{
            "method": "RIV n detected peaks",
            "feature": key,
            "value": rivs_counting[key]["n detected peaks"],
        } for key in rivs_counting.keys()])
    quality_indices.extend([{
            "method": "RIV skew",
            "feature": key,
            "value": rivs[key]["skew"],
        } for key in rivs_counting.keys()])
    quality_indices.extend([{
            "method": "RIV kurtosis",
            "feature": key,
            "value": rivs[key]["kurtosis"],
        } for key in rivs_counting.keys()])
    quality_indices.extend([{
            "method": "template matching",
            "feature": key,
            "value": value,
        } for key, value in ppg_quality.items()])
    # quality_indices.extend([{
    #             "method": "pct large stat-delta",
    #             "feature": key,
    #             "value": value,
    #         } for key, value in pct_stat_over_thresh.items()])
    quality_indices.extend([{
            "method": "ppg stats",
            "feature": "outliers in frame center",
            "value": artifact_in_frame_center,
        }])
    quality_indices.extend([{
        "method": "ppg stats",
        "feature": "skew",
        "value": skew(ppg_filt),
    }])
    quality_indices.extend([{
        "method": "ppg stats",
        "feature": "kurtosis",
        "value": kurtosis(ppg_filt),
    }])
    # fmt: on

    #  ================= RR Candidate Fusion  =================

    # drop ML in here. Candidates & quality -> ML -> single most likely candidate
    # [print(key, rivs_psd[key]["psd RR candidate"]) for key in rivs_psd.keys()]
    psd_rqi_thresh = 0.04 # peaks below this threshold are ignored
    rr_candidate_dict = {}
    rr_candidate_dict["PSD, closest to prev RR"] = np.median(
        [
            np.round(rivs_psd[key]["nearest psd RR candidate"], 2)
            for key in rivs_psd.keys()
        ]
    )
    rr_candidate_dict["PSD median"] = np.median(
        [np.round(rivs_psd[key]["psd RR candidate"], 2) for key in rivs_psd.keys()]
    )
    rr_candidate_dict["Counting, median # peaks"] = np.median(
        [
            np.round(rivs_counting[key]["pk_count RR num pks"], 2)
            for key in rivs_counting.keys() if rivs_psd[key]["PSD RQI"] > psd_rqi_thresh
        ]
    )
    rr_candidate_dict["Counting, median pk delta rqi cutoff"] = np.median(
        [
            np.round(rivs_counting[key]["pk_count RR median pk diff"], 2)
            for key in rivs_counting.keys() if rivs_psd[key]["PSD RQI"] > psd_rqi_thresh
        ]
    )
    rr_candidate_dict["Counting, median pk delta std cutoff"] = merge_counting_candidates(
        rivs_counting, rivs_psd, peak_to_peak_std_cutoff=13, psd_rqi_thresh=0.04
        )

    rr_candidate_dict["kalman"] = rr_kalman
    
    fusion_candidates = [
        "PSD, closest to prev RR",
        "PSD median",
        "Counting, median pk delta std cutoff",
        "kalman",
    ]
    fusion_candidate_rrs = [rr_candidate_dict[candidate] for candidate in fusion_candidates]
    rr_candidate_dict["mean of fused candidates"] = np.nanmean(
        fusion_candidate_rrs
        )
    rr_candidate_dict["median of fused candidates"] = np.nanmedian(
        fusion_candidate_rrs
        )
    rr_candidate_dict["mode of fused candidates"] = util.mode_within_tolerance(
        fusion_candidate_rrs,
        tol=5
        )
    rr_candidate_dict["simple median"] = np.nanmedian(
        [d["RR candidate"] for d in all_rr_candidates]
    )
    
    # "Smart Fusion" used by Karlen et al. drops RR estimations if standard deviation
    # across candidates is greater than 4 breaths per minute.
    quality_indices.extend([{
        "method": "fusion candidate quality",
        "feature": "std",
        "value": np.std(fusion_candidate_rrs),
    }])
    quality_indices.extend([{
        "method": "heart rate",
        "feature": "estimated",
        "value": hr_est,
    }])
    quality_indices.extend([{
        "method": "heart rate",
        "feature": "reliable",
        "value": reliable_hr,
    }])
    return rr_candidate_dict, all_rr_candidates, quality_indices, hr

def merge_counting_candidates(
    rivs_counting, 
    rivs_psd,
    peak_to_peak_std_cutoff: float = 10,
    psd_rqi_thresh: float = 0.04,
    ) -> float:
    """
    Strategically choosing the RIV candidates to merge by using their associated quality
    metrics can improve the overall RR estimation. The RR candidates based on 
    peak-counting analysis showed the most accuracy improvement when we use the 
    peak-to-peak standard deviation as the quality metric. 
    
    It is possible that a noisy signal has no RR candidates satisfying that quality 
    criteria. Therefore, as a backup, we select RR candidates using the PSD_rqi 
    threshold.
    
    If no RR candidate has a high enough corresponding quality metric, then 
    nan is returned.
    
    Args:
        rivs: nested dict with each RIV as the key
        quality_indices: The "RIV peak-diff std" for each RIV
        peak_to_peak_std_cutoff: The quality metric we are using is "RIV peak-diff std".
            Candidates with corresponding std > peak_to_peak_std_cutoff are not used.
            Defaults to 10, empirically determined.
        psd_rqi_thresh: Threshold used to select candidates based on psd RQI.
            Defaults to 0.04.
            
    Returns:
        float: merged RR candidate
    """
    # Take candidates whose peak-to-peak std is < cutoff
    rr_candidates_std_cutoff = [
        np.round(rivs_counting[key]["pk_count RR median pk diff"], 2)
        for key in rivs_counting.keys() if rivs_counting[key]["std"] < peak_to_peak_std_cutoff
    ]
    
    # weights = [
    #     (peak_to_peak_std_cutoff - rivs_counting[key]["std"])
    #     for key in rivs_counting.keys() if rivs_counting[key]["std"] < peak_to_peak_std_cutoff
    # ]
    
    if len(rr_candidates_std_cutoff) > 0:
        return np.median(rr_candidates_std_cutoff)
    
    # If no candidates satisfy the above requirement, 
    # take candidates whose RQI is > threshold
    rr_candidates_rqi_cutoff = [
        np.round(rivs_counting[key]["pk_count RR median pk diff"], 2)
        for key in rivs_counting.keys() if rivs_psd[key]["PSD RQI"] > psd_rqi_thresh
    ]
    
    if len(rr_candidates_rqi_cutoff) > 0:
        return np.median(rr_candidates_rqi_cutoff)
    
    # If all candidates fail quality metric...
    return np.nan

def estimate_rr(
    trial_num: int,
    frame_num: Optional[list[int]] = None,
    dataset: str = "capnobase",
    params: Optional[AlgorithmParams] = None,
    show: bool = False,
    save_fig: bool = False,
    fig_suffix: str = "",
    rr_seed: Optional[float] = None,
):
    """Estimates the RR from a single trial of a dataset.

    Each trial of a dataset contains at least three minutes of data.
    The PPG signal and reference signals are broken down into windows 30 seconds
    (specified in params.window_size), and each following windows are incremented
    by params.window_increment. This function then calls estimate_rr_single_frame().

    Args:
        trial_num (int): Trial number to process, independent from subject ID.
            Trial numbers range from 0 to length(subjects).
        frame_num (list[int], optional): If None (default), processes all frames in
            the given ppg. If a list of integer is given, process those frames.
        dataset (str, optional): Defaults to "capnobase".
        params (Optional[AlgorithmParams], optional): Defaults to None.
        show (bool): If frame_num is None, show results for each trial.
            Otherwise, show detailed information about the frames processed.
            Defaults to False.
        save_fig (bool, optional): Saves figure to data/ folder. Defaults to False.
        fig_suffix (str): Suffix to append to figure outputs.
        rr_seed (float): A guess of the RR to seed the PSD and Kalman analysis, in
            which the RR candidate is selected within a range based on this seed.

    Returns:
        frame_data: list of dictionary with frame time, reference, and
            quality metrics for each frame.
        rr_candidate_merged_list: list of dictionary containing merged
            candidates for each frame.
        all_rr_candidates_list: list of dictionary of all raw rr candidates
            for each frame.
        feature_quality_list: list of dictionary containing quality metrics
            for each frame.
        meta: dictionary of metadata for the signals used.
    """
    # constant parameters
    if params is None:
        params = AlgorithmParams(dataset=dataset)

    # logic for when to display figures
    if frame_num is not None:
        # user specify single frame(s)
        if show is False:
            show_frame = False
        elif show is True:
            show = False
            show_frame = True
    else:
        # computing over all frames
        show_frame = False

    # load raw PPG data
    # we expect ref_dict to contain keys 'hr' and 'rr' with values 'x' and 'y'.
    ppg_raw, fs_ppg, ref_dict, meta = import_ppg.load_ppg_data(
        dataset=dataset,
        trial=trial_num,
        window_size=params.window_size,
        probe_num=params.probe,
        read_cpo=params.cpo,
        show=show_frame,
    )

    # exit function if no data exists for the current trial
    if ppg_raw is None:
        return None, None, None, None, None

    # for datasets where RR from capnography is available
    co2_stats_df = None
    if dataset.lower() in ["kapiolani","capnobase"]:
        # TODO: calculate short-time window stats here
        fs_co2 = ref_dict["co2"]["fs"]
        window_length = int(10 * fs_co2)
        n_overlap = int(5 * fs_co2)
        co2_short_time_stats = sqi.moving_window_stats(
            signal=ref_dict["co2"]["y"],
            window_length=window_length,
            n_overlap=n_overlap,
            fs=fs_co2,
            standardize=False,
            difference=False,
            show=False,
        )
        co2_stats_df = pd.DataFrame.from_dict(co2_short_time_stats)
    if dataset.lower() == "mimic":
        ref_dict["hr"] = ref_dict["hr"]

    # Prep frame-wise estimation of RR
    win_increment = params.window_increment
    win_size = params.window_size
    num_windows = int(((len(ppg_raw) / fs_ppg) - win_size) // win_increment) + 2

    if frame_num is not None:
        if np.any(np.array(frame_num) > num_windows):
            raise ValueError("Frame number exceeds the number of possible frames.")

    rr_candidate_merged_list, all_rr_candidates_list, feature_quality_list = [], [], []
    frame_data: list[dict] = []
    frame_ref = {
        key: {"x": [], "y": [], "frame center": [], "y agg": [], "agg label": None}
        for key in ref_dict.keys()
    }

    t_list = frame_num if frame_num is not None else range(num_windows)
    rr_buffer = []
    for t_idx in t_list:
        if show:
            print("frame:", t_idx)
        this_frame = defaultdict(dict)
        # =========== define current frame window ==========
        # the logic for t_end causes the last two windows to be shorter than the rest
        # because the last three windows share the same t_end. To get around this,
        # we use t_start + win_size//2 as x-ticks when making figures
        # This may need to adjusted by adjusting the num_windows calculation.

        t_start = t_idx * win_increment
        t_end = min(t_start + win_size, len(ppg_raw) // fs_ppg)

        this_frame["frame index"] = t_idx
        this_frame["time start"] = t_start
        this_frame["time end"] = t_end

        # every ppg segment is normalized to [-1, 1]. PPG signal y-value affects the
        # prominance parameter during peak selection and y-value of the power spectra.
        ppg_raw_win = ppg_raw[int(t_start * fs_ppg) : int(t_end * fs_ppg)]
        ppg_win = util.normalize(ppg_raw_win, range=(-1,1))
        if len(t_list) == 1:
            # for debugging purposes
            this_frame["ppg"] = ppg_raw_win
        # ============ get current windows for each reference signal ============
        # heart rate, breathing rate, capnography level. Signal may be None.
        for key in ref_dict.keys():
            this_frame_ref_x = np.nan
            this_frame_ref_y = np.nan
            this_frame_ref_aggregated = np.nan
            agg_label = ''
            
            ref_x = ref_dict[key]["x"]
            if ref_x is not None:
                ref_win_idx = (ref_x >= t_start) & (ref_x < t_end)
                # duration of the reference window based on index
                if sum(ref_x[ref_win_idx]) > 2:
                    ref_win_dur = ref_x[ref_win_idx][-1] - ref_x[ref_win_idx][0]
                else:
                    ref_win_dur = 0
                    
                # it's possible for reference signal to be missing during the recording.
                if ref_win_dur > win_size/2:
                    this_frame_ref_x = ref_dict[key]["x"][ref_win_idx]
                    this_frame_ref_y = ref_dict[key]["y"][ref_win_idx]
                    if key == "co2 breath counts":
                        this_frame_ref_aggregated = this_frame_ref_y[-1]
                        agg_label = "final count"
                    else:
                        this_frame_ref_aggregated = np.nanmedian(this_frame_ref_y)
                        agg_label = "median"
                        
            frame_ref[key]["agg label"] = agg_label
            frame_ref[key]["x"].append(this_frame_ref_x)
            frame_ref[key]["y"].append(this_frame_ref_y)
            frame_ref[key]["y agg"].append(this_frame_ref_aggregated)
            frame_ref[key]["frame center"].append(t_start + win_size // 2)

        # Make refernce RR be the average of specified reference signals
        # we can drop RRs if reference signals differ by some threshold, per
        # Pimentel, et al (2017). Toward a robust estimation of respiratory rate from
        # pulse oximeters. IEEE Transactions on Biomedical Engineering, 64(8),
        # 1914–1923. https://doi.org/10.1109/TBME.2016.2613124
        this_frame_rrs = []
        for signal in params.reference_rr_signals:
            this_frame_rrs.append(frame_ref[signal]["y agg"][-1])
            
        if np.all(np.isnan(this_frame_rrs)):
            current_rr_ref = np.nan
            this_frame["RR ref (median)"] = np.nan
            this_frame["avg rr ref"] = np.nan
        else:
            avg_rr_ref = np.nanmean(this_frame_rrs)
            this_frame["avg rr ref"] = avg_rr_ref
            current_rr_ref = frame_ref[params.reference_rr_target]["y agg"][-1]
            this_frame["RR ref (median)"] = current_rr_ref
            
        if ("rr capno" in params.reference_rr_signals) & (current_rr_ref is not np.nan):
            this_frame["RR capno count"] = frame_ref["co2 breath counts"]["y"][-1][-1]
            this_frame["RR capno p25"] = np.percentile(frame_ref["rr capno"]["y"][-1],25)
            this_frame["RR capno p75"] = np.percentile(frame_ref["rr capno"]["y"][-1],75)

        current_rr_range = np.max(this_frame_rrs) - np.min(this_frame_rrs)
        this_frame["RR ref disagreement (bpm)"] = current_rr_range
        

        # # =============== reference heart rate ===============
        hr_ref = frame_ref["hr"]["y agg"][-1]
        this_frame["HR ref (median)"] = hr_ref

        # seed with known RR value for 1st frame.
        # TODO: replace this logic with a lookup table, based on age?
        current_rr_est = current_rr_ref if rr_seed is None else rr_seed

        # =========== Frame-wise quality checks ============
        # some quality measures are calculated in estimate_rr_single_frame() to
        # improve performance.
        this_frame["no rr ref"] = np.isnan(current_rr_ref)
        if this_frame["no rr ref"] or (current_rr_ref == 0):
            this_frame["aliased"] = False
        else:
            this_frame["aliased"] = hr_ref / current_rr_ref <= 2
        
        # NOTE: May need a separate behavior for flatline PPG
        flatline_ppg = (np.abs(ppg_win)<0.01).sum() > len(ppg_win)/2
        this_frame["no ppg"] = (len(ppg_win) == 0) | flatline_ppg        
        this_frame["clipping"] = np.any(ppg_raw_win >= params.ppg_max) or np.any(ppg_raw_win <= params.ppg_min)
        
        # breathing quality via capnography signal.
        if co2_stats_df is not None:
            this_frame['RR ref co2'] = frame_ref["rr capno"]["y agg"][-1]
            # capno_range = params["capno_max"] - params["capno_min"]
            capno_range = params.capno_max - params.capno_min
            
            fs_co2 = ref_dict["co2"]["fs"]
            co2_stats_t = co2_stats_df["time index"]
            ref_win_idx = (co2_stats_t >= t_start) & (co2_stats_t < t_end)
            this_frame_co2_stats = co2_stats_df[ref_win_idx]

            # consistent breathing depth: normalized IQR is stable over a timeframe
            this_frame["co2 std iqr"] = (
                np.std(this_frame_co2_stats["iqr"]) / capno_range
            )
            # deep breathing; mean short-time IQR > threshold
            this_frame["co2 mean iqr"] = (
                np.mean(this_frame_co2_stats["iqr"]) / capno_range
            )
            # mean capnography value is stable over a timeframe
            this_frame["co2 std mean"] = (
                np.std(this_frame_co2_stats["mean"] - params.capno_min) / capno_range
            )
            # consistent breathing rate
            capno_rr_std = np.std(frame_ref[params.reference_rr_target]["y"][-1])
            this_frame["co2 rr std/median"] = capno_rr_std / (current_rr_ref)

        # =========== process single frame ===========
        (
            rr_candidate_merged,
            all_rr_candidates,
            feature_quality,
            hr_estimated,
        ) = estimate_rr_single_frame(
            ppg=ppg_win,
            fs_ppg=fs_ppg,
            min_resp_freq=params.RR_min_bpm / 60,
            hr_min_bpm=params.hr_min_bpm,
            rr_est=current_rr_est,
            remove_riv_outliers=params.remove_riv_outliers,
            min_deviation_tolerance=params.min_deviation_tolerance,
            ppg_quality_corr_threshold=params.ppg_quality_corr_threshold,
            lowpass_cutoff_ppg=params.lowpass_cutoff_ppg,
            peak_counting_prominence=params.peak_counting_prominence,
            fs_riv=params.fs_riv,
            n_sig=params.n_kalman_fusion,
            show=show_frame,
        )

        # exclude median in candidate list
        temp = pd.DataFrame(feature_quality)
        this_frame_quality = temp.loc[temp['feature']=='pct diagnostic quality pulses','value'].iloc[0]
        
        # unpack heart rate information
        this_frame["HR estimated"] = hr_estimated['estimate']
        this_frame["HR est reliable"] = hr_estimated['reliable']
        this_frame["PPG quality"] = this_frame_quality
        # ================== Buffer for User Display  =====================
        rr_buffer, rr_display = rr_display_buffer(rr_buffer, rr_candidate_merged, this_frame_quality)
        rr_candidate_merged["buffered_display"] = rr_display

        # if ppg data exist, update rr_seed for the next frame
        # may need to be smarter about this one, maybe "if ppg data is high quality"
        # rr_seed = rr_candidate_merged["kalman"]
        if not (this_frame["no ppg"]):
            rr_seed = rr_display
            
        # ================== pack results to be returned ==================
        rr_candidate_merged_list.append(rr_candidate_merged)
        all_rr_candidates = [{**d, "frame index": t_idx} for d in all_rr_candidates]
        all_rr_candidates_list.extend(all_rr_candidates)
        feature_quality = [{**d, "frame index": t_idx} for d in feature_quality]
        feature_quality_list.extend(feature_quality)

        # add frame-wise information to list
        frame_data.append(this_frame)


    frame_data_df = pd.DataFrame(frame_data)
    assert len(rr_candidate_merged_list) == frame_data_df.shape[0]

    # ============================ Display Results ============================
    if show or save_fig:
        if dataset == "kapiolani":
            subtitle = f"P{params.probe}; {meta['age']} months; {meta['notes']}"
        elif dataset == "mimic":
            subtitle = ""
        else:
            subtitle = ""

        # prepare a dataframe of signal qualities to visualize
        qualities_df = pd.DataFrame(feature_quality_list)
        temp = qualities_df[
            (qualities_df["method"] == "pct large stat-delta")
            | (
                (qualities_df["method"] == "template matching")
                & (qualities_df["feature"] == "pct diagnostic quality pulses")
            )
        ]

        ppg_quality_df = temp.pivot(
            columns="feature", index="frame index", values="value"
        )
        if (dataset == "kapiolani") & ('RR capno p25' in frame_data_df.columns):
            ppg_quality_df = pd.concat(
                [
                    frame_data_df[
                        [
                            # "co2 std iqr",
                            # "co2 mean iqr",
                            # "co2 std mean",
                            "co2 rr std/median",
                        ]
                    ],
                    ppg_quality_df,
                ],
                axis=1,
            )
            ref_RR_shading_borders = frame_data_df[['RR capno p25','RR capno p75']]
        else:
            ref_RR_shading_borders = None

        ref_dict["avg rr ref"]["y"] = frame_data_df["avg rr ref"]
        ref_dict["avg rr ref"]["x"] = frame_data_df["time start"] + win_size // 2

        fig = individual_trial_fig(
            ppg_raw=ppg_raw,
            ppg_fs=fs_ppg,
            ppg_quality_df=ppg_quality_df,
            reference_rr_dict=ref_dict,
            reference_rr_keys=params.reference_rr_signals,
            gs_signal_raw_x=ref_dict[params.gold_standard_target]["x"],
            gs_signal_raw_y=ref_dict[params.gold_standard_target]["y"],
            gs_signal_label=params.gold_standard_label,
            frame_ref=frame_ref,
            rr_candidate_x=frame_data_df["time start"] + win_size // 2,
            rr_candidate_df=pd.DataFrame(rr_candidate_merged_list),
            title_str=(f"{dataset}: trial {trial_num}; filename: {meta['id']}"),
            subtitle=subtitle,
            reference_shading=ref_RR_shading_borders,
        )

    if show:
        fig.show()

    if save_fig:
        current_time = datetime.now().strftime("%Y%m%d %H%M%S")
        if dataset.lower() == "kapiolani":
            suffix = f"P{params.probe}_{meta['id']}_{fig_suffix}{current_time}"
        else:
            suffix = f"{meta['id']}_{current_time}"
        root = import_ppg.GIT_ROOT
        fig.write_image(f"{root}/data/results/{dataset}_{suffix}.png")

    return (
        frame_data,
        rr_candidate_merged_list,
        all_rr_candidates_list,
        feature_quality_list,
        meta,
    )

def rr_display_buffer(
    buffer: list[float], 
    rr_candidates: dict[str,float], 
    quality: float,
    buffer_len: int = 5,
    ) -> Tuple[list[float], float]:
    """
    Calculate the RR to display using a buffer. The buffered RR is a weighted 
    average between the buffer average and RR_est from our algorithm. 
    The weights depend on the "quality" of the signal and agreement across 
    RR candidates.
        
    Args:
        buffer: list of previous RR estimates
        rr_candidates: dictionary of rr candidates estimated via the algorithm
        quality: 1 - percent_nonconformant_pulses
        buffer_len: length of the buffer list. Default is 5.
        
    Returns
        updated buffer, rr_display
    """
    
    mean_rr_est = rr_candidates['mean of fused candidates']
    original_candidates = [v for v in rr_candidates.values()][:-2]
    
    if quality < 0.5:
        # waveform is low quality
        if len(buffer) > 0:
            rr_display = 0.5*np.nanmean(buffer) + 0.5*mean_rr_est
            buffer.append(rr_display)
        else:
            # ignore if bad frame & buffer is empty.
            rr_display = np.nan
    else:
        # waveform is high quality
        if len(buffer) == 0:
            rr_display = mean_rr_est
        elif np.std(original_candidates) <= 3:
            # high quality and high agreement among candidates
            rr_display = 0.1*np.nanmean(buffer) + 0.9*mean_rr_est
            # high quality but low agreement among candidates
        else:
            rr_display = 0.25*np.nanmean(buffer) + 0.75*mean_rr_est
        buffer.append(rr_display)
        
    if len(buffer) > buffer_len:
        # drop the first element of the buffer
        buffer = buffer[1:]
        
    return buffer, rr_display
    
def estimate_rr_dataset(
    dataset: str = "capnobase",
    trials: Optional[list[int]] = None,
    show: bool = False,
    params: Optional[AlgorithmParams] = None,
    save_fig: bool = False,
    save_df: bool = True,
    file_suffix: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calls estimate_rr() over an entire dataset & gather tabulated results.

    Args:
        dataset (literal): "capnobase" or "kapiolani".
        trials (list[int]): List of integers to run. Default None (all trials)
        params (dict): Algorithm parameters specified by the config dictionary.
        show (bool): If true, show individual trial results. Default False.
        save_fig (bool): If true, save figures for individual trial results.
            Default False.
        save_df (bool): If true, save results to csv file. Default True.
        file_suffix (str): string to append to the saved dataframe to note the
            experiment condition.

    Returns:
        - DataFrame summarizing the results of estimate_rr.
        - DataFrame summarizing all rr candidates
        - DataFrame summarizing all quality metrics
    """
    df = pd.DataFrame()
    all_rr_candidates = []
    all_quality_indices = []

    if params is None:
        params = AlgorithmParams(dataset=dataset)
    trials_to_eval = range(params.dataset_size) if trials is None else trials

    for trial in trials_to_eval:
        print("processing trial: ", trial)
        try:
            (
                frame_data,
                rr_candidate_merged,
                this_trial_rr_candidates,
                this_trial_feature_quality,
                meta,
            ) = estimate_rr(
                trial_num=trial,
                dataset=dataset,
                params=params,
                show=show,
                save_fig=save_fig,
                fig_suffix=file_suffix,
            )

            # skip current trial if estimate_rr returns none.
            if frame_data is None:
                continue
        except Exception as e:
            print(f"encountered error in trial {trial}")
            print(e)
            break

        # organize results summary into table; calculate error metric
        merged_candidate_df = pd.DataFrame(rr_candidate_merged)
        df_trial = pd.DataFrame(frame_data)
        df_trial["trial"] = np.ones(len(frame_data)) * trial
        df_trial["subject id"] = meta["id"]
        df_trial["subject age"] = meta["age"]
        df_trial = pd.concat([df_trial, merged_candidate_df], axis=1)
        df = pd.concat([df, df_trial], axis=0)

        # organize all rr candidates & quality
        this_trial_rr_candidates = [
            {**d, "trial": trial} for d in this_trial_rr_candidates
        ]
        all_rr_candidates.extend(this_trial_rr_candidates)
        this_trial_feature_quality = [
            {**d, "trial": trial} for d in this_trial_feature_quality
        ]
        all_quality_indices.extend(this_trial_feature_quality)

    if df.shape[0] > 0:
        df["trial"] = df["trial"].astype(int)
    df = df.reset_index()

    # save results
    if save_df:
        current_time = datetime.now().strftime("%Y%m%d %H%M%S")
        probe = f"P{params.probe}_" if dataset.lower == "kapiolani" else ""
        suffix = f"{probe}{file_suffix}_{current_time}"
        X = util.pivot_and_merge(
            references=df, 
            qualities=pd.DataFrame(all_quality_indices), 
            candidates=pd.DataFrame(all_rr_candidates)
            )
        X.to_csv(
            f"{import_ppg.GIT_ROOT}/data/results/{dataset}_{suffix}_features.csv",
        )

    return df, pd.DataFrame(all_rr_candidates), pd.DataFrame(all_quality_indices)


def individual_trial_fig(
    ppg_raw: npt.NDArray[np.float16],
    ppg_fs: float,
    ppg_quality_df: dict,
    reference_rr_dict: dict,
    reference_rr_keys: list[str],
    gs_signal_raw_x: npt.NDArray[np.float16],
    gs_signal_raw_y: npt.NDArray[np.float16],
    gs_signal_label: str,
    frame_ref: dict,
    rr_candidate_x: npt.NDArray[np.float16],
    rr_candidate_df: pd.DataFrame,
    title_str: str,
    subtitle: str,
    reference_shading = None,
) -> go.Figure:
    """Display PPG, RR estimates, and other info.

    Args:
        ppg_raw: Raw PPG signal
        ppg_fs: PPG sampling frequency, used to create time-axis for plotting.
        reference_rr_dict: Nested dictionary containing reference signals,
            with 'x' and 'y' values. e.g.,
            reference_rr_dict['co2']['x'], reference_rr_dict['co2']['y']
        reference_rr_keys: keys in reference_rr_dict to display; Each key contains a
            'x' and 'y' pair of values, corresponding to a reference respiratory rate.
        gs_signal_raw_x: Time-axis of the gold-standard signal
        gs_signal_raw_y: Gold-standard signal value.
        gs_signal_label: e.g., 'Capnography' or 'Monitor'
        frame_ref: nested dictionary containing frame-wise reference values.
        rr_candidate_x: time-axis of estimated RR
        rr_candidate_df: DataFrame containing RR candidates in each column.
            expects a column named "frame index" to indicate how they should be
            grouped.
        title_str: Figure title
        subtitle: Figure subtitle
    """
    # Check inputs
    if not (set(reference_rr_keys).issubset(reference_rr_dict.keys())):
        raise KeyError(
            f"{reference_rr_keys} are not a subset of the keys of reference_rr_dict"
        )

    # Display PPG, RR estimates, and other info
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.2, 0.1, 0.2, 0.5],  # relative heights of each row
    )

    t_axis = np.linspace(0, len(ppg_raw) / ppg_fs, len(ppg_raw))
    fig.add_scatter(x=t_axis, y=ppg_raw, mode="lines", name="Raw PPG", row=1, col=1)

    # CO2 Quality Metrics
    for col in ppg_quality_df.columns:
        fig.add_scatter(
            x=rr_candidate_x,
            y=ppg_quality_df[col],
            mode="lines",
            name=col,
            row=2,
            col=1,
        )

    # Gold Standard
    fig.add_trace(
        go.Scatter(
            x=gs_signal_raw_x,
            y=gs_signal_raw_y,
            name=gs_signal_label,
        ),
        row=3,
        col=1,
    )

    # ========================== plot references ==========================
    ref_colors = ["hsl(240, 17%, 77%)", "#b49c91", "#91b2b4"]
    ref_median_color = ["hsl(240, 2%, 38%)", "#424257", "#355253"]
    for idx, ref_key in enumerate(reference_rr_keys):
        fig.add_trace(
            go.Scatter(
                x=reference_rr_dict[ref_key]["x"],
                y=reference_rr_dict[ref_key]["y"],
                mode="lines",
                name=f"{ref_key}",
                line={"color": ref_colors[idx], "dash": "dash"},
            ),
            row=4,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=frame_ref[ref_key]["frame center"],
                y=frame_ref[ref_key]["y agg"],
                mode="lines",
                name=f"{ref_key} {frame_ref[ref_key]['agg label']}",
                line={"color": ref_median_color[idx], "dash": "solid"},
            ),
            row=4,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=reference_rr_dict["avg rr ref"]["x"],
            y=reference_rr_dict["avg rr ref"]["y"],
            mode="lines",
            name=f"avg of<br>{reference_rr_keys}",
            line={"color": "#2c2421", "dash": "solid", "width": 3},
        ),
        row=4,
        col=1,
    )

    # # shade +/- 2 breaths of the breath counts
    # if "co2 breath counts" in reference_rr_keys:
    #     fig.add_scatter(
    #         x=frame_ref["co2 breath counts"]["frame center"], 
    #         y=np.array(frame_ref["co2 breath counts"]["y agg"])-2,
    #         fill=None,
    #         mode='lines',
    #         line=dict(width=0.1),
    #         showlegend = False,
    #         row=4,
    #         col=1,
    #         )
    #     fig.add_scatter(
    #         x=frame_ref["co2 breath counts"]["frame center"], 
    #         y=np.array(frame_ref["co2 breath counts"]["y agg"])+2,
    #         fill='tonexty',
    #         fillcolor='rgba(145, 178, 180, 0.486)',
    #         mode='lines',
    #         line=dict(width=0.1),
    #         showlegend = False,
    #         row=4,
    #         col=1,
    #         )
    
    # the shaded area between 25th and 75th percentile
    if reference_shading is not None:
        fig.add_scatter(
            x=frame_ref[ref_key]["frame center"], 
            y=reference_shading["RR capno p25"],
            fill=None,
            mode='lines',
            line=dict(width=0.1),
            showlegend = False,
            row=4,
            col=1,
            )
        fig.add_scatter(
            x=frame_ref[ref_key]["frame center"], 
            y=reference_shading["RR capno p75"],
            fill='tonexty',
            fillcolor='rgba(145, 178, 180, 0.486)',
            mode='lines',
            line=dict(width=0.1),
            showlegend = False,
            row=4,
            col=1,
            )

    for col in rr_candidate_df.columns:
        fig.add_scatter(
            x=rr_candidate_x,
            y=rr_candidate_df[col],
            mode="lines",  # 'lines' or 'markers'
            name=f"rr candidate: <br>{col}",
            row=4,
            col=1,
        )

    fig.update_xaxes(title_text="Time (seconds)", row=4, col=1)
    fig.update_yaxes(title_text="Raw PPG", row=1, col=1)
    fig.update_yaxes(title_text=f"Raw {gs_signal_label}", row=3, col=1)
    fig.update_yaxes(title_text="Resp Rate (breaths per min)", row=4, col=1)

    sub_wrap = "<br>".join(textwrap.wrap(subtitle, width=100))
    fig.update_layout(
        height=2000,
        width=1600,
        yaxis2={"range": [0, 1]},
        title_text=f"{title_str}<br><span style='font-size:12px'>{sub_wrap}</span>",
        legend={
            "orientation": "h",
        },
    )

    return fig

    # TODO: Evaluate
    # compare with HeartPy?
    # https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/algorithmfunctioning.html # noqa E501
    # and RapidHRV https://peerj.com/articles/13147/
