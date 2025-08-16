import textwrap
from collections import defaultdict
from datetime import datetime
from typing import Literal, Optional, Tuple, Union
from statistics import median
import warnings
from time import sleep

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kurtosis, skew
import math

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
    max_resp_freq: float = 2.3,
    min_resp_freq: float = 0.13,
    lowpass_cutoff_ppg: Union[float,Literal["dynamic"]] = "dynamic",
    remove_riv_outliers: Literal["segment-wise", "point-wise"] = "segment-wise",
    fs_riv: float = DEFAULT_PARAMS.fs_riv,
    peak_counting_prominence: float = DEFAULT_PARAMS.peak_counting_prominence,
    n_sig: int = DEFAULT_PARAMS.n_kalman_fusion,
    min_deviation_tolerance: dict[str,float] = DEFAULT_PARAMS.min_deviation_tolerance,
    ppg_quality_corr_threshold: float = DEFAULT_PARAMS.ppg_quality_corr_threshold, # noqa E501
    params: Optional[AlgorithmParams] = DEFAULT_PARAMS,
    save_psd_fig_as: str = "",
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
        max_resp_freq, min_resp_freq: the upper and lower bound of respiratory
            range to search, in hertz. Defaults to 2.3 hz (138 breaths
            per minute) and 0.13 hz (7.8) breaths per minute. See addiional notes in riv_analyses.py.
        remove_riv_outliers: Either "segment-wise" (default), or "point-wise".
            If "segment-wise", entire segments of the artifact-corrupted PPG signal
            are removed. If "point-wise", individual points containing the detected
            outliers are removed from the PPG signal.
        lowpass_cutoff_ppg [hz]: The cutoff frequency used to pre-process raw ppg.
            If "dynamic" (default), the cutoff frequency is 2.1*hr_est.
        fs_riv: Sampling frequency of RIV waveforms. Default is 12 hz, empirically 
            determined. Literature uses anywhere from 10 hz [1], 4 hz [2].
        peak_counting_prominence: Passed to peak_counting_analysis() for RIV counting.
        n_sig: Number of RIV samples to use for Kalman Fusion. Default is 2.
        ppg_quality_corr_threshold: minimum correlation between each 
            individual pulse with the average pulse to be considered diagnostic quality.
        params: Container of various AlgorithmParams. This is admittedly redundant of other
            parameters that are members of AlgorithmParams. It was added late in the project 
            to support additional members.
        show: if True, displays plotly figures. Default false.


    Returns:
        rr_est_succeeded (bool): Whether estimation of RR succeeded or failed.
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
        
    # =============== Mains noise in PPG ===================
    ppg_max_noise_freq, ppg_Pxx_max_over_median, ppg_Pxx_mean = sqi.ppg_mains_noise(ppg, fs_ppg, show=show, save_fig_as=save_psd_fig_as)

    # =============== heart rate calculation ===============
    if hr_est is None:
        # estimate_avg_heart_rate returns np.nan if no reliable heart rate is found,
        # indicating noisy signal.
        hr_est = heart_rate.estimate_avg_heart_rate(
            ppg = ppg,
            sampling_rate = fs_ppg,
            min_heart_rate_hz = hr_min_bpm/60,
            show=show,
        )
        hr_est_y, hr_est_x = heart_rate.estimate_instantaneous_heart_rate_with_resample(
            ppg = ppg,
            fs = fs_ppg,
            smooth = False,
            hr_min = hr_min_bpm,
        )

        if hr_est is np.nan and len(hr_est_y)==0:
            # Heart rate cannot be determined, so we cannot proceed. 
            return None, None, None, None
        
        if hr_est is np.nan:
            reliable_hr = False
            hr_est = round(np.mean(hr_est_y),1)
        else:
            reliable_hr = True
    else:
        reliable_hr = True          # assume reliable, perhaps unwisely
            
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
            lowpass_cutoff = round(hr_est/60 * DEFAULT_PARAMS.lowpass_dynamic_scalar_ppg, 2)
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
    try:
        rivs, ppg_feat = riv_est.extract_rivs(
            ppg=ppg_filt,
            fs_riv=fs_riv,
            fs=fs_ppg,
            hr=hr_est,
            rr_max=rr_max,
            remove_outliers=remove_riv_outliers,
            outlier_tolerance=params.outlier_tolerance,
            min_deviation_tolerance=min_deviation_tolerance,
            min_keep_pct=params.min_keep_pct,
            show=show,
        )
        extract_rivs_succeeded = True
    except:
        # util.newprint("extract_rivs() failed.", on_a_new_line=True)
        extract_rivs_succeeded = False
        rivs = None
        ppg_feat = None

    if extract_rivs_succeeded and rivs["RIIV_upper"]["artifact in frame center"]:
        artifact_in_frame_center = True
    else:
        artifact_in_frame_center = False

    if extract_rivs_succeeded:
        # ================= PPG quality ===================
        # The indices of PPG pulse peaks are found during RIV extraction step
        
        ppg_quality, _ = sqi.template_match_similarity(
            ppg_filt, 
            peak_locs = ppg_feat["RIIV_upper"]["x"], 
            correlation_threshold = ppg_quality_corr_threshold,
            show=show
        )

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
            resp_range_max_f=max_resp_freq,
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
    all_rr_candidates = []
    if extract_rivs_succeeded:
        # ================= Pack RR Candidates =================
        # Build list of dictionarys containing RR candidates and associated info.
        # fmt: off
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
    if extract_rivs_succeeded:
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
    quality_indices.extend([{
        "method": "PPG noise",
        "feature": "PPG noise peak f",
        "value": ppg_max_noise_freq,
    }])
    quality_indices.extend([{
        "method": "PPG noise",
        "feature": "PPG noise ratio",
        "value": ppg_Pxx_max_over_median,
    }])
    quality_indices.extend([{
        "method": "PPG noise",
        "feature": "PPG noise mean Pxx",
        "value": ppg_Pxx_mean,
    }])
    
    # fmt: on

    rr_candidate_dict = {}
    if extract_rivs_succeeded:
        #  ================= RR Candidate Fusion  =================

        # drop ML in here. Candidates & quality -> ML -> single most likely candidate
        # [print(key, rivs_psd[key]["psd RR candidate"]) for key in rivs_psd.keys()]

        input_list = [
            np.round(rivs_psd[key]["nearest psd RR candidate"], 2)
            for key in rivs_psd.keys() if rivs_psd[key]["PSD RQI"] > params.psd_rqi_thresh
        ]
        rr_candidate_dict["PSD, closest to prev RR"] = np.median(input_list) if len(input_list) > 0 else None
        
        input_list = [
            np.round(rivs_psd[key]["psd RR candidate"], 2) 
            for key in rivs_psd.keys() if rivs_psd[key]["PSD RQI"] > params.psd_rqi_thresh
        ]
        rr_candidate_dict["PSD median"] = np.median(input_list) if len(input_list) > 0 else None

        input_list = [
            np.round(rivs_counting[key]["pk_count RR num pks"], 2)
            for key in rivs_counting.keys() if rivs_psd[key]["PSD RQI"] > params.psd_rqi_thresh
        ]
        rr_candidate_dict["Counting, median # peaks"] = np.median(input_list) if len(input_list) > 0 else None

        input_list = [
            np.round(rivs_counting[key]["pk_count RR median pk diff"], 2)
            for key in rivs_counting.keys() if rivs_psd[key]["PSD RQI"] > params.psd_rqi_thresh
        ]
        rr_candidate_dict["Counting, median pk delta rqi cutoff"] = np.median(input_list) if len(input_list) > 0 else None

        rr_candidate_dict["Counting, median pk delta std cutoff"] = merge_counting_candidates(
            rivs_counting, rivs_psd, peak_to_peak_std_cutoff=params.peak_to_peak_std_cutoff, psd_rqi_thresh=params.psd_rqi_thresh
        )

        rr_candidate_dict["kalman"] = rr_kalman
        
        fusion_candidates = [
            # "PSD, closest to prev RR",
            "PSD median",
            "Counting, median pk delta std cutoff",
            "kalman",
        ]
        fusion_candidate_rrs = [rr_candidate_dict[candidate] for candidate in fusion_candidates if rr_candidate_dict[candidate] is not None]
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
    return extract_rivs_succeeded, rr_candidate_dict, all_rr_candidates, quality_indices, hr

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
    fig_large: bool = False,
    save_fig: bool = False,
    save_psd_fig: bool = False,
    save_frame_psd_figs: bool = False,
    fig_suffix: str = "",
    rr_seed: Optional[float] = None,
    show_rr_candidates: bool = True
):
    """Estimates the RR from a single trial of a dataset.

    Each trial of a dataset contains at least three minutes of data. The PPG signal and reference
    signals are broken down into windows specified in params.window_size (typically 30 s), and
    following windows are incremented by params.window_increment (typically 5 s). This function then
    calls estimate_rr_single_frame().

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
        fig_large (bool): If True, size for big screens; else size for slides
        save_fig (bool, optional): Saves figure to data/ folder. Defaults to False.
        save_psd_fig (bool, optional): Saves PSD figure to data/ folder. Defaults to False.
        save_frame_psd_figs (bool, optional): Saves PSD figure for every frame. Defaults to False.
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
        trial_summary_df: DataFrame with meta and single-value highlights of 
            the entire trial (median, max, or similar, depending on the item),
            to facilitate making a simple summary of all trials.
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
        params=params,
        trial=trial_num,
        window_size=params.window_size,
        probe_num=params.probe,
        read_cpo=params.cpo,
        show=show_frame,
    )

    # exit function if no data exists for the current trial
    if ppg_raw is None:
        return None, None, None, None, None, None

    # We'll use these for individual_trial_fig() later, but we also need them for other figs sooner.
    root = import_ppg.GIT_ROOT
    result_path = f"{root}/data/results"
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if dataset.lower() == "kapiolani":
        suffix = f"P{params.probe}_{meta['id']}_{fig_suffix}_{current_time}"
    elif dataset.lower() == "3ps":
        suffix = f"{meta['id']}_{params.probe_type}{params.led_num}_{fig_suffix}_{current_time}"
    else:
        suffix = f"{meta['id']}_{fig_suffix}_{current_time}"

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
    # TODO: Add similar handling for mimic and vortal?

    # Prep frame-wise estimation of RR
    win_increment = params.window_increment
    win_size = params.window_size
    num_windows = ((len(ppg_raw) / fs_ppg) - win_size) / win_increment + 1
    if params.win_omit_first_and_last_frames:
        num_windows = math.floor(num_windows - 2)
    elif params.win_allow_partial_at_end:
        num_windows = math.ceil(num_windows)
    else:
        num_windows = math.floor(num_windows)

    if frame_num is not None:
        if np.any(np.array(frame_num) > num_windows):
            raise ValueError("Frame number exceeds the number of possible frames.")

    rr_candidate_merged_list, all_rr_candidates_list, feature_quality_list = [], [], []
    hr_list, other_metrics_list = [], []

    # Major variables:
    #
    #   frame_ref: Five metrics/labels for each key in ref_dict from import_ppg. Each leaf contains
    #   a list for all frames, making plotting easy. For datasets with nested reference data
    #   (multiple panelists, multiple passes), we nest these too but less heavily: if used, the uncertainty
    #   index, nested in ref_dict, is elevated to a top-level item here.
    #
    #   this_frame: Many metrics, some adapted from frame_ref, for the current frame. For the nested
    #   case, there are extra metrics for panelist (with an extra level to describe each panelist)
    #   and panel. 
    #
    #   frame_data: A list of this_frame for every frame. Not passed to individual_trial_fig().
    #
    #   frame_data_df: A DataFrame of frame_data, created after frame_data has been fully populated.

    frame_data: list[dict] = []                 
    frame_ref = {                               
        key: {"x": [], "y": [], "frame center": [], "y agg": [], "agg label": None}
        for key in ref_dict.keys()
    }

    t_list = frame_num if frame_num is not None else range(num_windows)
    rr_buffer = []
    no_messages_yet = True
    
    # Loop for each frame

    for t_idx in t_list:                
        if show:
            util.newprint(f"    frame {t_idx:3}", end='\r')
        this_frame = util.nested_dict()

        # =========== define current frame window ==========
        # the logic for t_end causes the last two windows to be shorter than the rest
        # because the last windows may share the same t_end. To get around this,
        # we use the true middle of each frame ((t_start + t_end) / 2) as x-ticks 
        # when making figures. 

        t_start = (t_idx + int(params.win_omit_first_and_last_frames)) * win_increment
        t_end = min(t_start + win_size, len(ppg_raw) // fs_ppg)
        win_size_actual = t_end - t_start

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
        # heart rate, breathing rate, panelist uncertain index (if used), capnography level. Signal may be None.
        
        # We define this function so that we can call it differently for nested elements
        def get_current_windows(this_ref_dict, input_key, store_in=None):
            if "x" in this_ref_dict[input_key] and not this_ref_dict[input_key]["x"] is None:
                if store_in is None:
                    frame_ref_alias = frame_ref[input_key]
                else:
                    frame_ref_alias = store_in

                # We will record this key, since it contains values for "x" for at least
                # some frames (before 2024-12-12 we recorded all keys).
                this_frame_ref_x = np.nan
                this_frame_ref_y = np.nan
                this_frame_ref_aggregated = np.nan
                
                ref_x = this_ref_dict[input_key]["x"]
                ref_win_idx = (ref_x >= t_start) & (ref_x < t_end)

                # duration of the reference window based on index
                if sum(ref_x[ref_win_idx]) > 2:
                    ref_win_dur = ref_x[ref_win_idx][-1] - ref_x[ref_win_idx][0]
                else:
                    ref_win_dur = 0
                    
                # it's possible for reference signal to be missing during the recording.
                if ref_win_dur > win_size_actual/2:
                    this_frame_ref_x = np.array(this_ref_dict[input_key]["x"][ref_win_idx])
                    this_frame_ref_y = np.array(this_ref_dict[input_key]["y"][ref_win_idx])
                    if input_key == "co2 breath counts":
                        this_frame_ref_aggregated = this_frame_ref_y[-1]
                        frame_ref_alias["agg label"] = "final count"
                    elif not np.all(np.isnan(this_frame_ref_y)):
                        if params.rate_at_frame_edge == "beyond_frame":
                            # Leave the start and end values as they are, the true average based on seeing
                            # beyond the frame.
                            pass    
                        elif "x events" in this_ref_dict[input_key]:
                            # Alter the values for the partial period (breath or beat) at the
                            # start and end of the frame. 
                            x_events = this_ref_dict[input_key]["x events"]     # for convenience
                            x_events_in_frame_idx = (x_events >= t_start) & (x_events < t_end)      # for Boolean filtering of x_events that are within the frame
                            if sum(x_events[x_events_in_frame_idx]) < 2:
                                # The frame doesn't include a whole breath, so prevent calculation of an aggregate
                                this_frame_ref_y = np.nan
                            else:
                                x_event_first = np.min(x_events[x_events_in_frame_idx])
                                x_event_last  = np.max(x_events[x_events_in_frame_idx])
                                before_and_after_breaths_idx = (this_frame_ref_x <= x_event_first) | (this_frame_ref_x > x_event_last)
                                if params.rate_at_frame_edge == "zeroed":
                                    this_frame_ref_y[before_and_after_breaths_idx] = 0  
                                elif params.rate_at_frame_edge == "excluded":
                                    this_frame_ref_y[before_and_after_breaths_idx] = np.nan
                                else:
                                    raise ValueError(f"params.rate_at_frame_edge = {params.rate_at_frame_edge}, which is not one of the expected values")

                        with warnings.catch_warnings():
                            # Because of the rate_at_frame_edge handling, the slice may now be all
                            # NaN. That's OK, so we ignore the warning.
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            this_frame_ref_aggregated = np.nanmean(this_frame_ref_y)
                        if params.rate_at_frame_edge == "zeroed":
                            # This method has a negative bias from dismissing both the starting and
                            # ending partial breaths. Remove this by, in effect, counting one of
                            # those as a breath. For our typical 30 s window, this adds 2 bpm.
                            this_frame_ref_aggregated += 60/win_size_actual
                        frame_ref_alias["agg label"] = "mean"

                    # else isnan, so leave unchanged

                frame_ref_alias["x"].append(this_frame_ref_x)
                frame_ref_alias["y"].append(this_frame_ref_y)
                frame_ref_alias["y agg"].append(this_frame_ref_aggregated)
                frame_ref_alias["frame center"].append(t_start + win_size_actual/2)

        for key in ref_dict.keys():
            if params.reference_rr_target_is_nested and key == params.reference_rr_target:
                for panelist_k, panelist_v in ref_dict[key].items():
                    # If not yet created (first frame), create each level of the dict as we drill down
                    if not panelist_k in frame_ref[key]:
                        frame_ref[key][panelist_k] = {"x": [], "y": [], "frame center": [], "y agg": [], "agg label": None}
                    for pass_k in panelist_v.keys():
                        if not pass_k in frame_ref[key][panelist_k]:
                            frame_ref[key][panelist_k][pass_k] = {"x": [], "y": [], "frame center": [], "y agg": [], "agg label": None}
                        get_current_windows(panelist_v, pass_k, store_in=frame_ref[key][panelist_k][pass_k])

            elif params.reference_rr_use_duration_markers and key == params.reference_rr_target[3:]:    # e.g., for "rr annnot post", "annot post", where we get the "uncertain" metric
                subkey = "uncertain"
                flat_key = key + " " + subkey       # e.g., "annot post uncertain"

                # Across panelists
                if not flat_key in frame_ref:
                    frame_ref[flat_key] = {"x": [], "y": [], "frame center": [], "y agg": [], "agg label": None}
                get_current_windows(ref_dict[key], subkey, store_in=frame_ref[flat_key])
                this_frame["RR uncertainty panel (mean)"] = frame_ref[flat_key]["y agg"][-1]

                # Per panelist
                for panelist_k, panelist_v in ref_dict[key]["panelists"].items():
                    if not panelist_k in frame_ref[flat_key]:
                        frame_ref[flat_key][panelist_k] = {"x": [], "y": [], "frame center": [], "y agg": [], "agg label": None}
                    get_current_windows(panelist_v, subkey, store_in=frame_ref[flat_key][panelist_k])
                    this_frame[f"RR uncertainty panelist {panelist_k} (mean)"] = frame_ref[flat_key][panelist_k]["y agg"][-1]

                    # Per pass
                    for pass_k, pass_v in panelist_v["passes"].items():
                        if not pass_k in frame_ref[flat_key][panelist_k]:
                            frame_ref[flat_key][panelist_k][pass_k] = {"x": [], "y": [], "frame center": [], "y agg": [], "agg label": None}
                        get_current_windows(pass_v[subkey], "state", store_in=frame_ref[flat_key][panelist_k][pass_k])
                        # Here, we don't bother to collect the uncertainty 

            else:
                get_current_windows(ref_dict, key)

        # Make reference RR be the average of specified reference signals, and calculate
        # disagreement as their range. 
        # 
        # For nested (panelist-pass) datasets, we calculate these metrics intra-panelist (across
        # passes) and inter-panelists (across the panel), as separate metrics from our standard
        # ones. We calculate these as nested dicts, but we also provide flattened versions for ease
        # of reading, and we nest the averages in frame_ref for plotting.
        # 
        # We can drop RRs if reference signals differ by some threshold, per Pimentel, et al (2017).
        # Toward a robust estimation of respiratory rate from pulse oximeters. IEEE Transactions on
        # Biomedical Engineering, 64(8), 1914–1923. https://doi.org/10.1109/TBME.2016.2613124

        this_frame_rrs = []
        # Collect the ref RRs that will go into the average and range
        for signal in params.reference_rr_signals:
            if signal is params.reference_rr_target and params.reference_rr_target_is_nested:

                for panelist_k, panelist_v in ref_dict[params.reference_rr_target].items():
                    
                    # Calculate intra-panelist stats as a separate, nested metric
                    this_frame_panelist_rrs = []

                    for pass_k in panelist_v.keys():
                        this_frame_panelist_rrs.append(frame_ref[params.reference_rr_target][panelist_k][pass_k]["y agg"][-1])

                    if np.all(np.isnan(this_frame_panelist_rrs)):
                        this_frame["avg rr ref panelist"][panelist_k] = np.nan
                        this_frame["RR ref disagreement panelist (bpm)"][panelist_k] = np.nan
                    else:
                        this_frame["avg rr ref panelist"][panelist_k] = np.nanmean(this_frame_panelist_rrs)
                        # Or maybe change to using the final pass, as some panelists suggested, or the median
                        if len(this_frame_panelist_rrs) == 1:      # if only one pass, don't calc disagreement
                            this_frame["RR ref disagreement panelist (bpm)"][panelist_k] = np.nan
                        else:
                            this_frame["RR ref disagreement panelist (bpm)"][panelist_k] = np.max(this_frame_panelist_rrs) - np.min(this_frame_panelist_rrs)
                    
                    # To make these easy to read, create flat versions too (duplicates of the nested dict)
                    this_frame[f"avg rr ref panelist {panelist_k}"] = this_frame["avg rr ref panelist"][panelist_k]
                    this_frame[f"RR ref disagreement panelist {panelist_k} (bpm)"] = this_frame["RR ref disagreement panelist (bpm)"][panelist_k]
                    
                    # To enable plotting, also store in frame_ref
                    frame_ref[params.reference_rr_target][panelist_k]["agg label"] = "panelist avg"
                    # I'm not sure what values would go in x and y below.
                    # frame_ref[params.reference_rr_target][panelist_k]["x"].append()
                    # frame_ref[params.reference_rr_target][panelist_k]["y"].append()
                    frame_ref[params.reference_rr_target][panelist_k]["y agg"].append(this_frame["avg rr ref panelist"][panelist_k])
                    # This is a crude way to get the frame center: from the most recent pass
                    frame_ref[params.reference_rr_target][panelist_k]["frame center"].append(frame_ref[params.reference_rr_target][panelist_k][pass_k]["frame center"][-1])
                    
                    # TODO: Calculate panelist's RR difference from the average RR

                # Calculate stats for the panel as a whole (with inter-panelist disagreement) as
                # another metric, using each panelist's average
                this_frame_panel_rrs = []
                for panelist_k in this_frame["avg rr ref panelist"]:
                    this_val = this_frame["avg rr ref panelist"][panelist_k]
                    # Skip NaNs so that we can calculate range
                    if not np.isnan(this_val):
                        this_frame_panel_rrs.append(this_val)
                if len(this_frame_panel_rrs) == 0:
                    avg_rr_ref_panel = np.nan
                    this_frame["RR ref disagreement panel (bpm)"] = np.nan
                else:
                    avg_rr_ref_panel = np.nanmean(this_frame_panel_rrs)
                    this_frame["RR ref disagreement panel (bpm)"] = np.max(this_frame_panel_rrs) - np.min(this_frame_panel_rrs)
                this_frame["avg rr ref panel"] = avg_rr_ref_panel
                this_frame["RR ref panel (mean)"] = avg_rr_ref_panel
                frame_ref[params.reference_rr_target]["agg label"] = "panel avg"
                frame_ref[params.reference_rr_target]["y agg"].append(avg_rr_ref_panel)
                frame_ref[params.reference_rr_target]["frame center"].append(frame_ref[params.reference_rr_target][panelist_k]["frame center"][-1])
                # Append the panel's average to any other references
                this_frame_rrs.append(avg_rr_ref_panel)
            else:   # non-nested
                this_frame_rrs.append(frame_ref[signal]["y agg"][-1])
            this_frame[signal] = this_frame_rrs[-1]

        # Using the collected ref RRs, calculate the average and range
        if np.all(np.isnan(this_frame_rrs)):
            current_rr_ref = np.nan
            this_frame["RR ref (mean)"] = np.nan
            this_frame["avg rr ref"] = np.nan
            this_frame["RR ref disagreement (bpm)"] = np.nan
        else:
            avg_rr_ref = np.nanmean(this_frame_rrs)
            this_frame["avg rr ref"] = avg_rr_ref
            if params.reference_rr_target == "":
                current_rr_ref = avg_rr_ref
            else:
                if params.reference_rr_target_is_nested:
                    current_rr_ref = avg_rr_ref_panel                                       # Poor form, since it doesn't allow for another reference to be selected, but works for now
                else:
                    current_rr_ref = frame_ref[params.reference_rr_target]["y agg"][-1]
            this_frame["RR ref (mean)"] = current_rr_ref
                
            current_rr_range = np.max(this_frame_rrs) - np.min(this_frame_rrs)
            this_frame["RR ref disagreement (bpm)"] = current_rr_range
        
        # Collect panelists' tags of video problems per panelist and combined for the whole panel,
        # as strings for readability
        if params.reference_rr_target_is_nested:
            probs_separator = ", "
            probs_panel = set()
            for panelist_k, panelist_v in ref_dict["annot post"]["panelists"].items():
                probs_panelist = set()
                for pass_k, pass_v in panelist_v["passes"].items():
                    if pass_v["video problems"]:
                        probs_panelist.update(pass_v["video problems"])
                        probs_panel.update(   pass_v["video problems"])
                this_frame[f"RR ref video problems panelist {panelist_k}"] = probs_separator.join(sorted(probs_panelist))
            this_frame["RR ref video problems panel"] = probs_separator.join(sorted(probs_panel))

        if ("rr capno" in params.reference_rr_signals) & (current_rr_ref is not np.nan):
            this_frame["RR capno count"] = frame_ref["co2 breath counts"]["y"][-1][-1]
            this_frame["RR capno p25"] = np.percentile(frame_ref["rr capno"]["y"][-1],25)
            this_frame["RR capno p75"] = np.percentile(frame_ref["rr capno"]["y"][-1],75)

        # # =============== reference heart rate ===============
        if "hr" in frame_ref:
            hr_ref = frame_ref["hr"]["y agg"][-1]
            this_frame["no hr ref"] = False
        else:
            hr_ref = np.nan
            this_frame["no hr ref"] = True
        # we will add this to this_frame later just for better presentation

        # seed with known RR value for 1st frame.
        # TODO: replace this logic with a lookup table, based on age?
        current_rr_est = current_rr_ref if rr_seed is None else rr_seed

        # =========== Frame-wise quality checks ============
        # some quality measures are calculated in estimate_rr_single_frame() to
        # improve performance.
        this_frame["no rr ref"] = np.isnan(current_rr_ref)
        if this_frame["no rr ref"] or this_frame["no hr ref"] or (current_rr_ref == 0):
            this_frame["aliased"] = None
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
            if params.reference_rr_target == "":
                capno_rr_std = None     # I'm not sure what else this should be
            else:
                capno_rr_std = np.std(frame_ref[params.reference_rr_target]["y"][-1])

            this_frame["co2 rr std/median"] = capno_rr_std / (current_rr_ref)

        # =========== process single frame ===========
        if save_frame_psd_figs:
            save_psd_fig_as=f"{result_path}/{dataset}_{suffix}_psd_f{t_idx:03}.png"
        else:
            save_psd_fig_as = None
        (
            rr_est_succeeded,
            rr_candidate_merged,
            all_rr_candidates,
            feature_quality,
            hr_estimated,
        ) = estimate_rr_single_frame(
            ppg=ppg_raw_win,
            fs_ppg=fs_ppg,
            max_resp_freq=params.RR_max_bpm / 60,
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
            params=params,
            save_psd_fig_as=save_psd_fig_as,
            show=show_frame,
        )
        
        # unpack reference information
        this_frame["HR ref (mean)"] = hr_ref
        try:
            this_frame["Ref RR/HR (mean)"] = this_frame["RR ref (mean)"] / this_frame["HR ref (mean)"]
        except:
            this_frame["Ref RR/HR (mean)"] = None

        hr_dict = dict()
        other_metrics_dict = dict()

        # A shorthand
        def copy_from_this_frame_to_other_metrics_dict(key_name):
            other_metrics_dict[key_name] = this_frame[key_name]

        if params.reference_rr_target_is_nested:
            for panelist_k in ref_dict[params.reference_rr_target].keys():
                copy_from_this_frame_to_other_metrics_dict(f"RR ref disagreement panelist {panelist_k} (bpm)")
            copy_from_this_frame_to_other_metrics_dict("RR ref disagreement panel (bpm)")
        copy_from_this_frame_to_other_metrics_dict("RR ref disagreement (bpm)")
        copy_from_this_frame_to_other_metrics_dict("Ref RR/HR (mean)")

        if hr_estimated is None:    
            # Estimation failed, so there's little to record.
            if show:
                newline = False
            else:
                newline = True
            util.newprint(f'    frame {t_idx:3}: HR could not be estimated.', on_a_new_line=newline)
            hr_dict["HR estimated (mean)"] = None
            hr_dict["HR est reliable"] = None
 
        else:
            # exclude median in candidate list
            temp = pd.DataFrame(feature_quality)
            if rr_est_succeeded:
                this_frame_quality = temp.loc[temp['feature']=='pct diagnostic quality pulses','value'].iloc[0]
            else:
                this_frame_quality = None

            # unpack additional heart rate information
            this_frame["HR estimated (mean)"] = hr_estimated['estimate']
            this_frame["HR est reliable"] = hr_estimated['reliable']
            this_frame["HR disagreement of means"] = abs(hr_estimated['estimate'] - hr_ref)
            this_frame["PPG quality"] = this_frame_quality
            this_frame["PPG noise peak f"]   = temp.loc[temp['feature']=='PPG noise peak f','value'].iloc[0]
            this_frame["PPG noise ratio"]    = temp.loc[temp['feature']=='PPG noise ratio','value'].iloc[0]
            this_frame["PPG noise mean Pxx"] = temp.loc[temp['feature']=='PPG noise mean Pxx','value'].iloc[0]
            
            if rr_est_succeeded:
                # ================== Buffer for User Display  =====================
                rr_buffer, rr_display = rr_display_buffer(rr_buffer, rr_candidate_merged, this_frame_quality)
                rr_candidate_merged["buffered_display"] = rr_display

                # if ppg data exist, update rr_seed for the next frame
                # may need to be smarter about this one, maybe "if ppg data is high quality"
                # rr_seed = rr_candidate_merged["kalman"]
                if not (this_frame["no ppg"]):
                    rr_seed = rr_display
                
            # ================== pack results to be returned ==================
            all_rr_candidates = [{**d, "frame index": t_idx} for d in all_rr_candidates]
            all_rr_candidates_list.extend(all_rr_candidates)
            feature_quality = [{**d, "frame index": t_idx} for d in feature_quality]
            feature_quality_list.extend(feature_quality)
            
            hr_dict["HR estimated (mean)"] = this_frame["HR estimated (mean)"]
            hr_dict["HR est reliable"] = this_frame["HR est reliable"]

            copy_from_this_frame_to_other_metrics_dict("HR disagreement of means")
            
            rr_candidate_merged_list.append(rr_candidate_merged)

        # collect frame-wise information
        hr_list.append(hr_dict)
        other_metrics_list.append(other_metrics_dict)
        frame_data.append(this_frame)

        # End of loop for each frame

    frame_data_df = pd.DataFrame(frame_data)
    # assert len(rr_candidate_merged_list) == frame_data_df.shape[0]
    
    rr_candidate_df=pd.DataFrame(rr_candidate_merged_list)

    hr_df = pd.DataFrame(hr_list)
    other_metrics_df = pd.DataFrame(other_metrics_list)

    # ============================ Display Results ============================
    if show or save_fig:
        if dataset == "kapiolani":
            subtitle = f"P{params.probe}; {meta['age']:.2f} years; {meta['notes']}"
        elif dataset == "mimic":
            subtitle = f"age: {meta['age']} years; gender: {meta['gender']}"
        else:
            subtitle_meta = meta.copy()
            subtitle_meta.pop('id')
            subtitle = f"{subtitle_meta}"   
            subtitle = subtitle.replace("\'","")   # strip single quotes
            subtitle = subtitle.replace("{","")
            subtitle = subtitle.replace("}","")

        # prepare a dataframe of signal qualities to visualize
        qualities_df = pd.DataFrame(feature_quality_list)
        temp = qualities_df[
            (qualities_df["method"] == "pct large stat-delta")
            | (
                (qualities_df["method"] == "template matching")
                & (qualities_df["feature"] == "pct diagnostic quality pulses")
            ) | (
                (qualities_df["method"] == "heart rate")
                & (qualities_df["feature"] == "reliable")
            )
        ]

        ppg_quality_df = temp.pivot(
            columns="feature", index="frame index", values="value"
        )
        with pd.option_context("future.no_silent_downcasting", True):
            ppg_quality_df = ppg_quality_df.replace({True:'1', False:'0'}).infer_objects(copy=False)

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
        ref_dict["avg rr ref"]["x"] = (frame_data_df["time start"] + frame_data_df["time end"]) / 2

        if dataset.lower() == '3ps':
            # Make a filtered PPG to line noise for display.
            # This calc is somewhat redundant, but elsewhere it is calculated per frame.
            if params.lowpass_cutoff_ppg is int or params.lowpass_cutoff_ppg is float:
                f_max = params.lowpass_cutoff_ppg
            else:
                f_max = 35      # The "dynamic" calc isn't available here, and anyway it's too aggressive a filter for this
            ppg_filt_fig = util.lowpass_bessel(ppg_raw, signal_fs=fs_ppg, order=8, f_max=f_max, show=False)
        else:
            ppg_filt_fig = None

        fig = individual_trial_fig(
            params = params,
            ppg_raw=ppg_raw,
            ppg_filt=ppg_filt_fig,
            ppg_fs=fs_ppg,
            ppg_quality_df=ppg_quality_df,
            ref_dict=ref_dict,
            frame_ref=frame_ref,
            rr_candidate_x=(frame_data_df["time start"] + frame_data_df["time end"]) / 2,
            rr_candidate_df=rr_candidate_df,
            hr_df=hr_df,
            other_metrics_df=other_metrics_df,
            title_str=(f"{dataset}: trial {trial_num}; id: {meta['id']}"),
            subtitle=subtitle,
            reference_shading=ref_RR_shading_borders,
            show_rr_candidates=show_rr_candidates,
            fig_large = fig_large,
        )

        # Spectrogram

        fig_spectrogram, ax = plt.subplots(figsize=(8,6))
        Pxx, freqa, bins, im = ax.specgram(ppg_raw, NFFT=1024, Fs=fs_ppg, scale='dB')
        fig_spectrogram.colorbar(im).set_label('Intensity [dB]')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

    if show:
        fig.show()
        # suppress warning
        warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive, and thus cannot be shown")
        fig_spectrogram.show()

    if save_fig:
        fig.write_image(f"{result_path}/{dataset}_{suffix}.png")
    
    if save_psd_fig:
        fig_spectrogram.savefig(f"{result_path}/{dataset}_{suffix}_spectrogram.png")
        save_psd_fig_as=f"{result_path}/{dataset}_{suffix}_PSD.png"
    else:
        save_psd_fig_as=None


    _, _, _ = sqi.ppg_mains_noise(ppg_raw, fs_ppg, show=show, save_fig_as=save_psd_fig_as)
    
    # ============================ Summarize Results ============================

    trial_summary = []

    def add_frame_data(metric, statistic, method):
        if (metric in frame_data_df) and (len(frame_data_df[metric].dropna()) > 0):
            return {'metric':metric, 'statistic':statistic, 'value':method(frame_data_df[metric].dropna())}
        else:
            return {'metric':metric, 'statistic':statistic, 'value':np.nan}

    def add_frame_data_framecount(metric, statistic):
        if (metric in frame_data_df):
            return {'metric':metric, 'statistic':statistic, 'value':frame_data_df[metric].count()}
        else:
            return {'metric':metric, 'statistic':statistic, 'value':np.nan}

    def add_frame_data_not(metric, statistic, method):
        if (metric in frame_data_df):
            return {'metric':metric, 'statistic':statistic, 'value':method(frame_data_df[metric].dropna() == False)}
        else:
            return {'metric':metric, 'statistic':statistic, 'value':np.nan}

    def add_frame_data_isna(metric, statistic, method):
        if (metric in frame_data_df):
            return {'metric':metric, 'statistic':statistic, 'value':method(frame_data_df[metric].isna())}
        else:
            return {'metric':metric, 'statistic':statistic, 'value':np.nan}

    # def add_all_rr(metric, statistic, method):
    #     return {'metric':metric, 'statistic':statistic, 'value':method(rr_candidate_df[metric].dropna())}
    
    def add_other(metric, statistic, value):
        return {'metric':metric, 'statistic':statistic, 'value':value}
    
    trial_summary.append(add_frame_data('RR ref (mean)',            'mean',             np.mean))
    trial_summary.append(add_frame_data('RR ref disagreement (bpm)',  'mean',             np.mean))
    trial_summary.append(add_frame_data('RR ref disagreement (bpm)',  'max',                max))

    if params.reference_rr_target_is_nested:
        trial_summary.append(add_frame_data('RR ref disagreement panel (bpm)',  'mean',             np.mean))
        trial_summary.append(add_frame_data('RR ref disagreement panel (bpm)',  'max',                max))
        for panelist_k in ref_dict[params.reference_rr_target].keys():
            trial_summary.append(add_frame_data(f'RR ref disagreement panelist {panelist_k} (bpm)',  'mean',             np.mean))
            trial_summary.append(add_frame_data(f'RR ref disagreement panelist {panelist_k} (bpm)',  'max',                max))
        if params.reference_rr_use_duration_markers:
            trial_summary.append(add_other(     'RR uncertainty panel (mean)',  'frames',     len(frame_data_df[frame_data_df['RR uncertainty panel (mean)'] > 0])))
            trial_summary.append(add_frame_data('RR uncertainty panel (mean)',  'mean',             np.mean))
            trial_summary.append(add_frame_data('RR uncertainty panel (mean)',  'max',                max))
            for panelist_k in ref_dict[params.reference_rr_target].keys():
                trial_summary.append(add_other(f'RR uncertainty panelist {panelist_k} (mean)',  'frames',     len(frame_data_df[frame_data_df[f'RR uncertainty panelist {panelist_k} (mean)'] > 0])))
        trial_summary.append(add_other(f'RR ref video problems panel',                  '-',      this_frame['RR ref video problems panel']))
        for panelist_k in ref_dict[params.reference_rr_target].keys():
            trial_summary.append(add_other(f'RR ref video problems panelist {panelist_k}',  '-',      this_frame[f'RR ref video problems panelist {panelist_k}']))

    trial_summary.append(add_frame_data('HR ref (mean)',            'mean',             np.mean))
    trial_summary.append(add_frame_data('HR estimated (mean)',      'mean',             np.mean))
    trial_summary.append(add_frame_data('HR disagreement of means', 'max',                max))
    trial_summary.append(add_frame_data('Ref RR/HR (mean)',         'mean',             np.mean))
    trial_summary.append(add_frame_data('Ref RR/HR (mean)',         'max',                max))

    trial_summary.append(add_frame_data_framecount('frame index',     'frames'))
    trial_summary.append(add_frame_data_not('HR est reliable',        'False frames',       sum))
    trial_summary.append(add_frame_data_isna('HR est reliable',       'NaN frames',         sum))

    trial_summary.append(add_frame_data('no rr ref',                  'frames',             sum))
    trial_summary.append(add_frame_data('aliased',                    'frames',             sum))
    trial_summary.append(add_frame_data('no ppg',                     'frames',             sum))
    trial_summary.append(add_frame_data('clipping',                   'frames',             sum))
    trial_summary.append(add_other     ('Raw PPG',                    'min',                min(ppg_raw)))
    trial_summary.append(add_other     ('Raw PPG',                    'max',                max(ppg_raw)))
    trial_summary.append(add_frame_data('PPG noise peak f',           'median',             median))
    trial_summary.append(add_frame_data('PPG noise ratio',            'mean',             np.mean))
    trial_summary.append(add_frame_data('PPG noise ratio',            'max',                max))
    trial_summary.append(add_frame_data('PPG noise mean Pxx',         'mean',             np.mean))
    trial_summary.append(add_frame_data('PPG noise mean Pxx',         'max',                max))    

    if "artifact_count" in ref_dict["co2"]:
        trial_summary.append(add_other('human-marked CO2 artifacts', 'count', ref_dict["co2"]["artifact_count"]))
    if "artifact_count" in ref_dict["ppg"]:
        trial_summary.append(add_other('human-marked PPG artifacts', 'count', ref_dict["ppg"]["artifact_count"]))


    # Create a DataFrame listing trial number, meta, and this summary

    supermeta = {'trial':trial_num} 
    supermeta.update(meta)

    meta_df = pd.DataFrame(list(supermeta.items()), columns=['metric', 'value'])
    meta_df.insert(1, 'statistic', '-')

    trial_summary_df = pd.concat([meta_df, pd.DataFrame(trial_summary).convert_dtypes()])

    return (
        frame_data,
        rr_candidate_merged_list,
        all_rr_candidates_list,
        feature_quality_list,
        meta,
        trial_summary_df
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
    original_candidates = [v for v in rr_candidates.values() if v is not None][:-2]
    
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
    params: Optional[AlgorithmParams] = None,
    show: bool = False,
    fig_large: bool = False,
    save_fig: bool = False,
    save_psd_fig: bool = False,
    save_frame_psd_figs: bool = False,
    save_df: bool = True,
    file_suffix: str = "",
    show_rr_candidates: bool = True,
    stop_on_error: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calls estimate_rr() over an entire dataset & gather tabulated results.

    Args:
        dataset (literal): "capnobase", "kapiolani", or similar.
        trials (list[int]): List of integers to run. Default None (all trials)
        params (dict): Algorithm parameters specified by the config dictionary.
        show (bool): If true, show individual trial results. Default False.
        fig_large (bool): If True, size for big screens; else size for slides
        save_fig (bool): If true, save figures for individual trial results.
            Default False.
        save_psd_fig (bool, optional): Saves PSD figure to data/ folder. Defaults to False.
        save_frame_psd_figs (bool, optional): Saves PSD figure for every frame (slow). Defaults to False.
        save_df (bool): If true, save results to csv file. Default True.
        file_suffix (str): string to append to the saved dataframe to note the
            experiment condition.
        show_rr_candidates: If False, hide the RR candidates so you can focus on 
            the ref RR.
        stop_on_error (bool): If False, keep going after a trial that causes an 
            error.

    Returns:
        - DataFrame summarizing the results of estimate_rr.
        - DataFrame summarizing all rr candidates
        - DataFrame summarizing all quality metrics
        - DataFrame combining the above, as written to CSV
        - DataFrame with one row per participant, compiled from each trial_summary_df
    """
    df = pd.DataFrame()
    all_rr_candidates = []
    all_quality_indices = []
    per_trial_df = pd.DataFrame()

    if params is None:
        params = AlgorithmParams(dataset=dataset)
    trials_to_eval = range(params.dataset_size) if trials is None else trials

    per_trial_row = 0
    for trial in trials_to_eval:
        util.newprint(f"processing trial: {trial:3}", end='\r')
        try:
            (
                frame_data,
                rr_candidate_merged,
                this_trial_rr_candidates,
                this_trial_feature_quality,
                meta,
                trial_summary_df
            ) = estimate_rr(
                trial_num=trial,
                dataset=dataset,
                params=params,
                show=show,
                fig_large=fig_large,
                save_fig=save_fig,
                save_psd_fig=save_psd_fig,
                save_frame_psd_figs=save_frame_psd_figs,
                fig_suffix=file_suffix,
                show_rr_candidates=show_rr_candidates
            )
            # clean up any plot objects to free memory and avoid warnings
            plt.close()

            # skip current trial if estimate_rr returns none.
            if frame_data is None:
                continue
        except Exception as e:
            # This will occur if there are fewer files than indicated by trials_to_eval,
            # which is not a serious error
            util.newprint(f"    encountered error: {e}", on_a_new_line=True)
            if stop_on_error:
                break
            else:
                continue        # proceed with the rest of the dataset

        # organize results summary into table; calculate error metric
        merged_candidate_df = pd.DataFrame(rr_candidate_merged)
        df_trial = pd.DataFrame(frame_data)
        df_trial["dataset"] = dataset
        # df_trial["trial"] = np.ones(len(frame_data)) * trial      # This seemed unnecessary, so I replaced it with the following
        df_trial["trial"] = trial
        df_trial["dataset-id"] = f'{dataset} {meta["id"]:03}'
        df_trial["subject id"] = meta["id"]
        df_trial["subject age"] = meta["age"]
        if dataset.lower() == "capnobase":
            df_trial["weight"] = meta["weight"]
            df_trial["ventilation"] = meta["ventilation"]
        if dataset.lower() == "3ps":
            df_trial["weight"] = meta["weight"]
            df_trial["ITA mean"] = meta["ITA mean"]
            df_trial["ITA SD"] = meta["ITA SD"]
        df_trial = pd.concat([df_trial, merged_candidate_df], axis=1)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",          
                category=FutureWarning, 
                message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. *"
            )
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

        per_trial_row += 1
        trial_summary_df.rename(columns={'value': per_trial_row}, inplace=True)    # Pandas requires uniqueness
        if per_trial_df.empty:
            per_trial_df = trial_summary_df.copy()
        else:
            # per_trial_df = pd.concat([per_trial_df, trial_summary_df['value']], axis=1)
            per_trial_df = per_trial_df.merge(trial_summary_df, how='outer', on=['metric', 'statistic'])

    if df.shape[0] > 0:
        df["trial"] = df["trial"].astype(int)
    df = df.reset_index()

    per_trial_df.set_index(['metric', 'statistic'], inplace=True)
    per_trial_df = per_trial_df.transpose()

    # save results
    if save_df:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if dataset.lower() == "kapiolani":
            probe = f"P{params.probe}_"  
        elif dataset.lower() == "3ps":
            probe = f"{params.probe_type}{params.led_num}_"
        else: 
            probe = ""
        suffix = f"{probe}{file_suffix}_{current_time}"

        # with one row per frame
        df_combo = util.pivot_and_merge(
            references=df, 
            qualities=pd.DataFrame(all_quality_indices), 
            candidates=pd.DataFrame(all_rr_candidates)
            )
        results_path = f"{import_ppg.GIT_ROOT}/data/results/"
        X_fn = f"{dataset}_{suffix}_features.csv"
        df_combo.to_csv(results_path + X_fn)

        # with one row per trial
        per_trial_fn = f"{dataset}_{suffix}_per_trial.csv"
        per_trial_df.to_csv(results_path + per_trial_fn)

        if False:
            util.newprint("", on_a_new_line=True)
            util.newprint(f"Results saved in {results_path}:")
            util.newprint(f"{X_fn}")
            util.newprint(f"{per_trial_fn}")

    return df, pd.DataFrame(all_rr_candidates), pd.DataFrame(all_quality_indices), df_combo, per_trial_df


def individual_trial_fig(
    params: AlgorithmParams,
    ppg_raw: npt.NDArray[np.float16],
    ppg_filt: npt.NDArray[np.float16],
    ppg_fs: float,
    ppg_quality_df: dict,
    ref_dict: dict,
    frame_ref: dict,
    rr_candidate_x: npt.NDArray[np.float16],
    rr_candidate_df: pd.DataFrame,
    hr_df: pd.DataFrame,
    other_metrics_df: pd.DataFrame,
    title_str: str,
    subtitle: str,
    reference_shading = None,
    show_rr_candidates: bool = True,
    fig_large: bool = False
) -> go.Figure:
    """Display PPG, RR estimates, and other info.

    Args:
        params (dict): Algorithm parameters, used for display ranges
        ppg_raw: Raw PPG signal
        ppg_filt: Filtered PPG signal. Optional.
        ppg_fs: PPG sampling frequency, used to create time-axis for plotting.
        ref_dict: Nested dictionary containing reference signals,
            with 'x' and 'y' values. e.g.,
            ref_dict['co2']['x'], ref_dict['co2']['y']
        frame_ref: nested dictionary containing frame-wise reference values.
        rr_candidate_x: time-axis of estimated RR and the other DFs below. For 
            frame-wide metrics, these are the times as the middle of each frame.
        rr_candidate_df: DataFrame containing RR candidates in each column.
            expects a column named "frame index" to indicate how they should be
            grouped.
        hr_df: DataFrame containing HR estimate(s).
        other_metrics_df: DataFrame containing HR and RR disagreement.
        title_str: Figure title
        subtitle: Figure subtitle
        show_rr_candidates: Whether to display the RR candidates. Default: True.
        fig_large (bool): If True, size for big screens; else size for slides
    """
    # Check inputs
    if not (set(params.reference_rr_signals).issubset(ref_dict.keys())):
        raise KeyError(
            f"{params.reference_rr_signals} are not a subset of the keys of ref_dict"
        )

    # Setup
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    line_opacity = 0.5    # for overlapping lines
    rows = 8    # you must synchronize this with the max of current_row below
    row_heights = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.5]  # relative heights of each row
    if params.comparator_signal_raw is not None:
        # add a row for the comparator
        rows += 1
        row_heights.insert(-1, 0.2)
    if params.dataset.lower() == "3ps":
        # add rows at the top for the accelerometers
        rows += 2
        row_heights.insert(0, 0.1)
        row_heights.insert(0, 0.1)
    if params.reference_rr_use_duration_markers:
        # add row above Reference signal for panel uncertainty
        rows += 1
        row_heights.insert(-2, 0.1)

    fig = make_subplots(
        rows=rows,     
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    if params.reference_rr_target_is_nested:
        panelist_colors = [             # Using brightened Adobe Premiere label colors. Note that alpha and opacity don't work for lines.
            "rgb( 70, 178, 200)",       # iris
            "rgb( 91,  31, 255)",       # violet
            "rgb( 68, 152, 136)",       # teal
        ]

    # Subplots
    current_row = 0         # We increment this before creating each subplot

    # Accelerometers (3ps only)
    if params.dataset.lower() == "3ps":
        acc_colors = ["orchid", "seagreen", "orange", "black"]
        
        for j, acc in enumerate(["acc chest", "acc probe"]):
            current_row += 1
            title_text = "Accel<br>probe" if acc == "acc probe" else "Accel<br>chest"
            fig.update_yaxes(title_text=title_text, row=current_row, col=1)

            for i, axis in enumerate(["x", "y", "z", "magnitude"]):
                fig.add_trace(
                    go.Scatter(
                        x=ref_dict[acc]["t"], 
                        y=ref_dict[acc][f"acc {axis}"], 
                        mode="lines", 
                        line={"color":acc_colors[i], "width":1},
                        name=axis, 
                        hovertemplate = "%{y:.3f}",
                        legendgroup=current_row,
                        legendgrouptitle_text = "Accelerometers",
                        showlegend = False if j else True,          # To save space, create legends only for the first set
                    ),
                    row=current_row, 
                    col=1
                )

    # PPG signal
    current_row += 1
    fig.update_yaxes(title_text="PPG", row=current_row, col=1, 
        # range=[params.ppg_min-1, params.ppg_max+1]
    )
    fig.update_xaxes(row=current_row, col=1, showticklabels=True)

    t_axis = np.linspace(0, len(ppg_raw) / ppg_fs, len(ppg_raw))
    fig.add_trace(
        go.Scatter(
            x=t_axis, 
            y=ppg_raw, 
            mode="lines", 
            line={"color":"coral", "width":0.7},
            name="Raw PPG", 
            legendgroup=current_row,
            legendgrouptitle_text = 'PPG',
        ),
        row=current_row, 
        col=1
    )
    # Markers on the PPG according to the dataset's human rater, if available
    if "peak_x" in ref_dict["ppg"]:
        fig.add_trace(
            go.Scatter(
                x=ref_dict["ppg"]["peak_x"],
                y=ref_dict["ppg"]["peak_y"],
                name="human-marked PPG peaks",
                mode="markers",
                legendgroup=current_row,
                marker=dict(
                    symbol="triangle-down-open",
                    color="navy",
                    opacity=0.75,
                    size=12
                )
            ),
            row=current_row,
            col=1,
        )

    if "artifact_count" in ref_dict["ppg"]:
        if ref_dict["ppg"]["artifact_count"] > 0:
            artifacts = ref_dict["ppg"]["artifact_x"]
            # At least in capnobase, the artifact markers appear to be in pairs
            # that bracket the duration of the artifact
            for i in range( len(artifacts) // 2 ):
                x0 = artifacts[i*2]
                x1 = artifacts[i*2 + 1]
                if i == 0:
                    name = 'human-marked artifacts'
                    showlegend = True
                else:
                    name = ''
                    showlegend = False
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    row=current_row,
                    col=1,
                    fillcolor="red",
                    opacity=0.15,
                    line_width=0,
                    name = name,
                    legendgroup=current_row,
                    showlegend = showlegend
                )
    if ppg_filt is not None:
        fig.add_trace(
            go.Scatter(
                x=t_axis, 
                y=ppg_filt, 
                mode="lines", 
                line={"color":"mediumslateblue", "width":0.7},
                name="Filtered PPG", 
                legendgroup=current_row,
                legendgrouptitle_text = 'PPG',
            ),
            row=current_row, 
            col=1
        )

        
    # PPG Quality Metrics
    current_row += 1
    fig.update_yaxes(title_text="PPG<BR>quality", range=[-0.03, 1.03], row=current_row, col=1)

    for col in ppg_quality_df.columns:
        fig.add_scatter(
            x=rr_candidate_x,
            y=ppg_quality_df[col],
            mode="lines",
            name=col,
            hovertemplate = "%{y:.2f}",
            legendgroup=current_row,
            legendgrouptitle_text = 'Quality',
            row=current_row,
            col=1,
        )

    # RR/HR ratio for Nyquist check
    current_row += 1
    fig.update_yaxes(
        row=current_row, 
        col=1,
        title_text="Ref<BR>RR/HR",
        autorangeoptions=dict(
            minallowed = -0.03,
            include = 0.55,
        ),
    )

    fig.add_scatter(
        x=rr_candidate_x,
        y=other_metrics_df['Ref RR/HR (mean)'],
        mode="lines", 
        name='Ref RR/HR (mean)',
        hovertemplate = "%{y:.2f}",
        legendgroup=current_row,
        legendgrouptitle_text = 'RR/HR (Nyquist check)',
        row=current_row,
        col=1,
    )

    # Nyquist warning line. Oddly, this code won't work if you put it before the add_scatter().
    fig.add_hline(
        y = 0.5,
        line_width = 1,
        line_color = 'gray',
        line_dash = 'dot',
        label = dict(
            text = ' Nyquist',
            textposition = 'start',
            yanchor = 'top'
        ),
        row = current_row,
        col = 1
    )

    # Heart rate
    current_row += 1
    fig.update_yaxes(title_text="HR<br>(bpm)", row=current_row, col=1)

    if 'hr' in ref_dict:
        fig.add_trace(
            go.Scatter(
                x=ref_dict['hr']['x'],
                y=ref_dict['hr']['y'],
                name='HR ref',
                hovertemplate = "%{y:.1f}",
                legendgroup=current_row,
                line={'color':'navy', 'dash':'solid', 'width':0.5}
            ),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame_ref['hr']['frame center'],
                y=frame_ref['hr']['y agg'],
                name='HR ref mean',
                hovertemplate = "%{y:.1f}",
                legendgroup=current_row,
                line={'color':'navy'}
            ),
            row=current_row,
            col=1,
        )
    fig.add_scatter(
        x=rr_candidate_x,
        y=hr_df['HR estimated (mean)'],
        mode="lines", 
        name='HR estimated (mean)',
        hovertemplate = "%{y:.1f}",
        legendgroup=current_row,
        legendgrouptitle_text = 'Heart rate',
        line={'color':'darkorange'},
        row=current_row,
        col=1,
    )

    # HR Disagreement
    current_row += 1
    fig.update_yaxes(
        title_text="HR dis<br>(bpm)", 
        row=current_row, 
        col=1,
        autorangeoptions=dict(
            minallowed = -0.1,
            include = 5,        # Ensure that tiny disagremeents look tiny
        ),
    )
    fig.add_scatter(
        x=rr_candidate_x,
        y=other_metrics_df['HR disagreement of means'],
        mode="lines", 
        name="HR disagreement of means",  
        hovertemplate = "%{y:.1f}",
        legendgroup=current_row,
        legendgrouptitle_text = 'HR disagreement',
        line={'color':'coral'},
        showlegend = False,
        row=current_row,
        col=1,
    )

    # RR Disagreement
    current_row += 1
    fig.update_yaxes(
        title_text="RRR dis<br>(bpm)", 
        row=current_row, 
        col=1,
        autorangeoptions=dict(
            minallowed = -0.1,
            include = 3,        # Ensure that tiny disagremeents look tiny
        ),
        # range=[0,20],
    )
    if params.reference_rr_target_is_nested:
        for idx, (panelist_k, panelist_v) in enumerate(ref_dict[params.reference_rr_target].items()):
            # If there are multiple passes, plot the panelist's disagreement
            if len(panelist_v.keys()) > 1:
                fig.add_scatter(
                    x=rr_candidate_x,
                    y=other_metrics_df[f'RR ref disagreement panelist {panelist_k} (bpm)'],
                    mode="lines", 
                    name=f"panelist {panelist_k}",
                    hovertemplate = "%{y:.1}",
                    line={'color':panelist_colors[idx], 'width':0.75},
                    legendgroup=current_row,
                    # legendgrouptitle_text = 'RRR dis',
                    showlegend = True,
                    row=current_row,
                    col=1,
                )
        # If there are multiple panelists, plot the panel's disagreement
        if len(ref_dict[params.reference_rr_target].keys()) > 1:
                fig.add_scatter(
                    x=rr_candidate_x,
                    y=other_metrics_df[f'RR ref disagreement panel (bpm)'],
                    mode="lines", 
                    name=f"panel",
                    hovertemplate = "%{y:.1f}",
                    line={'color':'dimgray', 'width':2.5},
                    legendgroup=current_row,
                    # legendgrouptitle_text = 'RRR dis',
                    showlegend = True,
                    row=current_row,
                    col=1,
                )

    fig.add_scatter(
        x=rr_candidate_x,
        y=other_metrics_df['RR ref disagreement (bpm)'],
        mode="lines", 
        name="overall",
        hovertemplate = "%{y:.1f}",
        line={'color':'black', 'width':2.5},
        legendgroup=current_row,
        legendgrouptitle_text = 'Ref RR disagreement',
        showlegend = True,
        row=current_row,
        col=1,
    )
        
    # 3ps: Uncertain state (0-1) of breath marks
    if params.reference_rr_use_duration_markers:

        current_row += 1
        fig.update_yaxes(title_text='Uncer-<BR>tain', range=[-0.05,1.05], row=current_row, col=1)
        hovertemplate_text = "%{y:.0%}"

        # per pass
        for panelist_idx, panelist_k in enumerate(ref_dict["annot post"]["panelists"].keys()):
            for pass_idx, pass_k in enumerate(ref_dict["annot post"]["panelists"][panelist_k]["passes"].keys()):
                fig.add_trace(
                    go.Scatter(
                        x=ref_dict["annot post"]["panelists"][panelist_k]["passes"][pass_k]["uncertain"]["state"]["x"],
                        y=ref_dict["annot post"]["panelists"][panelist_k]["passes"][pass_k]["uncertain"]["state"]["y"],
                        name=f"{panelist_k} {pass_k}",
                        legendgroup=current_row,
                        legendgrouptitle_text = 'Annot post uncertain state',
                        hovertemplate = hovertemplate_text,
                        mode="lines",
                        line=dict(
                            color=panelist_colors[panelist_idx],
                            width=0.4,
                        ),
                    ),
                    row=current_row,
                    col=1,
                )
        # per panelist average
        for panelist_idx, panelist_k in enumerate(ref_dict["annot post"]["panelists"].keys()):
            fig.add_trace(
                go.Scatter(
                    x=ref_dict["annot post"]["panelists"][panelist_k]["uncertain"]["x"],
                    y=ref_dict["annot post"]["panelists"][panelist_k]["uncertain"]["y"],
                    name=f"{panelist_k} mean",
                    legendgroup=current_row,
                    hovertemplate = hovertemplate_text,
                    mode="lines",
                    line=dict(
                        color=panelist_colors[panelist_idx],
                        width=0.7,
                    ),
                ),
                row=current_row,
                col=1,
            )
        # per panel average
        fig.add_trace(
            go.Scatter(
                x=ref_dict["annot post"]["uncertain"]["x"],
                y=ref_dict["annot post"]["uncertain"]["y"],
                name=f"panel mean",
                legendgroup=current_row,
                hovertemplate = hovertemplate_text,
                mode="lines",
                line=dict(
                    color="gray",
                    width=1,
                ),
            ),
            row=current_row,
            col=1,
        )
        # The frame-wide mean, "annot post uncertain"
        # For clarity, plot only the uncertainty values > 0
        fig.add_trace(
            go.Scatter(
                x=frame_ref["annot post uncertain"]["frame center"],
                y=[y if y > 0 else None for y in frame_ref['annot post uncertain']['y agg']],
                name=f"frame mean where > 0",
                legendgroup=current_row,
                hovertemplate = hovertemplate_text,
                mode="lines",
                line=dict(
                    color="black",
                    width=1.5,
                ),
            ),
            row=current_row,
            col=1,
        )


    # Gold standard (GS) signal
    current_row += 1
    fig.update_yaxes(title_text='Reference<BR>signal', row=current_row, col=1)

    fig.add_trace(
        go.Scatter(
            x=ref_dict[params.gold_standard_signal_raw]['x'],
            y=ref_dict[params.gold_standard_signal_raw]['y'],
            name=params.gold_standard_signal_label,
            legendgroup=current_row,
            legendgrouptitle_text = 'Reference respiratory signals',
            line={"color": "LightSkyBlue"},
        ),
        row=current_row,
        col=1,
    )
    # Markers on the gold standard for our detected start of each breath, according to our algorithm
    if "rr ref breath starts" in ref_dict:
        fig.add_trace(
            go.Scatter(
                x=ref_dict["rr ref breath starts"]["x"],
                y=ref_dict["rr ref breath starts"]["y"],
                name="GS signal breath starts",
                legendgroup=current_row,
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    color="darkorange",
                    opacity=0.9,
                    size=8
                )
            ),
            row=current_row,
            col=1,
        )
    # Markers on the gold standard according to the dataset's human rater, if available
    if "startexp_x" in ref_dict[params.gold_standard_signal_raw]:
        fig.add_trace(
            go.Scatter(
                x=ref_dict[params.gold_standard_signal_raw]["startexp_x"],
                y=ref_dict[params.gold_standard_signal_raw]["startexp_y"],
                name="human-marked expiration starts",
                legendgroup=current_row,
                mode="markers",
                marker=dict(
                    symbol="triangle-down-open",
                    color="navy",
                    opacity=0.75,
                    size=12
                )
            ),
            row=current_row,
            col=1,
        )
    if "ip annot1" in ref_dict:
        fig.add_trace(
            go.Scatter(
                x=ref_dict["ip annot1"]["x"],
                y=ref_dict["ip annot1"]["y"],
                name="human-1-marked breath starts",
                legendgroup=current_row,
                mode="markers",
                marker=dict(
                    symbol="triangle-down-open",
                    color="navy",
                    opacity=0.6,
                    size=12
                )
            ),
            row=current_row,
            col=1,
        )
    if "ip annot2" in ref_dict:
        fig.add_trace(
            go.Scatter(
                x=ref_dict["ip annot2"]["x"],
                y=ref_dict["ip annot2"]["y"],
                name="human-2-marked breath starts",
                legendgroup=current_row,
                mode="markers",
                marker=dict(
                    symbol="triangle-down-open",
                    color="green",
                    opacity=0.6,
                    size=12
                )
            ),
            row=current_row,
            col=1,
        )

    if "startinsp_x" in ref_dict[params.gold_standard_signal_raw]:
        fig.add_trace(
            go.Scatter(
                x=ref_dict[params.gold_standard_signal_raw]["startinsp_x"],
                y=ref_dict[params.gold_standard_signal_raw]["startinsp_y"],
                name="human-marked inspiration starts",
                legendgroup=current_row,
                mode="markers",
                marker=dict(
                    symbol="triangle-up-open",
                    color="navy",
                    opacity=0.75,
                    size=12
                )
            ),
            row=current_row,
            col=1,
        )
    if "artifact_count" in ref_dict["co2"]:
        if ref_dict["co2"]["artifact_count"] > 0:
            artifacts = ref_dict["co2"]["artifact_x"]
            for i in range( len(artifacts) // 2 ):
                x0 = artifacts[i*2]
                x1 = artifacts[i*2 + 1]
                if i == 0:
                    name = 'human-marked artifacts'
                    showlegend = True
                else:
                    name = ''
                    showlegend = False
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    row=current_row,
                    col=1,
                    fillcolor="red",
                    opacity=0.15,
                    line_width=0,
                    name = name,
                    legendgroup=current_row,
                    showlegend = showlegend
                )
    if "onp dataset breaths" in ref_dict:
        fig.add_trace(
            go.Scatter(
                x=ref_dict["onp dataset breaths"]["x"],
                y=ref_dict["onp dataset breaths"]["y"],
                name="dataset breath starts",
                legendgroup=current_row,
                mode="markers",
                marker=dict(
                    symbol="triangle-down-open",
                    color="navy",
                    opacity=0.75,
                    size=12
                )
            ),
            row=current_row,
            col=1,
        )
    if "onp NH breath starts" in ref_dict:
        fig.add_trace(
            go.Scatter(
                x=ref_dict["onp NH breath starts"]["x"],
                y=ref_dict["onp NH breath starts"]["y"],
                name="onp NH breath starts",
                legendgroup=current_row,
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    color="darkorange",
                    opacity=0.9,
                    size=8
                )
            ),
            row=current_row,
            col=1,
        )
    # 3ps
    if "annot live" in ref_dict:
        fig.add_trace(
            go.Scatter(
                x=ref_dict["annot live"]["x"],
                y=ref_dict["annot live"]["y"],
                name="annot live",
                legendgroup=current_row,
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    color="LightSkyBlue",
                    opacity=0.9,
                    size=8
                )
            ),
            row=current_row,
            col=1,
        )
    if params.reference_rr_target_is_nested:
        if params.reference_rr_use_duration_markers:
            # any uncertain periods, plotted underneath the breath marks
            for panelist_idx, panelist_k in enumerate(ref_dict["annot post"]["panelists"].keys()):
                for pass_idx, pass_k in enumerate(ref_dict["annot post"]["panelists"][panelist_k]["passes"].keys()):
                    if "count" in ref_dict["annot post"]["panelists"][panelist_k]["passes"][pass_k]["uncertain"]:
                        fig.add_trace(
                            go.Scatter(
                                x=ref_dict["annot post"]["panelists"][panelist_k]["passes"][pass_k]["uncertain"]["count"]["x"],
                                y=ref_dict["annot post"]["panelists"][panelist_k]["passes"][pass_k]["uncertain"]["count"]["y"],
                                name=f"annot post uncertain {panelist_k} {pass_k}",
                                legendgroup=current_row,
                                mode="lines",
                                line=dict(
                                    color="yellow",
                                    dash="solid",
                                    width=3,
                                ),
                            ),
                            row=current_row,
                            col=1,
                        )
        # the breath marks themselves, regardless of uncertain periods
        for panelist_idx, panelist_k in enumerate(ref_dict["annot post"]["panelists"].keys()):
            for pass_idx, pass_k in enumerate(ref_dict["annot post"]["panelists"][panelist_k]["passes"].keys()):
                fig.add_trace(
                    go.Scatter(
                        x=ref_dict["annot post"]["panelists"][panelist_k]["passes"][pass_k]["breaths"]["x"],
                        y=ref_dict["annot post"]["panelists"][panelist_k]["passes"][pass_k]["breaths"]["y"],
                        name=f"annot post {panelist_k} {pass_k}",
                        legendgroup=current_row,
                        mode="lines+markers",
                        line=dict(
                            color=panelist_colors[panelist_idx],
                            width=0.7,
                        ),
                        marker=dict(
                            symbol="x-thin",
                            angle=pass_idx * 30,    # an asterisk when passes align
                            color=panelist_colors[panelist_idx],
                            size=8,
                            line=dict(
                                color=panelist_colors[panelist_idx],
                                width=1
                            )
                        )
                    ),
                    row=current_row,
                    col=1,
                )


    # Comparator signal (currently used only in vortal)
    if params.comparator_signal_raw is not None:
    
        current_row += 1
        fig.update_yaxes(title_text='Comparator<BR>signal', row=current_row, col=1)

        fig.add_trace(
            go.Scatter(
                x=ref_dict[params.comparator_signal_raw]['x'],
                y=ref_dict[params.comparator_signal_raw]['y'],
                name=params.comparator_signal_label,
                legendgroup=current_row,
                legendgrouptitle_text = 'Comparator signal',
                line={"color": "darkturquoise"},
            ),
            row=current_row,
            col=1,
        )

        if "ip NH breath starts" in ref_dict:
            fig.add_trace(
                go.Scatter(
                    x=ref_dict["ip NH breath starts"]["x"],
                    y=ref_dict["ip NH breath starts"]["y"],
                    name="ip NH breath starts",
                    legendgroup=current_row,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        color="darkorange",
                        opacity=0.9,
                        size=8
                    )
                ),
                row=current_row,
                col=1,
            )

    # ========================== RR ==========================
    current_row += 1
    fig.update_yaxes(
        title_text="Respiratory rate<br>(bpm)", 
        row=current_row, col=1, 
        # we set range options later
    )

    # RR References
    ref_colors = ["navy", "darkorange", "gray", "green"]
    for idx, ref_key in enumerate(params.reference_rr_signals):
        if params.reference_rr_target_is_nested and ref_key is params.reference_rr_target:
            for panelist_idx, panelist_k in enumerate(ref_dict["annot post"]["panelists"].keys()):
                for pass_idx, pass_k in enumerate(ref_dict["annot post"]["panelists"][panelist_k]["passes"].keys()):
                    # Each pass's instantaneous. These can be distractingly noisy, so we sometimes disable them.
                    if False:
                        fig.add_trace(
                            go.Scatter(
                                x=ref_dict["rr annot post"][panelist_k][pass_k]["x"],
                                y=ref_dict["rr annot post"][panelist_k][pass_k]["y"],
                                mode="lines",
                                name=f"{ref_key} {panelist_k} {pass_k} inst",
                                hovertemplate = "%{y:.1f}",
                                legendgroup=current_row,
                                legendgrouptitle_text = 'RR references',
                                line={"color": panelist_colors[panelist_idx], "width": 0.5, "dash":"solid"},
                            ),
                            row=current_row,
                            col=1,
                        )
                        # If there are multiple passes, plot each's average.
                        if len(ref_dict["annot post"]["panelists"][panelist_k]["passes"].keys()) > 1:      
                                # Each pass's frame means
                                fig.add_trace(
                                    go.Scatter(
                                        x=frame_ref[ref_key][panelist_k][pass_k]["frame center"],
                                        y=frame_ref[ref_key][panelist_k][pass_k]["y agg"],
                                        mode="lines",
                                        name=f"{ref_key} {panelist_k} {pass_k}",
                                        hovertemplate = "%{y:.1f}",
                                        legendgroup=current_row,
                                        legendgrouptitle_text = 'RR references',
                                        line={"color": panelist_colors[panelist_idx], "width": 0.9},
                                    ),
                                    row=current_row,
                                    col=1,
                                )
                # Each panelist's average
                if frame_ref[ref_key][panelist_k]["y agg"][0] >= 0:
                    fig.add_trace(
                        go.Scatter(
                            x=frame_ref[ref_key][panelist_k]["frame center"],
                            y=frame_ref[ref_key][panelist_k]["y agg"],
                            mode="lines",
                            name=f"{ref_key} {panelist_k} avg",
                            hovertemplate = "%{y:.1f}",
                            legendgroup=current_row,
                            legendgrouptitle_text = 'RR references',
                            opacity=line_opacity,
                            line={"color": panelist_colors[panelist_idx], "width": 3},
                        ),
                        row=current_row,
                        col=1,
                    )
            # The panel's average will be reported after this conditional block
        if len(params.reference_rr_signals) > 1:
            fig.add_trace(
                go.Scatter(
                    x=frame_ref[ref_key]["frame center"],
                    y=frame_ref[ref_key]["y agg"],
                    mode="lines",
                    name=f"{ref_key} {frame_ref[ref_key]['agg label']}",
                    hovertemplate = "%{y:.1f}",
                    legendgroup=current_row,
                    legendgrouptitle_text = 'RR references',
                    opacity=line_opacity,
                    line={"color": ref_colors[idx], "dash": "dash", "width": 3},
                ),
                row=current_row,
                col=1,
            )

        # These instantaneous (non-mean) values can be distractingly noisy, so we sometimes disable them.
        if False:
            fig.add_trace(
                go.Scatter(
                    x=ref_dict[ref_key]["x"],
                    y=ref_dict[ref_key]["y"],
                    mode="lines",
                    name=f"{ref_key}",
                    hovertemplate = "%{y:.1f}",
                    legendgroup=current_row,
                    legendgrouptitle_text = 'RR references',
                    line={"color": ref_colors[idx], "dash": "dot"},
                ),
                row=current_row,
                col=1,
            )
    fig.add_trace(
        go.Scatter(
            x=ref_dict["avg rr ref"]["x"],
            y=ref_dict["avg rr ref"]["y"],
            mode="lines",
            name=f"avg of means of RR refs",
            hovertemplate = "%{y:.1f}",
            legendgroup=current_row,
            opacity=line_opacity,
            line={"color": "#2c2421", "dash": "solid", "width": 4},
        ),
        row=current_row,
        col=1,
    )

    # Set range to at least include the average ref value ± 5, to not exaggerate small differences
    plot_y_avg = round(np.mean(ref_dict["avg rr ref"]["y"]), 0)
    if plot_y_avg >= 0:
        fig.update_yaxes(
            row=current_row, col=1, 
            autorangeoptions=dict(
                include = [plot_y_avg - 5, plot_y_avg + 5],
                clipmin = 0,
                clipmax = 120,
            ),
        )
    else:
        fig.update_yaxes(
            row=current_row, col=1, 
            autorangeoptions=dict(
                clipmin = 0,
                clipmax = 120,
            ),
        )

    # Shade ± 3 bpm from average RR as a visual cue 
    fig.add_scatter(
        x=ref_dict["avg rr ref"]["x"],
        y=np.array(ref_dict["avg rr ref"]["y"]) - 2,
        fill=None,
        mode='lines',
        name="avg of means of RR refs ± 3 bpm",
        hoverinfo="skip",
        line=dict(width=0),
        showlegend = False,
        row=current_row,
        col=1,
        )
    fig.add_scatter(
        x=ref_dict["avg rr ref"]["x"],
        y=np.array(ref_dict["avg rr ref"]["y"]) + 2,
        fill='tonexty',
        fillcolor='rgba(145, 178, 180, 0.2)',
        mode='lines',
        name="avg of means of RR refs ± 3 bpm",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend = True,
        legendgroup=current_row,
        row=current_row,
        col=1,
        )

    # # shade +/- 2 breaths of the breath counts
    # if "co2 breath counts" in params.reference_rr_signals:
    #     fig.add_scatter(
    #         x=frame_ref["co2 breath counts"]["frame center"], 
    #         y=np.array(frame_ref["co2 breath counts"]["y agg"])-2,
    #         fill=None,
    #         mode='lines',
    #         line=dict(width=0.1),
    #         showlegend = False,
    #         row=current_row,
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
    #         row=current_row,
    #         col=1,
    #         )
    
    # the shaded area between 25th and 75th percentile
    if reference_shading is not None:
        fig.add_scatter(
            x=frame_ref[ref_key]["frame center"], 
            y=reference_shading["RR capno p25"],
            fill=None,
            mode='lines',
            hovertemplate = "%{y:.1f}",
            line=dict(width=0.1),
            showlegend = False,
            row=current_row,
            col=1,
            )
        fig.add_scatter(
            x=frame_ref[ref_key]["frame center"], 
            y=reference_shading["RR capno p75"],
            fill='tonexty',
            fillcolor='rgba(145, 178, 180, 0.486)',
            mode='lines',
            hovertemplate = "%{y:.1f}",
            line=dict(width=0.1),
            showlegend = False,
            row=current_row,
            col=1,
            )

    # RR candidates
    if show_rr_candidates:
        for col in rr_candidate_df.columns:
            fig.add_scatter(
                x=rr_candidate_x,
                y=rr_candidate_df[col],
                mode="lines",  # 'lines' or 'markers'
                name=col,
                hovertemplate = "%{y:.1f}",
                legendgroup=current_row + 1,
                legendgrouptitle_text = 'RR candidates',
                row=current_row,
                col=1,
            )

    # ========================== General formatting ==========================

    fig.update_xaxes(title_text="Time (s)", row=current_row, col=1)
    fig.update_xaxes(
        # dtick=15, 
        range=[min(t_axis), max(t_axis)]    # Otherwise, Plotly pads both ends because the plot uses markers
        # range=[135,175],
    )
    # Repeat the x-axis tick labels on the first subplot (as well as at the bottom)
    # fig.update_layout(xaxis_showticklabels=True)

    # Enforce consistent decimal places within each subplot
    for row in range(1, rows + 1):
        fig.update_yaxes(
            row = row, col = 1,
            tickformatstops = [
                {'dtickrange':[None, 0.0009], 'value': ".4f"},
                {'dtickrange':[0.0009, 0.009], 'value': ".3f"},
                {'dtickrange':[0.009,0.09], 'value': ".2f"},
                {'dtickrange':[0.09,0.9], 'value': ".1f"},
                {'dtickrange':[0.9,9], 'value': ".0f"},
            ]
        )
    # But show more decimal places in hover text
    # fig.update_traces(hovertemplate = "%{y:.3f}")   
    
    # Ensure that hover labels are not truncated (https://community.plotly.com/t/how-to-not-abbreviate-the-hover-text/49838)
    fig.update_layout(hoverlabel_namelength=-1)

    if fig_large:
        height = 1600
        width = int(height * 16/9)
    else:
        height = 1200
        width = int(height * 16/9)
        # height = 1000
        # width = 1000
    sub_wrap = "<br>".join(textwrap.wrap(subtitle, width=width/3))
    
    fig.update_layout(
        title_text=f"{title_str}<br><span style='font-size:12px'>{sub_wrap}</span>",
        height=height,
        width=width,
        legend_tracegroupgap = 20,
        legend_groupclick = 'toggleitem',   # unfortunately you can toggle either groups or items, not both
        hovermode = "x",
    )

    return fig

    # TODO: Evaluate
    # compare with HeartPy?
    # https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/algorithmfunctioning.html # noqa E501
    # and RapidHRV https://peerj.com/articles/13147/
