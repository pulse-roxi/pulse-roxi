from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union, Dict


@dataclass
class AlgorithmParams:
    """Parameters for the RR estimation algorithm and hyperparameters for
    algorithm evaluation.

    NOTE: There may be other parameters floating around (especially default 
    parameters) that have not been added to this class yet.

    Args:
        dataset: name of dataset to load
        probe (int): 1 or 2, For Kapi'olani dataset, probe 1 or 2 to be imported.
        probe_type (str): For 3ps dataset, probe "Tr"ansmissive or "Re"flective to be imported.
        led_num (int): 1 or 2. For 3ps, LED1 or LED2 (not sure which is which wavelength).
        cpo (bool): Whether custom probe is used, only applicable to kapiolani dataset.
            Default to False.
        window_size (int, seconds): Length of analyis window. Defaults to 30 seconds.
        window_increment (int, seconds): Increment between successive analysis windows.
            Defaults to 5 seconds.
        win_omit_first_and_last_frames (bool): If True, do not process the first and last frame.
            This avoids edge effects, such as the reference RR not being calculated before the first
            breath event or after the last. Omission is the safest approach, but for 3ps it appears
            that omission is not necessary: breath marks start quite near the beginning, and they
            always extend past the last frame (180 s). Default: False.
        win_allow_partial_at_end (bool): Allow a reduced-length frame at the end of the trial.
            Otherwise, the frames stop after the last full-size frame. Default to False, and
            effectively forced to False if win_omit_first_and_last_frames is True.
        duration_limit (int): Ignore data past this many seconds. Useful to avoid processing bad
            data when the recording accidentally runs long. Currently used only with 3ps. 
            It may be preferable to pad this slightly to avoid edge effects, such as partial breath
            rates. Default to 0 (no limit).
        reference_rr_signals (List[str]): Signals that could be used as references.
            The list needs to corresponds to the names of the reference signals.
            Each data set would have its own list of reference signals.
        reference_rr_target_is_nested (bool): 3ps uses a nested dict for its panel of raters.
        reference_rr_use_duration_markers (bool): 3ps trials contain period markers from the 
            panel of raters. To analyze those, set to True. Default: False.
        reference_rr_expand_other_duration_markers (bool): 3ps batch 1 (the first 100 sessions) was
            marked under the rule that duration markers named anything other than "uncertain" (or
            "un" anything) are construed to exclude the entire session. Subsequent batches were
            instead marked for exactly the duration to be excluded. Default: False (for subsequent batches).
        gold_standard_signal_raw (str): The underlying signal used for reference RR. 
            For several datasets, including capnobase, this is the CO2 signal, and it
            plotted for each individual subject. 
            Warning: For some datasets, this is currently set to an RR reference 
            (in breaths per minute), for unknown reasons.
        rate_at_frame_edge: For frame-wise aggregation, what rate to use for the partial period
            (heartbeat or breath) at the start and at the end of the frame. 
            TODO: Add "x events" key to load_ppg_data() for datasets other than 3ps so that this
            has an effect.
            - 'beyond_frame' (default) uses the un-framed rate. It is accurate but it may not
            be an appropriate reference when judging the accuracy of an algorithm that doesn't have 
            access to data beyond the frame.
            - 'zeroed' sets those rates to zero. This causes the average rate to behave like simple
            counting of whole breaths, including a negative bias of one breath per frame (2 bpm for 
            a 30 s frame) and worst-case error of twice that.
            - 'excluded': the aggregation ignores the partial breaths. The effect is less
            predictable but tends towards a smaller positive bias but a larger distribution than
            with 'zeroed'.
        ppg_min, ppg_max (int): If these are exceeded during a frame, the frame will be classified
            as "clipping".
        rr_min_bpm, rr_max_bpm (int): Minimum and maximum expected breaths per minute.
            Limits the algorithm search range, can help improve performance.
        hr_min_bpm, hr_max_bpm (int): Minimum and maximum expected heart rates in 
            beats per minute.
        lowpass_cutoff_ppg (float or str, hz): The cutoff frequency used to pre-process
            raw ppg signals default "dynamic". Empirical testing showed that using a 
            dynamic cutoff = 2.1*pulse_rate yields better results over a constant value.
        lowpass_dynamic_scalar_ppg (float): The scalar that is multiplied times the estimated pulse
            rate when using "dynamic". Default: 2.1.
        fs_riv (float, hz): Sampling Frequency of the respiratory induced waveforms.
            literature suggests 10-12 hz. From empirical testing, 12 hz yields better
            performance than 10 hz.
        remove_riv_outliers (str): Either "segment-wise" or "point-wise". Removes
            sections of the PPG waveforms corrupted with poor signal.
        n_kalman_fusion (int): Number of waveforms used for Kalman-Fusion.
        peak_counting_prominence (float): The find_peaks parameter used in 
            peak_counting_analysis. The default value is 0.5, based on the assumption
            that the RIV has been normalized to have mean 0 and standard deviation of 1.
        min_deviation_tolerance (dict[str, float]): threshold used for outlier 
            detection. Deviations smaller than this value are not considered outliers.
        ppg_quality_corr_threshold (float): minimum correlation between each
            individual pulse with the average pulse to be considered diagnostic quality.
        min_keep_pct: minimum percentage of frame to be kept. Segments shorter than 
            this are not kept. Between 0 and 1.
        psd_rqi_thresh: peaks below this threshold are ignored. Default: 0.04.
        peak_to_peak_std_cutoff: The quality metric we are using is "RIV peak-diff std".
            Counting candidates with corresponding std > peak_to_peak_std_cutoff are not 
            used. Defaults to 13, empirically determined (old documentation said 10 but the 
            code used 13).
    """

    dataset: str
    probe: Optional[int] = 1
    probe_type: Literal["Tr", "Re"] = None
    led_num: Optional[int] = 1
    cpo: bool = False

    window_size: int = 30  # seconds
    window_increment: int = 5  # seconds
    win_allow_partial_at_end: bool = False
    win_omit_first_and_last_frames: bool = False
    duration_limit: int = 0 # seconds

    reference_rr_use_post: bool = False
    reference_rr_target_is_nested: bool = False
    reference_rr_use_duration_markers: bool = False
    reference_rr_expand_other_duration_markers: bool = False
    uncertain_edge_offset: float = 0.01     # transition time in seconds when a panelist's uncertain period starts or stops
    rate_at_frame_edge: Literal["beyond_frame", "zeroed", "excluded"] = "beyond_frame"

    # quality metric
    ppg_quality_corr_threshold: float = 0.99
    psd_rqi_thresh: float = 0.04

    # Algorithm parameters
    ## General parameters
    RR_min_bpm: int = 15
    RR_max_bpm: int = 100
    hr_max_bpm: int = 240

    ## Pre-processing
    lowpass_cutoff_ppg: Union[float,Literal["dynamic"]] = "dynamic"
    lowpass_dynamic_scalar_ppg: float = 2.1
    
    ## RIV-processing
    fs_riv: int = 12
    remove_riv_outliers: Literal["segment-wise", "point-wise"] = "segment-wise"
    n_kalman_fusion: int = 3
    peak_counting_prominence: float = 0.5
    peak_to_peak_std_cutoff: float = 18
    outlier_tolerance: float = 5
    min_keep_pct: float = 0.5

    min_deviation_tolerance: dict = field(default_factory=lambda: {
        "RIFV_max": 25,
        "RIFV_min": 25,
        "RIFV_mas": 25,
        "AUDP": 0.01,
        "RIIV_upper": 0.05,
        "RIIV_lower": 0.05,
        "ppg": 2,
        "RIAV_descending": 0.2,
        "RIAV_ascending": 0.2,
        "STT": 0.2,
        "notch area ratio": 0.05,
        "notch rel amp": 0.05,
    })

    # Dataset-specific hyperparameters; initialized in __post_init__
    dataset_size: int = field(init=False)
    gold_standard_signal_raw: str = field(init=False)
    gold_standard_signal_label: str = field(init=False)
    comparator_signal_raw: str = None
    comparator_signal_label: str = None
    reference_rr_target: str = field(init=False)
    reference_rr_signals: List[str] = field(init=False)
    reference_hr_target: str = field(init=False)
    ppg_max: int = field(init=False)
    ppg_min: int = field(init=False)
    capno_max: Optional[int] = None
    capno_min: Optional[int] = None

    def __post_init__(self):
        if self.dataset.lower() == "synthetic":
            self.dataset_size = 87
            self.gold_standard_signal_raw = "rr"
            self.gold_standard_signal_label = "Resp Rate"
            self.reference_rr_target = "rr"
            self.reference_rr_signals = ["rr"]
            self.reference_hr_target = "hr"
            self.ppg_max = 1
            self.ppg_min = 0

            # Algorithm params
            self.RR_min_bpm = 4
            self.RR_max_bpm = 60
            
        elif self.dataset.lower() == "mimic":
            self.dataset_size = 53
            self.gold_standard_signal_raw = "ip"
            self.gold_standard_signal_label = "IP"
            self.reference_rr_target = ""     # Instead of picking one, use the average
            # self.reference_rr_signals = ["rr ip", "rr dataset", "rr annot1", "rr annot2"]
            self.reference_rr_signals = ["rr annot1", "rr annot2"]
            self.reference_hr_target = "hr"
            self.hr_min_bpm: int = 35
            
            self.ppg_max = 1
            self.ppg_min = 0

            # Algorithm params
            self.RR_min_bpm = 4
            self.RR_max_bpm = 60

        elif self.dataset.lower() == "capnobase":
            self.dataset_size = 42
            self.ppg_max = 10.16
            self.ppg_min = -10.24
            self.gold_standard_signal_raw = "co2"
            self.gold_standard_signal_label = "Capnography"
            self.reference_rr_target = "rr capno"
            self.reference_rr_signals = ["rr", "rr capno"]
            self.reference_hr_target = "hr"
            
            self.hr_min_bpm: int = 35
            self.capno_max = 3182
            self.capno_min = 400

            # Algorithm params
            self.RR_min_bpm = 5
            self.RR_max_bpm = 78

        elif self.dataset.lower() == "kapiolani":
            self.dataset_size = 126
            self.gold_standard_signal_raw = "co2"
            self.gold_standard_signal_label = "Capnography"
            self.reference_rr_target = "rr capno"
            self.reference_rr_signals = ["rr capno", "rr monitor", "co2 breath counts"]
            self.reference_hr_target = "hr monitor"

            self.capno_max = 3182
            self.capno_min = 400
            self.hr_min_bpm: int = 60
            if self.cpo is True:
                self.ppg_max = 16380
                self.ppg_min = 0
            else:
                self.ppg_max = 3071
                self.ppg_min = 1024

            # Algorithm params
            self.RR_min_bpm = 15
            self.RR_max_bpm = 120

        elif self.dataset.lower() == "3ps":
            self.dataset_size = 509
            self.gold_standard_signal_raw = "annot live"            # Since we can't show the video, we show this running breath count,
            self.gold_standard_signal_label = "annot live count"    # even though it's not a gold standard
            
            self.reference_rr_use_post = True
            if self.reference_rr_use_post:
                # When we have post (panel) breath marks, use them as the reference
                self.reference_rr_target_is_nested = True               # Levels per panelist and per pass
                self.reference_rr_target = "rr annot post"
                self.reference_rr_signals = ["rr annot post", "rr annot live"]
                self.reference_rr_use_duration_markers = True
            else:
                # When we don't, use the live annotation as the reference
                self.reference_rr_target_is_nested = False
                self.reference_rr_target = "rr annot live"
                self.reference_rr_signals = ["rr annot live"]

            # Set this to False unless using an old version of Batch 1 prior to 20250731, when we
            # manually expanded its markers to match the style of later batches
            self.reference_rr_expand_other_duration_markers = False

            self.short_breath_threshold = 0.3333                # In seconds. Breaths shorter than this will be reported, since they may be erroneous (two markers where there should be one)

            self.reference_hr_target = None

            self.hr_min_bpm: int = 60
            self.ppg_max =   -20000                             # Just inside of the 2^21 DAC limit (22-bit 2s complement)
            self.ppg_min = -2050000                             # We receive a positive signal but invert it to match PPG convention
            self.duration_limit = 184

            # Algorithm params
            self.RR_min_bpm = 15
            self.RR_max_bpm = 120

        elif self.dataset.lower() == "vortal":
            self.dataset_size = 39
            self.ppg_max = 10.16    # This and most numbers below were copied from other datasets. They might not be right.
            self.ppg_min = -10.24
            self.gold_standard_signal_raw = 'onp'
            self.gold_standard_signal_label = 'Oral-nasal pressure'
            self.comparator_signal_raw = 'ip'
            self.comparator_signal_label = "Impedance pneumography"
            self.reference_rr_target = "rr onp dataset breaths"
            self.reference_rr_signals = ["rr onp dataset breaths", "rr ip monitor"]
            self.reference_hr_target = "hr"
            
            self.hr_min_bpm: int = 35
            self.capno_max = 3182
            self.capno_min = 400

            # Algorithm params
            self.RR_min_bpm = 5
            self.RR_max_bpm = 78

        else:
            raise KeyError(
                f"dataset: '{self.dataset}' is undefined. Please check your spelling? " +
                "Must be one of: 'kapiolani', '3ps', 'capnobase', 'mimic', 'vortal', or 'synthetic'."
            )

    @property
    def ppg_range(self):
        return self.ppg_max - self.ppg_min
    
    @property
    def riv_list(self):
        return list(self.min_deviation_tolerance.keys())