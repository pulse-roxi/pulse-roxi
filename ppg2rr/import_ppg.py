import glob
import os
import warnings
from collections import defaultdict
from os.path import join as pjoin
from os.path import splitext
from typing import Literal, Optional
import re

import git  # for finding the current repo root directory
import h5py
import pymatreader as pymatreader
import numpy as np
import pandas as pd
import scipy.io
import scipy.interpolate
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from plotly import express as px
import bisect
from itertools import chain
from datetime import datetime, timezone, timedelta

from ppg2rr import util
from ppg2rr.capnography import rr_from_capnography, rolling_window_count
from ppg2rr import config
from ppg2rr import heart_rate

cwd = os.getcwd()


def get_git_root(path):
    """Returns the current git repo root directory."""
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


GIT_ROOT = os.path.normpath(get_git_root(cwd))

def read_matlab(filename):
    """Converts .mat file into nested dictionary.

    Many published datasets are in the .mat format, such as the synthetic PPG dataset.
    This function was used to help convert the .mat file to the .parquet format.
    
    This function is unsuccessful at reading the mimic dataset's 'bidmc_data.mat', so an alternate
    approach is used for that.

    """

    def conv(path=""):
        """File is loaded with h5py.
        
        From https://stackoverflow.com/a/58026181. 
        """
        p = path or "/"
        paths[p] = ret = {}
        for k, v in f[p].items():
            if type(v).__name__ == "Group":
                ret[k] = conv(f"{path}/{k}")  # Nested struct
                continue
            v = v[()]  # It's a Numpy array now
            if v.dtype == "object":
                # HDF5ObjectReferences are converted into a list of actual pointers
                ret[k] = [r and paths.get(f[r].name, f[r].name) for r in v.flat]
            else:
                # Matrices and other numeric arrays
                ret[k] = v if v.ndim < 2 else v.swapaxes(-1, -2)
        return ret

    def load_mat_scipy(filename=None, data={}, loaded=None):
        """Use scipy.io.loadmat to read the .mat file.

        source: https://stackoverflow.com/questions/62995712/extracting-mat-files-with-structs-into-python # noqa E501
        """
        if filename:
            vrs = scipy.io.whosmat(filename)
            name = vrs[0][0]
            loaded = scipy.io.loadmat(filename, struct_as_record=True)
            loaded = loaded[name]
        whats_inside = loaded.dtype.fields
        fields = list(whats_inside.keys())
        for field in fields:
            if len(loaded[0, 0][field].dtype) > 0:  # it's a struct
                data[field] = {}
                data[field] = load_mat_scipy(
                    data=data[field], loaded=loaded[0, 0][field]
                )
            else:  # it's a variable
                data[field] = loaded[0, 0][field]
        return data

    paths = {}
    try:
        with h5py.File(filename, "r") as f:
            # print("using h5py")
            return conv()
    except OSError:
        # print("using scipy")
        return load_mat_scipy(filename)



def load_ppg_data(
    dataset: Literal["capnobase", "kapiolani", "3ps", "mimic", "vortal", "synthetic"],
    params,
    trial: int,
    window_size: int = 30,
    probe_num: Optional[int] = None,
    read_cpo: bool = False,
    show: bool = False,
):
    """Import raw PPG data from the given dataset.

    Args:
        dataset (str): Name of dataset we experiment on. Assumes dataset is located in
            the 'repository_root/data/' directory.
        params (AlgorithmParams): Dataset-specific parameters.
        trial (int): The trial number to load
        window_size (int): Used to count the number of breaths in the previous 
            window_size seconds. Defaults to 30.
        read_cpo (bool): Only applicable if dataset == 'Kapiolani'. Read data from
            custom pulse ox (CPO) or not. Defaults to False.
        probe_num (int): For Kapi'olani dataset, probe 1 or 2 to be imported.
            Default None.
        show: If true, display imported data

    Returns:
        ppg_raw: raw ppg waveform
        fs_ppg
        ref - dictionary containing reference breathing and heart rate, gold-standard
            reference traces
        meta - ventilation, age, subject_id, etc
    """
    root_dir = pjoin(GIT_ROOT, "data")  # relative to execution script filepath

    # prepare output dictinoaries
    ref = util.nested_dict()
    meta = {}

    if dataset == "synthetic":
        df = pd.read_parquet(path=f"{root_dir}/Synthetic/synthetic_ppg.parquet")
        this_trial = df.iloc[trial]
        ppg_raw = this_trial['ppg_v']
        fs_ppg = this_trial['ppg_fs']
        ref = {}
        ref['hr'] = this_trial['hr_beatsPerMin']
        ref['rr'] = this_trial['rr_breathsPerMin']
        return ppg_raw, fs_ppg, ref, meta

    elif dataset.lower() == "capnobase":
        data_dir = pjoin(root_dir, "CapnoBase", "RR benchmark", "mat")
        mat_files = glob.glob(pjoin(data_dir, "*mat"))
        mat_files.sort()    # otherwise, sorting may vary depending on the file system

        # read and parse
        mat = read_matlab(mat_files[trial])

        fs_co2 = mat["param"]["samplingrate"]["co2"][0][0]
        fs_ppg = mat["param"]["samplingrate"]["pleth"][0][0]

        # reference resp rates and heart rates over time, based on a human rater's markings described later
        # Oddly, the CapnoBase RR values seem to lag by 15 s, which we correct here.
        # That happens to be half of our 30 s window, but the CapnoBase RR is supposed to be, and
        # looks to be, instantaneous, yet the lag is obvious and the CapnoBase RR values continue 
        # after the end of the capnography signal at 480 s recording.
        ref["rr"]["x_raw"] = mat["reference"]["rr"]["co2"]["x"][0] - 15
        ref["rr"]["y_raw"] = mat["reference"]["rr"]["co2"]["y"][0]
        ref["hr"]["x_raw"] = mat["reference"]["hr"]["pleth"]["x"][0]
        ref["hr"]["y_raw"] = mat["reference"]["hr"]["pleth"]["y"][0]

        # ppg and capnography traces
        ppg_raw = mat["signal"]["pleth"]["y"].flatten()
        # Eventually, the following ref["ppg"] pair could replace the above ppg_raw for consistency
        ref["ppg"]["x_raw"] = np.linspace(0, len(ppg_raw) / fs_ppg, len(ppg_raw))  # adapted from rr_est.individual_trial_fig()
        ref["ppg"]["y_raw"] = ppg_raw

        ref["co2"]["y_raw"] = mat["signal"]["co2"]["y"].flatten()
        ref["co2"]["x_raw"] = np.arange(len(ref["co2"]["y_raw"])) / fs_co2
        ref["co2"]["fs"] = fs_co2
        assert len(ref["co2"]["y_raw"]) == len(ref["co2"]["x_raw"])

        # Capnography 
        
        # Markings by a human rater (or maybe just validated by a human)
        # We use unique() because some trials have repeated x values, strangely
        ref["co2"]["startexp_x"] = np.unique(mat["labels"]["co2"]["startexp"]["x"][0] / fs_co2)
        ref["co2"]["startexp_y"] = util.subset(ref["co2"]["startexp_x"], ref["co2"]["x_raw"], ref["co2"]["y_raw"], fs_co2)

        ref["co2"]["startinsp_x"] = np.unique(mat["labels"]["co2"]["startinsp"]["x"][0] / fs_co2)
        ref["co2"]["startinsp_y"] = util.subset(ref["co2"]["startinsp_x"], ref["co2"]["x_raw"], ref["co2"]["y_raw"], fs_co2)

        # I've observed that if there are no artifacts, the contents are [0, 0].
        # When there are artifacts, they appear in pairs, apparently at the start
        # and end of each artifact period.
        if mat["labels"]["co2"]["artif"]["x"][0].all() == 0:
            ref["co2"]["artifact_x"] = None
            ref["co2"]["artifact_count"] = 0
        else:
            ref["co2"]["artifact_x"] = mat["labels"]["co2"]["artif"]["x"][0] / fs_co2
            ref["co2"]["artifact_count"] = len(ref["co2"]["artifact_x"])

        # PPG 
        # Markings by a human rater (or maybe just validated by a human)
        # We aren't using these, and for some reason we found too few matches to the PPG timebase,
        # so we've disabled these.
        # ref["ppg"]["peak_x"] = mat["labels"]["pleth"]["peak"]["x"][0] / fs_ppg
        # ref["ppg"]["peak_x"] = util.round_to_multiple(ref["ppg"]["peak_x"], 1/fs_ppg, 5)
        # ref["ppg"]["peak_y"] = util.subset(ref["ppg"]["peak_x"], ref["ppg"]["x_raw"], ref["ppg"]["y_raw"])

        # See notes on CO2 artifacts above
        if mat["labels"]["pleth"]["artif"]["x"][0].all() == 0:
            ref["ppg"]["artifact_x"] = None
            ref["ppg"]["artifact_count"] = 0
        else:
            ref["ppg"]["artifact_x"] = mat["labels"]["pleth"]["artif"]["x"][0] / fs_ppg
            ref["ppg"]["artifact_count"] = len(ref["ppg"]["artifact_x"])

        # Calculate RR from CO2 reference & interpolate to 1hz
        rr_capno_t, rr_capno, breaths_x, breaths_y = rr_from_capnography(
            co2=ref["co2"]["y_raw"], fs_co2=fs_co2, show=show
        )
        if len(rr_capno) > 2:
            interp_rr_capno_t, interp_rr_capno = util.interpolate(
                rr_capno_t,
                rr_capno,
                kind="slinear",
            )
            breath_count_t, breath_count_y = rolling_window_count(
                times = rr_capno_t, window_length=window_size
                )
            ref["co2 breath counts"]["x"] = breath_count_t
            ref["co2 breath counts"]["y"] = breath_count_y
            ref["rr capno"]["x"] = interp_rr_capno_t
            ref["rr capno"]["y"] = interp_rr_capno
            ref["rr ref breath starts"]["x"] = breaths_x
            ref["rr ref breath starts"]["y"] = breaths_y
        else:
            ref["rr capno"]["breath times"] = None
            ref["rr capno"]["x"] = None
            ref["rr capno"]["y"] = None
            ref["rr ref breath starts"]["x"] = None
            ref["rr ref breath starts"]["y"] = None

        # interpolate unevenly sampled reference signals into 1 sample/second
        for key in ["rr", "hr", "co2"]:
            if sum(np.abs(np.diff(np.diff(ref[key]["x_raw"])))) > 5:
                # remove inf or nans from reference
                invalid_values = np.isinf(ref[key]["y_raw"]) | np.isnan(
                    ref[key]["y_raw"]
                )
                valid_x_raw = ref[key]["x_raw"][~invalid_values]
                valid_y_raw = ref[key]["y_raw"][~invalid_values]
                # interpolate
                interp_ref_x, interp_ref_y = util.interpolate(
                    valid_x_raw,
                    valid_y_raw,
                    kind="slinear",
                )
                ref[key]["x"] = interp_ref_x
                ref[key]["y"] = interp_ref_y
            else:
                ref[key]["x"] = ref[key]["x_raw"]
                ref[key]["y"] = ref[key]["y_raw"]

        # meta["id"] = mat["param"]["case"]["id"].astype(np.uint8).tobytes().decode("ascii")
        id = mat["param"]["case"]["id"].astype(np.uint8).tobytes().decode("ascii") 
        # An example ID is '0014_8min'. Let's turn that into the int 14.
        if len(id) == 9 and id[-5:] == '_8min':
            id = int(id[0:4])
        meta['id'] = id
        meta["age"] = mat["meta"]["subject"]["age"][0][0]
        meta["weight"] = mat["meta"]["subject"]["weight"][0][0]
        meta["gender"] = mat["meta"]["subject"]["gender"][0]
        meta["ventilation"] = mat["param"]["case"]["ventilation"].astype(np.uint8).tobytes().decode("ascii")

        # print(ref['rr'])
        # print(ref['rr'].keys())

        return ppg_raw, fs_ppg, ref, meta

    elif dataset.lower() == "vortal":
        
        """ 
        This section was adapted from the mimic section.
        
        Regarding the dataset's reference for breaths and RR: "Windows in which the pressure signal
        had a low signal-to-noise ratio were excluded from the analysis. The threshold for exclusion
        was chosen to eliminate windows in which breaths could not be identified visually." [1] The
        dataset doesn't shows these exclusions--nor does it even include the authors' RR reference.

        [1] Charlton PH, Bonnici T, Tarassenko L, Clifton DA, Beale R, Watkinson PJ. An assessment
        of algorithms to estimate respiratory rate from the electrocardiogram and
        photoplethysmogram. Physiol Meas. 2016 Apr;37(4):610-26. doi: 10.1088/0967-3334/37/4/610.
        Epub 2016 Mar 30. PMID: 27027672; PMCID: PMC5390977. 
        
        """

        data_dir = pjoin(root_dir, "Vortal")
        mat_file_path = pjoin(data_dir, "VORTAL_rest_data.mat")
        data = pymatreader.read_mat(mat_file_path)["data"]  

        # ppg waveform
        ppg_raw = data["ppg"][trial]["v"]
        fs_ppg = data["ppg"][trial]["fs"]

        # Impedance pneumography (IP) signal
        fs_ip = data["ref"][trial]["resp_sig"]["imp"]["fs"]
        ref["ip"]["y"] = data["ref"][trial]["resp_sig"]["imp"]["v"]
        ref["ip"]["x"] = np.arange(len(ref["ip"]["y"])) / fs_ip

        # Oral-nasal pressure (ONP) signal
        fs_onp = data["ref"][trial]["resp_sig"]["paw"]["fs"]
        ref["onp"]["y"] = data["ref"][trial]["resp_sig"]["paw"]["v"]
        ref["onp"]["x"] = np.arange(len(ref["onp"]["y"])) / fs_onp

        # This ONP signal is sometimes noisy that we may want to filter it before using it for anything
        # ref["onp"]["y"] = util.lowpass_butter(ref["onp"]["y"], fs_onp, params.RR_max_bpm * 2.2 / 60, show=False)

        # The dataset's markings of breaths, algorithmically detected in the ONP signal. We observed
        # that these don't align with a timebase that starts at 0 s. The interval makes sense (0.04
        # s or 25 Hz) but there's a small offset that is consistent within a trial but varies
        # between trials. So, we round to match our interval, without explicitly adjusting for any
        # offset.
        
        ref["onp dataset breaths"]["x"] = data["ref"][trial]["breaths"]["t"]
        ref["onp dataset breaths"]["x"] = util.round_to_multiple(ref["onp dataset breaths"]["x"], 1/fs_onp, 2)
        ref["onp dataset breaths"]["y"] = util.subset(ref["onp dataset breaths"]["x"], ref["onp"]["x"], ref["onp"]["y"], fs_onp)

        # Calculation of RR from the dataset's markings of breaths
        diff = np.diff(ref["onp dataset breaths"]["x"])
        ref["rr onp dataset breaths raw"]["x"] = ref["onp dataset breaths"]["x"][1:][diff > 0]
        ref["rr onp dataset breaths raw"]["y"] = 60 / diff[diff > 0]

        # The dataset's RR, determined by the clinical monitor from IP. The rates are integers at,
        # strangely, slightly less than 1 Hz, and the first value may be slightly negative, like
        # -0.24 s. As with some other datasets, these values seem to be aligned to the end of the
        # window, so we shift them forward by half our window size (usually 15 s).
        #
        # Caution: This RR is not what the authors used as their reference. They calculated RR from
        # the ONP signal and excluded noisy periods, but they did not give us those RR values. They
        # used this IP-based RR as a comparator, I think.

        ref["rr ip monitor raw"]["x"] = data["ref"][trial]["params"]["rr"]["t"] - params.window_size / 2
        ref["rr ip monitor raw"]["y"] = data["ref"][trial]["params"]["rr"]["v"]

        # Heart rate values, determined by the clinical monitor from ECG. As with RR, the dataset
        # does not provide the authors' calculation of HR/PR, nor does it provide any pulse rate
        # from PPG).
        if 'hr' in data["ref"][trial]["params"]:
            ref["hr raw"]["x"] = data["ref"][trial]["params"]["hr"]["t"]
            ref["hr raw"]["y"] = data["ref"][trial]["params"]["hr"]["v"]
        else:
            # Trials 8, 14, and 20 lack HR data, unfortunately
            raise KeyError(f'load_ppg_data(): this trial has no HR reference data.')

        # Calculation of RR from our own detection of breaths, in IP and ONP
        # As with mimic, this is performing poorly, so we disable it.
        # for signal in ["ip", "onp"]:
        #     ref[f"rr {signal} NH raw"]["x"], ref[f"rr {signal} NH raw"]["y"], ref[f"{signal} NH breath starts"]["x"], ref[f"{signal} NH breath starts"]["y"] = rr_from_capnography(
        #         ref[signal]["y"], eval(f"fs_{signal}"), show=False
        #     )

        # Interpolate rates at 1 Hz
        for rate in ["rr onp dataset breaths", "rr ip monitor", "hr"]:
            ref[rate]["x"], ref[rate]["y"] = util.interpolate(
                ref[f"{rate} raw"]["x"],
                ref[f"{rate} raw"]["y"],
                kind="slinear",
            )

        # Metadata. Identical for every row: 
        # {'ventilation': 'spontaneous', 'recording_conditions': 'lying supine, at rest'} 
        #
        # Unfortunately, age is not available. From the paper, we know "between 18 and 40 years of
        # age" and "The median (lower, upper quartiles) age of analysed subjects was 29 (26, 32) years."
        #
        # No ID is provided besides the trial number.
        meta["treatment_ventilation"] = data["fix"][trial]["ventilation"]
        meta["recording_conditions"] = data["fix"][trial]["recording_conditions"]
        meta["age"] = "18-40"
        meta["id"] = trial

        return ppg_raw, fs_ppg, ref, meta

    elif dataset.lower() == "kapiolani":
        dataset_dir = pjoin(root_dir, "Kapiolani")
        
        ppg_raw, fs_ppg, ref = load_philips_monitor_data(
            root_dir=dataset_dir, 
            trial=trial,
            window_size=window_size,
            probe_num=probe_num,
            read_cpo=read_cpo,
            show=show
            )

        # ================== metadata information ==================
        patient_meta = pd.read_excel(
            pjoin(dataset_dir, "Patient Intake.xlsx"), sheet_name="Condensed"
        )
        patient_dirs = [
            d
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]
        this_pt_idx = patient_meta["Subject ID"] == int(sorted(patient_dirs)[trial])

        if ~np.any(this_pt_idx):
            # patient metadata not found in intake form
            meta = {}
            meta["treatment_ventilation"] = "n/a"
            meta["age"] = np.nan
            meta["id"] = int(sorted(patient_dirs)[trial])
            meta["probe model"] = "n/a"
            meta["notes"] = "n/a"
            meta["probe location"] = "n/a"
        else:
            this_patient_meta = patient_meta[this_pt_idx].iloc[0]
            meta = {}
            meta["treatment_ventilation"] = "spontaneous"
            meta["age"] = this_patient_meta["Age [months]"] / 12    # convert to years like the other datasets
            meta["id"] = this_patient_meta["Subject ID"]
            meta["probe model"] = this_patient_meta["Standard-of-Care probe: Model "]
            meta["notes"] = this_patient_meta["Notes"]
            if read_cpo:
                probe_location_key = f"CPO Probe f'{probe_num}' Location"
            else:
                probe_location_key = "Standard-of-Care probe Location"
            meta["probe location"] = this_patient_meta[probe_location_key]
        return ppg_raw, fs_ppg, ref, meta

    elif dataset.lower() == "3ps":

        if params.probe_type not in ["Tr", "Re"]:
            raise ValueError(f"load_ppg_data(): probe_num must be `Tr` (transmissive) or `Re` (reflective) but is `{params.probe_type}`.")
        if params.led_num not in [1, 2]:
            raise ValueError(f"load_ppg_data(): led_num must be 1 or 2 but is {params.led_num}.")

        dataset_dir = pjoin(root_dir, "3ps")
        dataset_primary_dir = pjoin(dataset_dir, "primary")
        
        patient_dirs = [
            d
            for d in sorted(os.listdir(dataset_primary_dir))
            if (os.path.isdir(os.path.join(dataset_primary_dir, d)) and d[0] == "N" and len(d) == 7)    # A trial looks like Nxx-yyy, e.g. N07-003
        ]
        this_pt_dir = pjoin(dataset_primary_dir, patient_dirs[trial])                   # A few data files live here
        sub_dirs = sorted([d for d in os.listdir(this_pt_dir) if os.path.isdir(pjoin(this_pt_dir, d))])
        subtrial_dir = sub_dirs[-1]              # Use the highest-numbered subtrial (ignore any earlier subtrials)
        # if subtrial_dir != '1':
        #     util.newprint(f"    {patient_dirs[trial]}: using subtrial {subtrial_dir}", on_a_new_line=True)
        this_pt_sensor_dir = pjoin(this_pt_dir, subtrial_dir, "per_sensor")     # The other data files live here

        # --- PPG probes
        if params.probe_type == "Tr":
            pleth_filename = "TrProbe.csv"
        elif params.probe_type == "Re":
            pleth_filename = "RefProbe.csv"
        pleth_df = pd.read_csv(pjoin(this_pt_sensor_dir, pleth_filename))

        # For all time sync, we use the first system_time (UTC UNIX time, added by the capturing PC)
        # as zero. The MCU t zero varies between signals.
        t_zero = pleth_df.at[0, "system_time"]
        duration = pleth_df["system_time"].iloc[-1] - t_zero
        fs_ppg = (len(pleth_df["system_time"]) - 1) / duration
        if abs(fs_ppg - 500) >= 2:
            warnings.warn(f"load_ppg_data(): For 3ps, fs_ppg is expected to be 500 Hz but was determined to be {fs_ppg:0.2f} Hz, a discrepancy over the {duration:0.1f} s trial of {duration * (500-fs_ppg)/500:0.2f} s.")
        fs_ppg = 500        # Since we've seen errors in system_time, it seems better to assume the nominal rate rather than use the reported actual

        # Trim to duration_limit
        if params.duration_limit:
            index_limit = params.duration_limit * fs_ppg
            if len(pleth_df) > index_limit:
                pleth_df = pleth_df[pleth_df.index < index_limit]

        # This PPG signal seems inverted, which we correct. And its DELTA does not properly handle
        # rollover. That only happens when the probe falls off or something else goes wrong, but
        # that disturbs some of our calculations for the entire sessions, so we calculate the delta
        # here, where rollover doesn't occur.
        ppg_raw = -(pleth_df[f"LED{params.led_num}VAL"] - pleth_df[f"ALED{params.led_num}VAL"])

        # Optionally check for extreme PPG values, though with our calculation of ppg_raw above,
        # this shouldn't matter anymore.
        if False:
            if max(ppg_raw) > params.ppg_max:
                util.newprint(f"    {patient_dirs[trial]}: max(ppg_raw) is {max(ppg_raw)}, exceeding params.ppg_max ({params.ppg_max})", on_a_new_line=True)
            if min(ppg_raw) < params.ppg_min:
                util.newprint(f"    {patient_dirs[trial]}: min(ppg_raw) is {min(ppg_raw)}, exceeding params.ppg_min ({params.ppg_min})", on_a_new_line=True)
        
        # Check for irregular timing. As with other datasets, we don't use the timestamp of each
        # row, but here we check to see if that's a reasonable assumption. For this dataset, we've
        # seen irregularities in system_time interval as large as 30 ms instead of the expected
        # 2 ms, but the microcontroller-generated t has never been off by more than 1 ms and that
        # has been rare.

        # duration_uc = (pleth_df["t"].iloc[-1] - pleth_df["t"].iloc[0])/1000
        # util.newprint(f"    {patient_dirs[trial]}: Pleth duration µC - duration PC ({duration:.3f}): {duration_uc - duration:.3f} s", on_a_new_line=True)

        # duration_pc_diff = np.diff(pleth_df["system_time"])
        # util.newprint(f"    {patient_dirs[trial]}: Pleth PC t diff: min {np.min(duration_pc_diff):.3f} s, max {np.max(duration_pc_diff):.3f} s, stdev {np.std(duration_pc_diff)}", on_a_new_line=True)

        # --- HR
        
        # Since this dataset has no HR reference, we calculate one from PPG to allow Nyquist checks
        # etc. During RR estimation, a more complex HR estimation is used, with PSD unless that
        # fails, in which case this same method is used.

        ref["hr"]["y"], ref["hr"]["x"] = heart_rate.estimate_instantaneous_heart_rate_with_resample(
            ppg_raw, 
            fs_ppg, 
            hr_min = params.hr_min_bpm
        )

        # --- Breath annotations: live at the clinic
        column_names = ["system_time", "device", "fw_version", "t", "key_down"]
        annot_live_df = pd.read_csv(pjoin(this_pt_sensor_dir, "a.txt"), names=column_names, header=None)
        annot_live_df["system_time"] -= t_zero
        
        # The live annotater may press the restart button. If so, ignore all earlier breath marks.
        annot_live_restarts_filename = pjoin(this_pt_dir, "annotation_restarts.txt")
        if os.path.exists(annot_live_restarts_filename):
            column_names = ["system_time"]
            annot_live_restarts_df = pd.read_csv(pjoin(this_pt_dir, "annotation_restarts.txt"), names=column_names, header=None)
            annot_live_restarts_df["system_time"] -= t_zero
            if len(annot_live_restarts_df) > 0:
                restart_time = annot_live_restarts_df["system_time"].iloc[-1]
                annot_live_df = annot_live_df[annot_live_df["system_time"] >= restart_time]

        # 0: unpressed. 1: pressed. Find rising edges. (We assume this signal was debounced, as it
        # appeared in an example file.)
        diff = np.diff(annot_live_df["key_down"])
        if len(diff) == 0:
            warnings.warn(f"load_ppg_data(): No breaths were annotated live; the button was not pressed.")
            ref["annot live"]["breaths"]["x"] = None
            ref["annot live"]["breaths"]["y"] = None
        else:
            ref["annot live"]["x"] = (annot_live_df["system_time"][1:][diff > 0]).to_numpy()
            # Since this dataset has no capnography-like reference signal, for y values we use the
            # running count of breaths.
            ref["annot live"]["y"] = np.arange(1, len(ref["annot live"]["x"]) + 1)

        # --- Breath annotations: afterwards by our panel of emergency physicians
        # 
        # These marks are recorded in Adobe Premiere Pro and exported as one CSV file per panelist.
        # Instantaneous marks signify breaths. 
        # 
        # Markers with duration > 0 are used to record that the panelist was not able to mark
        # breaths: 
        #
        #   - Markers named "uncertain" (case-insensitive and including anything "unsure" and
        #     anything else starting with "un") signify that just a portion was unmarkable. Often
        #     this is because the participant was moving too much.
        # 
        #   - Markers named anything else signify that the entire session was unmarkable. Often this
        #     is because of the reason written in the marker name: "focus" or "framing". We asked
        #     the panelists to be consistent in those names, but the field accepts any text.
        #
        # Unlike the other 3ps files, all breath CSVs are in one folder and contain the patient ID,
        # rater name, and pass number, such as `N04-010 Daniel 1.csv`. There may be one or several
        # passes.
        
        breath_dir = pjoin(dataset_dir, "breath marks")
        breath_fns = glob.glob(f"{patient_dirs[trial]}*.csv", root_dir=breath_dir)
        breath_fns.sort()
        if len(breath_fns) == 0:
            if "rr annot post" in params.reference_rr_signals:
                util.newprint(f"    {patient_dirs[trial]}: No CSV files of panel-annotated breaths were found.", on_a_new_line=True)
            # This will lead to an error if these panel-marked breaths are the reference
        else:
            breath_dfs = [pd.read_csv(pjoin(breath_dir, breath_fn)) for breath_fn in breath_fns]
            # For each breath-mark file
            for i, breath_df in enumerate(breath_dfs):
                # Parse the filename
                fn_stem, _ = splitext(breath_fns[i])
                panelist_initial = fn_stem.split()[1][0:1]
                if len(fn_stem.split()) == 2:                   # if no pass_num
                    pass_num = "1"                              # use a default
                else:
                    pass_num = " ".join(fn_stem.split()[2:])    # the rest of the stem
                ref_alias = ref["annot post"]["panelists"][panelist_initial]["passes"][pass_num]
                
                if panelist_initial == "D" and (patient_dirs[trial] in (f"N04-0{i}" for i in range(10,17))):
                    # For these trials, Dan's marks were recorded as the video played at 25 fps
                    # instead of the proper 30 fps, so we need to speed everything up.
                    fps_correction = 25/30

                    breath_df["In"]       *= fps_correction
                    breath_df["Out"]      *= fps_correction
                    breath_df["Duration"] *= fps_correction

                # Some panelists accidentally use the "Description" field instead of "Marker Name",
                # which we accommodate.
                breath_df["Marker Name"] = np.where(breath_df["Marker Name"].isna(), breath_df["Description"], breath_df["Marker Name"])

                # Breath markers
                filtered_breaths_df = breath_df[breath_df["Duration"] == 0]
                start_times = tuple(filtered_breaths_df["In"])
                ref_alias["breaths"]["x"] = start_times
                ref_alias["breaths"]["y"] = np.arange(1, len(start_times) + 1)

                # Any breath marks with a name
                named_filtered_breaths_df = filtered_breaths_df[filtered_breaths_df["Marker Name"].notna()]
                if len(named_filtered_breaths_df):
                    util.newprint(f"    {patient_dirs[trial]} {panelist_initial} {pass_num}: Named instantaneous breaths found:", on_a_new_line=True)
                    util.newprint(named_filtered_breaths_df[["Marker Name", "In"]].fillna("").to_string(index=False))

                if params.reference_rr_use_duration_markers:

                    # Any marks with duration (non-instantaneous)
                    periods_df = breath_df[breath_df["Duration"] > 0]

                    # We map anything starting with "un", such as "unsure", to "unceratain".
                    regex = re.compile(r"^un.*$", re.IGNORECASE)
                    pd.set_option('future.no_silent_downcasting', True)     # to prevent FutureWarning
                    periods_df = periods_df.replace(regex, "uncertain", inplace=False, regex=True)     # inplace=True caused a warning

                    periods_uncertain_df = periods_df[periods_df["Marker Name"] == "uncertain"]

                    # Print short uncertain markers for review
                    if False:
                        short_uncertain_limit = 5
                        periods_uncertain_short_df = periods_uncertain_df[periods_uncertain_df["Duration"] < short_uncertain_limit]
                        if len(periods_uncertain_short_df):
                            util.newprint(f"    {patient_dirs[trial]} {panelist_initial} {pass_num}: Uncertain markers shorter than {short_uncertain_limit} s found:", on_a_new_line=True)
                            util.newprint(periods_uncertain_short_df[["Marker Name", "In", "Out", "Duration"]].fillna("").to_string(index=False))

                    # We log the names of markers other than uncertain as "video problems" such as
                    # "focus" and "framing", and we treat these too as uncertain periods so that
                    # they can be excluded from use as a reference.
                    periods_other_df = periods_df[periods_df["Marker Name"] != "uncertain"]

                    if len(periods_other_df) == 0:
                        ref_alias["video problems"] = None
                    else:
                        periods_no_name_df = periods_other_df[periods_other_df["Marker Name"].isna()]
                        if len(periods_no_name_df):
                            util.newprint(f"    {patient_dirs[trial]} {panelist_initial} {pass_num}: Unnamed duration markers found:", on_a_new_line=True)
                            util.newprint(periods_no_name_df[["Description", "In", "Out"]].fillna("").to_string(index=False))
                            # Replace those NaNs to avoid errors right after this
                            periods_other_df.fillna({"Marker Name": "<missing>"}, inplace=True)
                        
                        ref_alias["video problems"] = sorted(list(periods_other_df["Marker Name"].str.strip().str.lower().unique()))
                        # This can be helpful to print, or it can be too much
                        if False:
                            util.newprint(f"    {patient_dirs[trial]} {panelist_initial} {pass_num}: Video problems: {', '.join(ref_alias['video problems'])}", on_a_new_line=True)

                    # In Batch 1 (the first 100 sessions), periods with other names, no matter how
                    # long, are used to exclude the entire session via
                    # reference_rr_expand_other_duration_markers.
                    #
                    # For stats, create a paired series that is 1 when uncertain and 0 when not,
                    # with a transition of params.uncertain_edge_offset before and after. Domain is
                    # 0 to the end of the trial, but with y values of NaN after the last marked
                    # breath (no extrapolation on that side). Create these even if there are no
                    # uncertain periods).
                    ref_alias["uncertain"]["state"]["x"] = [0]
                    ref_alias["uncertain"]["state"]["y"] = [0]

                    if len(periods_df) == 0:
                        # Certain for the whole time
                        ref_alias["uncertain"]["state"]["x"].append(ref_alias["breaths"]["x"][-1])
                        ref_alias["uncertain"]["state"]["y"].append(0)
                    else:
                        if (len(filtered_breaths_df) == 0) or (params.reference_rr_expand_other_duration_markers and len(periods_other_df)):
                            # Force the whole session to uncertain status
                            ref_alias["uncertain"]["x start"] = [0]
                            ref_alias["uncertain"]["x stop"]  = [params.duration_limit]
                        else: 
                            ref_alias["uncertain"]["x start"] = periods_uncertain_df["In"].tolist()
                            ref_alias["uncertain"]["x stop"]  = periods_uncertain_df["Out"].tolist()
                            if not params.reference_rr_expand_other_duration_markers and len(periods_other_df):
                                ref_alias["uncertain"]["x start"] += periods_other_df["In"].tolist()
                                ref_alias["uncertain"]["x stop"]  += periods_other_df["Out"].tolist()

                        # For plotting, create a paired series that can be laid under or over the breath-count series
                        # to show uncertainty.
                        ref_alias["uncertain"]["count"]["x"] = []
                        ref_alias["uncertain"]["count"]["y"] = []

                        for this_start, this_stop in zip(ref_alias["uncertain"]["x start"], ref_alias["uncertain"]["x stop"]):
                            # Build paired lists for this period and concatenate them onto the full-trial lists
                            
                            # *** Count
                            these_count_x = []        # the time
                            these_count_y = []        # the runnning breath count

                            # Create the count_x series: start, stop, and anything in between
                            these_count_x.append(this_start)
                            x_in_period = {x for x in ref_alias["breaths"]["x"] if x >= this_start and x <= this_stop}
                            x_in_period = sorted(x_in_period)
                            if len(x_in_period) > 0:
                                these_count_x.extend(x_in_period)
                            these_count_x.append(this_stop)

                            # Create the corresponding y values
                            if len(ref_alias["breaths"]["x"]):
                                these_count_y = list(np.interp(these_count_x, ref_alias["breaths"]["x"], ref_alias["breaths"]["y"]))
                            else:
                                these_count_y = [0] * len(these_count_x)

                            # After this period, add a None spacer so that Plotly will show a gap                        
                            these_count_x.append(None)
                            these_count_y.append(None)

                            ref_alias["uncertain"]["count"]["x"].extend(these_count_x)
                            ref_alias["uncertain"]["count"]["y"].extend(these_count_y)

                            # *** State: create a pulse for this period
                            ref_alias["uncertain"]["state"]["x"].extend([
                                max(0, this_start - params.uncertain_edge_offset),      # don't go negative
                                this_start,
                                this_stop,
                                this_stop + params.uncertain_edge_offset
                            ])
                            ref_alias["uncertain"]["state"]["y"].extend([0, 1, 1, 0])

                        # If the last uncertain period ends before the last marked breath, 
                        # end the "state" series at the last marked breath (rather than at the last
                        # uncertain period, which was added already).
                        if len(ref_alias["breaths"]["x"]) and ref_alias["uncertain"]["x stop"][-1] < ref_alias["breaths"]["x"][-1]:
                            ref_alias["uncertain"]["state"]["x"].append(ref_alias["breaths"]["x"][-1])
                            ref_alias["uncertain"]["state"]["y"].append(0)

                    # To facilitate averaging of the uncertain state series, interpolate at a fixed
                    # interval with no extrapolation (NaN outside the breath marks). Floating-point
                    # imprecision in the time values won't matter because we'll just assume that all 
                    # the series are synchronized.
                    step = 0.1
                    max_time = params.duration_limit
                    uninterp_state_x = ref_alias["uncertain"]["state"]["x"]
                    uninterp_state_y = ref_alias["uncertain"]["state"]["y"]
                    ref_alias["uncertain"]["state"]["x"] = np.arange(0, max_time + step, step)      
                    interp_f = scipy.interpolate.interp1d(uninterp_state_x, uninterp_state_y, kind="nearest", bounds_error=False, fill_value=np.nan, assume_sorted=True)
                    ref_alias["uncertain"]["state"]["y"] = interp_f(ref_alias["uncertain"]["state"]["x"])
                
                    # Having read all the breath mark files, we calculate averages for the uncertain state,
                    # intra and inter panelists (consistent with how we calculate RR disagreement later). We
                    # use a less-nested key since we don't have the various sub-types.

                    panelists_uncert_y = []     # one row per panelist
                    for panelist_k, panelist_v in ref["annot post"]["panelists"].items():

                        passes_uncert_y = []    # one row per pass
                        for pass_k in panelist_v["passes"].keys():
                            if len(passes_uncert_y) == 0:   # first loop
                                # Use the first (and perhaps only) pass for x values, since we assume all passes have the same x series. 
                                ref["annot post"]["panelists"][panelist_k]["uncertain"]["x"] = ref["annot post"]["panelists"][panelist_k]["passes"][pass_k]["uncertain"]["state"]["x"]
                            
                            # Collect each pass' y, for subsequent averaging
                            passes_uncert_y.append(ref["annot post"]["panelists"][panelist_k]["passes"][pass_k]["uncertain"]["state"]["y"])

                        passes_uncert_y = np.array(passes_uncert_y)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')     # suppress warning about mean of empty slice
                            ref["annot post"]["panelists"][panelist_k]["uncertain"]["y"] = np.nanmean(passes_uncert_y, axis=0)
                        
                        if len(panelists_uncert_y) == 0:   # first loop
                            # Use the first (and perhaps only) panelist for x values, since we assume all have the same x series. 
                            ref["annot post"]["uncertain"]["x"] = ref["annot post"]["panelists"][panelist_k]["passes"][pass_k]["uncertain"]["state"]["x"]

                        # Collect each panelist's y, for subsequent averaging
                        panelists_uncert_y.append(ref["annot post"]["panelists"][panelist_k]["uncertain"]["y"])

                    panelists_uncert_y = np.array(panelists_uncert_y)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')     # suppress warning about mean of empty slice
                        ref["annot post"]["uncertain"]["y"] = np.nanmean(panelists_uncert_y, axis=0)


        # --- RR reference calculation from breath annotations

        # We're no longer using valid_signal_found and could delete it
        # valid_signal_found = False
        
        # If no breaths were recorded, write a flag rather than no rate so that the trial can be processed
        rr_flag_for_no_breaths = -1

        # Interpolation settings
        interp_step = 0.01      # seconds

        # The following two blocks are similar and probably could be integrated

        # --- Block 1: annot live
        signal = "annot live"
        rate = f"rr {signal}"
        rate_raw = f"{rate} raw"

        if len(ref[signal]["x"]) > 1:
            # valid_signal_found = True

            # Calculate rate of each breath and assign it to the time of the end of that breath.
            # For our subsequent interpolation, assign to the first breath mark the rate of the second.
            diff = np.diff(ref[signal]["x"])
            diff = np.insert(diff, 0, diff[0])
            ref[rate_raw]["x"] = ref[signal]["x"]
            ref[rate_raw]["y"] = 60/diff
        else:   # no breaths found
            # Rather than omitting the rate, which would cause an error, write a rate before the
            # start of the trial
            ref[rate_raw]["x"] = [-1, -1 + interp_step * 2]
            ref[rate_raw]["y"] = [rr_flag_for_no_breaths, rr_flag_for_no_breaths]     

        # Interpolate with "next" per Berger 1986 so that this stepwise signal can be averaged accurately
        ref[rate]["x"], ref[rate]["y"] = util.interpolate(
            ref[rate_raw]["x"],
            ref[rate_raw]["y"],
            kind="next",
            step=interp_step
        )
        # Within the rate group, record the times of the events (the breath marks) so that
        # frame-wise calculations can restrict their averages to the time between breath marks,
        # instead of including partial breaths at the beginning and end.
        ref[rate]["x events"] = np.array(ref[rate_raw]["x"])            

        # --- Block 2: annot post

        signal = "annot post"
        rate = f"rr {signal}"
        rate_raw = f"{rate} raw"
        for panelist_k, panelist_v in ref["annot post"]["panelists"].items():
            for pass_k, pass_v in panelist_v["passes"].items():
                breaths_x = np.array(pass_v["breaths"]["x"])
                if len(breaths_x) > 1:
                    # valid_signal_found = True

                    # Calculate rate of each breath and assign it to the time of the end of that breath.
                    # For our subsequent interpolation, assign to the first breath mark the rate of the second.
                    diff = np.diff(breaths_x)
                    diff = np.insert(diff, 0, diff[0])
                    ref[rate_raw][panelist_k][pass_k]["x"] = breaths_x
                    ref[rate_raw][panelist_k][pass_k]["y"] = 60/diff

                    # Report any very short or long breaths so that the panelist can check for mistakes
                    short_breaths_idx = (diff < params.short_breath_threshold)
                    # Ignore the first diff value because we created it as a duplicate of the second
                    short_breaths_idx[0] = False
                    if sum(short_breaths_idx) > 0:      # if any short breaths found

                        if panelist_k == "D" and (patient_dirs[trial] in (f"N04-0{i}" for i in range(10,17))):
                            # For these trials, Dan's marks were recorded as the video played at 25 fps
                            # instead of the proper 30 fps, so we sped everything up. Now we need to
                            # convert back to his marks' native time so that he can find them in Premiere.
                            diff      /= fps_correction
                            breaths_x /= fps_correction
                            # util.newprint(f"                 (Because frame rate conversion was required for this file, these times may be slightly off.)", on_a_new_line=True)

                        check_for_short_breaths = False
                        if check_for_short_breaths:
                            for dur, t in zip(diff[short_breaths_idx], breaths_x[short_breaths_idx]):
                                minutes = int(t // 60)
                                seconds = int(t % 60)
                                frames  = int(int(round(t*30,0)) % 30)      # assuming 30 fps
                                dur_frames = int(round(dur*30,0))           # assuming 30 fps
                                util.newprint(f"    {patient_dirs[trial]} {panelist_k} {pass_k}: Breath ending at {minutes}:{seconds:02d}:{frames:02d} is only {dur_frames} {'frame' if dur_frames == 1 else 'frames'} [{dur:.2f} s] long", on_a_new_line=True)

                else:   # no breaths found
                    # Rather than omitting the rate, which would cause an error, write a rate before the
                    # start of the trial
                    ref[rate_raw][panelist_k][pass_k]["x"] = [-1, -1 + interp_step * 2]
                    ref[rate_raw][panelist_k][pass_k]["y"] = [rr_flag_for_no_breaths, rr_flag_for_no_breaths]     

                # Interpolate with "next" per Berger 1986 so that this stepwise signal can be averaged accurately
                ref[rate][panelist_k][pass_k]["x"], ref[rate][panelist_k][pass_k]["y"] = util.interpolate(
                    ref[rate_raw][panelist_k][pass_k]["x"],
                    ref[rate_raw][panelist_k][pass_k]["y"],
                    kind="next",
                    step=interp_step
                )
                # Within the rate group, record the times of the events (the breath marks) so that
                # frame-wise calculations can restrict their averages to the time between breath marks,
                # instead of including partial breaths at the beginning and end.
                ref[rate][panelist_k][pass_k]["x events"] = np.array(ref[rate_raw][panelist_k][pass_k]["x"])

        # if valid_signal_found == False:
        #     raise ValueError("load_ppg_data(): No reference RR could be determined for the entire trial because only 1 or 0 breaths were annotated.")
            
        # --- Accelerometers on the selected probe and the chest

        column_names = ["system_time", "device", "fw_version", "t", "significant_motion", "x", "y", "z"]

        for acc in ["acc probe", "acc chest"]:
            if acc == "acc probe":
                if params.probe_type == "Tr":
                    acc_filename = "ta1.txt"
                elif params.probe_type == "Re":
                    acc_filename = "ra1.txt"
                acc_df = pd.read_csv(pjoin(this_pt_sensor_dir, acc_filename), names=column_names, header=None)
            else:   # Chest probe
                # "ta2.txt" was used until reworked boards were installed in March 2025. Since then, "ra2.txt" is used.
                try:    
                    acc_filename = "ta2.txt"        
                    acc_df = pd.read_csv(pjoin(this_pt_sensor_dir, acc_filename), names=column_names, header=None)
                except:
                    acc_filename = "ra2.txt"        # on chest ("ta2.txt" was used until reworked boards were installed in March 2025)
                    acc_df = pd.read_csv(pjoin(this_pt_sensor_dir, acc_filename), names=column_names, header=None)

            # Check for gaps and NaN, which have happened
            acc_txyz_df = acc_df[["system_time", "x", "y", "z"]].copy()
            acc_prob_df = acc_txyz_df[acc_txyz_df.isna().any(axis=1)]   # only rows with NaN somewhere
            acc_prob_count = len(acc_prob_df)
            print_missing_acc = False
            if print_missing_acc and acc_prob_count > 0:
                acc_count = len(acc_df)
                util.newprint(f"    {patient_dirs[trial]}: Accelerometer file {acc_filename} contains {acc_prob_count} of {acc_count} rows ({acc_prob_count/acc_count:.1%}) with missing data. See `acc_prob_df`.", on_a_new_line=True)

            # These rows are unevenly spaced in time, so we need to use each row's timestamp. The
            # microcontroller's "t" is probably more reliable than the PC "system_time", which we use
            # only for cross-file sync. (Inspection of one trial, N09-026, showed "system_time" to be
            # as reliable as "t" within 1 ms, but "t" must be the safer choice.) "t" is in ms, which
            # we convert to our normal s.

            this_t_offset = acc_df["system_time"].iloc[0] - t_zero
            if this_t_offset > 0.5:
                util.newprint(f'    {patient_dirs[trial]}: {acc} time offset is large: {this_t_offset:.3f} s', on_a_new_line=True)
            this_t_zero = acc_df["t"].iloc[0]/1000 + (acc_df["system_time"].iloc[0] - t_zero) 
            ref[acc]["t"] = acc_df["t"]/1000 - this_t_zero
            sum_squares = 0 * len(ref[acc]["t"])
            for axis in ["x", "y", "z"]:
                # print(f"acc {axis} min, max: {min(acc_df[axis]):9d}, {max(acc_df[axis]):9d}")
                ref[acc][f"acc {axis}"] = acc_df[axis] / 16384   # normalize to what seems to be 1 g
                sum_squares += ref[acc][f"acc {axis}"] ** 2
            ref[acc]["acc magnitude"] = sum_squares ** (1/2)    # root of sum of squares

        # Camera times
        #
        # From reviewing the first ~100 trials with camera CSV files collected in late 2024 and
        # early 2025, before the formal study started, it wasn't clear how to properly synchronize
        # the vidoes with the pleth. The first row of Camera1.csv was as much as 123 ms later than
        # the first row of the pleth, but that Camera1's row-by-row intervals were strange, covering
        # only a few ms over the first four frames before finding a proper 33 ms rhythm (with
        # jitter). I also saw anomalies in that trial's pleth file, but they were at most 24 ms. So,
        # we report any large offsets but don't shift anything.
        #
        # Early in the real trial, we saw much larger offsets, up to 88 s on one video but little on
        # the other, yet the videos and accelerometers were well aligned, so we disabled this check.
        #
        # If we were to shift things, it would be simplest to use the video file as t_zero and align
        # the pleth and others to it. That way, the breath marks in Adobe Premiere wouldn't need to
        # be shifted.
        
        if False:
            try:
                cam1_df = pd.read_csv(pjoin(this_pt_dir, subtrial_dir, "Camera1.csv"))
                cam2_df = pd.read_csv(pjoin(this_pt_dir, subtrial_dir, "Camera2.csv"))
                cam1_t_offset = cam1_df.at[0, "system_time"] - t_zero
                cam2_t_offset = cam2_df.at[0, "system_time"] - t_zero
                if (abs(cam1_t_offset) > 0.1) or (abs(cam2_t_offset) > 0.1):
                    util.newprint(f"    {patient_dirs[trial]}: Time offset from video to pleth is large: Camera1.csv: {cam1_t_offset:.3f} s; Camera2.csv: {cam2_t_offset:.3f} s", on_a_new_line=True)
                
                # with open('/Users/wallace/Projects/Pulse_ROxi/git/pulseox_pubalgs/data/results/cam_times.csv', 'a') as fd:
                #     fd.write(f"{patient_dirs[trial]},{cam1_t_offset:.3f},{cam2_t_offset:.3f},{cam2_t_offset - cam1_t_offset:.3f}\n")
                # util.newprint(f"    {patient_dirs[trial]}: cam1_t_offset,cam2_t_offset,cam_diff", on_a_new_line=True)
                # util.newprint(f"Cam_times,{patient_dirs[trial]},{cam1_t_offset:.3f},{cam2_t_offset:.3f},{cam2_t_offset - cam1_t_offset:.3f}", on_a_new_line=True)
            except:
                pass

        # --- Patient metadata
        meta = {}

        meta["id"] = patient_dirs[trial]
        t_zero_dt = datetime.fromtimestamp(t_zero, tz=timezone.utc)
        t_zero_dt = t_zero_dt.astimezone(timezone(timedelta(hours=1)))            # Account for WAT, UTC +1
        meta["local datetime"] = t_zero_dt.strftime("%Y-%m-%d %H:%M")

        form_path = pjoin(this_pt_dir, "form.txt")
        meta_from_file = {}
        if os.path.exists(form_path):
            with open(form_path, "r") as file:
                for line in file:
                    key, value = line.strip().split(':')
                    if value != '':
                        meta_from_file[key] = value
                if "age" in meta_from_file.keys() and meta_from_file["age"].isdigit():
                    meta_from_file["age"] = round(int(meta_from_file["age"]) / 12, 2)   # convert from months to years

        # Individual typology angle (ITA), a measurement of skin tone by spectrophotometer
        ita_path = pjoin(dataset_dir, "spectrophotometer", f"{patient_dirs[trial]}.csv")
        if os.path.exists(ita_path):
            ita_df = pd.read_csv(ita_path)
            ita_raws = ita_df["ITA D65_10°"]
            ita_mean = round(np.mean(ita_raws),3)   # rounded to simplify display
            ita_sd = round(np.std(ita_raws),3)
        else:
            ita_mean = None
            ita_sd = None

        # The file's contents are randomly ordered. We standardize the order and revise some names,
        # and insert other data in our preferred order.
        meta["age"] =       meta_from_file.pop("age", "-")
        meta["weight"] =    meta_from_file.pop("weight", "-")
        meta["gender"] =    meta_from_file.pop("gender", "-")
        meta["ITA mean"] =  ita_mean
        meta["ITA SD"] =    ita_sd
        meta["complaint"] = meta_from_file.pop("complaint", "-")
        meta["temperature"] = meta_from_file.pop("temperature", "-")
        meta["CRF SpO2"] =  meta_from_file.pop("oxy", "-")
        meta["CRF HR"] =    meta_from_file.pop("heart_rate", "-")
        meta["CRF RR"] =    meta_from_file.pop("rr", "-")
        # Append any additional parameters
        for key, value in meta_from_file.items():
            meta[key] = value

        if params.probe_type == "Tr":
            meta["probe type"] = "transmissive"
        elif params.probe_type == "Re":
            meta["probe type"] = "reflective"
        meta["led num"] = params.led_num
        # meta["treatment_ventilation"] = "spontaneous"

        # Probe calibration parameters that are automatically set at the beginning of each trial.
        # The value should not change during the trial.
        def meta_probe(probe_name: str):
            meta[f"probe {probe_name}"]      = pleth_df[probe_name].iloc[0]
            if (pleth_df[probe_name].nunique() > 1):
                warnings.warn(f"load_ppg_data(): pleth_df['{probe_name}'].nunique() == {pleth_df[probe_name].nunique()} but should == 1")

        meta_probe("TIAGAIN")
        meta_probe("TIA_AMB_GAIN")
        meta_probe("LEDCNTRL")

        return ppg_raw, fs_ppg, ref, meta

    elif dataset == "kelseyRef":
        pass
    elif dataset.lower() == "mimic":
        """
        About the reference RR rate from the authors of this dataset:

        'The impedence pneumography (IP) waveform for each record was used as the
        reference recording for RR. Each breath in the IP signals was manually
        (independently) annotated by two research assistants, and both sets of
        annotations were used to derive the reference RR values. For each set of
        annotations, the RR value was determined based on the average time between
        consecutive breaths within a given window; only those windows of data for
        which the agreement between both estimates was within 2 breaths per min
        were retained, and the mean value of the two estimates was taken as the
        reference RR.'[1]

        This doesn't align with our methods. The authors seem to have assigned each RR value (a
        count) to the time at the end of the window, and I think they used a 32 s window for this 
        count (they did in general). We instead assign our RR value (a median) to the middle of 
        our window. When the inter-breath interval is changing rapidly, these results disagree
        considerably. I've seen a 3 rpm difference on a signal that was 16 rpm one way and 19 rpm by
        the other.

        Mysteriously, the authors' counted RR values go to the very beginning and end of the
        trial, and they do change near those ends (they aren't just repeats). Example: trial 3. Perhaps the authors 
        calculated RR on longer traces and trimmed them before publication. I think the same is true 
        of the HR/PR values, but I haven't looked closely.

        Also, on some trials (examples: 4, 13, 17, 32), the authors' RR values just don't make
        sense. They are far higher or lower than those we calculate from the authors' annotations of
        breaths. 

        Overall, I think it's better to ignore the authors' RR and instead calculate our own from
        the authors' annotations.

        [1] Pimentel et al.,  (2017). Toward a robust estimation of respiratory rate from
        pulse oximeters. IEEE Transactions on Biomedical Engineering, 64(8), 1914–1923.
        https://doi.org/10.1109/TBME.2016.2613124
        """
        dataset_dir = pjoin(root_dir, "mimic2")
        mat_file_path = pjoin(dataset_dir, f"bidmc_data.mat")
        data = pymatreader.read_mat(mat_file_path)["data"]  # read_matlab() doesn't work with this file, but this does

        # ppg waveform
        ppg_raw = data["ppg"][trial]["v"]
        fs_ppg = data["ppg"][trial]["fs"]

        # impedance pneumography signals
        fs_ip = data["ref"][trial]["resp_sig"]["imp"]["fs"]
        ref["ip"]["y"] = data["ref"][trial]["resp_sig"]["imp"]["v"]
        ref["ip"]["x"] = np.arange(len(ref["ip"]["y"])) / fs_ip

        # human annotations of breaths
        ref["ip annot1"]["x"] = np.unique(data["ref"][trial]["breaths"]["ann1"] / fs_ip)
        ref["ip annot1"]["y"] = util.subset(ref["ip annot1"]["x"], ref["ip"]["x"], ref["ip"]["y"], fs_ip)

        ref["ip annot2"]["x"] = np.unique(data["ref"][trial]["breaths"]["ann2"] / fs_ip)
        ref["ip annot2"]["y"] = util.subset(ref["ip annot2"]["x"], ref["ip"]["x"], ref["ip"]["y"], fs_ip)

        # Import of RR from the dataset: disabled because of the problems described above.
        # ref["rr dataset"]["x"] = data["ref"][trial]["params"]["rr"]["t"]
        # ref["rr dataset"]["y"] = data["ref"][trial]["params"]["rr"]["v"]

        # Calculation of RR from the dataset's human annotations of breaths
        # This older code could be better integrated with the above.
        ip_annot1 = data["ref"][trial]["breaths"]["ann1"]
        ip_annot2 = data["ref"][trial]["breaths"]["ann2"]
        diff1 = np.diff(ip_annot1) # avoid division by zero
        diff2 = np.diff(ip_annot2)
        ref["rr annot1 raw"]["x"] = ip_annot1[1:][diff1>0] / fs_ip
        ref["rr annot1 raw"]["y"] = 60 * fs_ip / diff1[diff1>0]
        ref["rr annot2 raw"]["x"] = ip_annot2[1:][diff2>0] / fs_ip
        ref["rr annot2 raw"]["y"] = 60 * fs_ip / diff2[diff2>0]

        # As the dataset's authors do, we mark breaths at minima, unlike with capnography where we
        # look for maxima, so we negate the argument.
        # But this is performing so poorly with the changing dynamic range that we are disabling it.
        """
        rr_impedance_t, rr_impedance, breaths_x, breaths_y = rr_from_capnography(
            co2=-ref["ip"]["y"], fs_co2=fs_ip, show=show
        )
        ref["rr ip raw"]["y"] = rr_impedance
        ref["rr ip raw"]["x"] = rr_impedance_t

        # and here we negate the IP values from the negated IP signal above
        ref["rr ref breath starts"]["y"] = -breaths_y
        ref["rr ref breath starts"]["x"] = breaths_x
        """
        
        # interpolate rr values at 1 Hz
        # for rr_signal in ["rr annot1", "rr annot2", "rr ip"]:
        for rr_signal in ["rr annot1", "rr annot2"]:
            interp_rr_t, interp_rr_y = util.interpolate(
                ref[f"{rr_signal} raw"]["x"],
                ref[f"{rr_signal} raw"]["y"],
                kind="slinear",
            )
            ref[rr_signal]["y"] = interp_rr_y
            ref[rr_signal]["x"] = interp_rr_t

        # pulse rate values (from PPG; we'll ignore the heart rate from EKG)
        ref["hr"]["x"] = data["ref"][trial]["params"]["pr"]["t"]
        ref["hr"]["y"] = data["ref"][trial]["params"]["pr"]["v"]

        # patient info, which isn't available in the .mat file
        with open(pjoin(dataset_dir, "bidmc_csv", f"bidmc_{trial+1:02}_Fix.txt"), "r") as file:
            lines = file.readlines()
            id = int(lines[0].rstrip()[-2:])    # the last two characters are the ID number
            ageline = lines[5].split(" ")
            age = ageline[1].rstrip()
            genderline = lines[6].split(" ")
            gender = genderline[1].rstrip()

        meta["treatment_ventilation"] = data["fix"][trial]["ventilation"]
        meta["age"] = age
        meta["gender"] = gender
        meta["id"] = id

        return ppg_raw, fs_ppg, ref, meta

def load_philips_monitor_data(
    root_dir: str,
    trial: int = None,
    folder_id: str = None,
    window_size: int = 30,
    probe_num: int = 1,
    read_cpo: bool = False,
    show: bool = False,
    ):
    """
    Read data obtained from the philips monitor, organized in the format used in the 
    Kapiolani study
    """
    # NOTE: PPG and CO2 time seem to be somehwat aligned, but monitor time lags by
        # a second.
    ref = defaultdict(dict)
    dataset_dir = root_dir
    
    if probe_num is None:
        raise ValueError("probe_num must be defined for the Kapi'olani dataset")

    # ============ set up path info to the specific subject ==============
    p_num = f"P{probe_num}"  # P1 or P2
    patient_dirs = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and (len(d) == 3) and d.isnumeric()    # ignore directories that aren't 3 digits
    ]
    patient_dirs.sort()             # otherwise, order may vary depending on the file system

    # user can pass in folder name or an index string to select the data to load from
    if folder_id is None:
        if trial >= len(patient_dirs):
            raise ValueError(f'load_philips_monitor_data(): trial ({trial}) >= len(patient_dirs) ({len(patient_dirs)})')
        this_pt_dir = pjoin(dataset_dir, patient_dirs[trial])
    else:
        this_pt_dir = pjoin(dataset_dir, folder_id)
        
    pleth_df_file_path = pjoin(this_pt_dir, p_num, "NOM_PLETHWaveExport.csv")
    if not os.path.exists(pleth_df_file_path):
        warnings.warn(f"pleth file does not exist: {pleth_df_file_path}")
        return None, None, None

    # ================== Read PPG data ==================
    # PPG and CO2 csv's have 4 unlabeled columns, we label them here.
    column_names = ["time", "relative time", "obs", "unk"]
    if read_cpo:
        # custom pulse ox probe
        cpo_file = glob.glob(pjoin(this_pt_dir, p_num, "CPO*"))
        pleth_df = pd.read_csv(cpo_file[0], names=column_names, header=None)

        ppg_raw = pleth_df["obs"]
        pleth_df["time"] = pd.to_datetime(pleth_df["time"], format="%H:%M:%S.%f")
        duration_seconds = (
            pleth_df["time"].iloc[-1] - pleth_df["time"].iloc[0]
        ).total_seconds()
        fs_ppg = round(len(pleth_df["time"]) / duration_seconds)
    else:
        # standard of care probe
        pleth_df = pd.read_csv(pleth_df_file_path, names=column_names, header=None)
        pleth_df["time"] = pd.to_datetime(
            pleth_df["time"], format="%d-%m-%Y %H:%M:%S.%f"
        )
        fs_ppg = 125

    # ================== load MPData (data from patient monitor) ==================
    # NOTE:
    # Philips patient monitor samples at 1 sample per second
    # 1. Must read P1 to get column names; P2 monitor file does not contain column
    # names.
    # 2. The "time" column has a strange format where the y/m/d is invalid.
    #    We opt to use relative time to align the signals instead of actual
    #    timestamps.
    # 3. Based on the "Time" column, tt seems like the fs is 1 second.
    pt_monitor = pd.read_csv(pjoin(this_pt_dir, "P1", "MPDataExport.csv"))
    raw_monitor_column_names = pt_monitor.columns
    if probe_num == 2:
        pt_monitor = pd.read_csv(
            pjoin(this_pt_dir, p_num, "MPDataExport.csv"),
            names=raw_monitor_column_names,
        )
    pt_monitor["RelativeTime"] = pd.to_numeric(
        pt_monitor["RelativeTime"], errors="coerce"
    )
    pt_monitor = pt_monitor.rename(
        columns={"Time": "time", "RelativeTime": "relative time"}
    )
    fs_monitor = 1

    # ============== reference capnography info ==============
    # note 1: Because the recording equipment only update the timestamp every 8
    # samples (or something like that), calculating fs based on length and time
    # stamp will result in a slightly inaccurate number. So we set fs to a constant
    # for higher accuracy.
    # note 2: The capnography timestamp and monitor reference timestamps are
    # slightly shifted, presumably due to smoothing or moving average operation.
    co2_file_path = pjoin(this_pt_dir, p_num, "NOM_AWAY_CO2WaveExport.csv")
    co2_df = pd.read_csv(co2_file_path, names=column_names, header=None)
    co2_df["time"] = pd.to_datetime(co2_df["time"], format="%d-%m-%Y %H:%M:%S.%f")
    fs_co2 = 62.5

    # ============== align times across sensors ==============
    aligned_dfs = align_time_axes(
        [co2_df, pleth_df, pt_monitor], [fs_co2, fs_ppg, fs_monitor]
    )

    # ============== organize output values  ==============
    # values may be skipped, need to coerce to numeric
    ppg_raw = aligned_dfs[1]["obs"].values
    ref["rr monitor"]["x"] = aligned_dfs[2]["x"].values
    ref["hr monitor"]["x"] = aligned_dfs[2]["x"].values
    ref["rr monitor"]["y"] = pd.to_numeric(
        aligned_dfs[2]["NOM_AWAY_RESP_RATE"].values, errors="coerce"
    )
    # pt_monitor["NOM_RESP_RATE"]
    ref["hr monitor"]["y"] = pd.to_numeric(
        aligned_dfs[2]["NOM_PLETH_PULS_RATE"].values, errors="coerce"
    )
    ref["hr"] = ref["hr monitor"]
    
    if ref["rr monitor"]["y"] is None:
        print("no RR from monitor?")
    #     MPRef = nan;
    #     warning('no NOM_RESP_RATE found');
    # else
    #     MPRef = MPData.('NOM_AWAY_RESP_RATE');
    #     HRRef = MPData.('NOM_PLETH_PULS_RATE');
    # end

    ref["co2"]["y"] = aligned_dfs[0]["obs"].values
    ref["co2"]["x"] = aligned_dfs[0]["x"].values
    fs_co2 = np.round(len(ref["co2"]["y"]) / ref["co2"]["x"][-1], 1)
    ref["co2"]["fs"] = fs_co2

    # Calculate RR from CO2 reference & interpolate to 1hz
    rr_capno_t, rr_capno, breaths_x, breaths_y = rr_from_capnography(
        co2=ref["co2"]["y"], fs_co2=fs_co2, show=False
    )
    if len(rr_capno) > 2:
        interp_rr_capno_t, interp_rr_capno = util.interpolate(
            rr_capno_t,
            rr_capno,
            kind="slinear",
        )
        breath_count_t, breath_count_y = rolling_window_count(
            times = rr_capno_t, window_length=window_size)
        ref["co2 breath counts"]["x"] = breath_count_t
        ref["co2 breath counts"]["y"] = breath_count_y
        ref["rr capno"]["x"] = interp_rr_capno_t
        ref["rr capno"]["y"] = interp_rr_capno
        ref["rr ref breath starts"]["x"] = breaths_x
        ref["rr ref breath starts"]["y"] = breaths_y
    else:
        ref["co2 breath counts"]["x"] = None
        ref["co2 breath counts"]["y"] = None
        ref["rr capno"]["x"] = None
        ref["rr capno"]["y"] = None
        ref["rr ref breath starts"]["x"] = None
        ref["rr ref breath starts"]["y"] = None

    if show:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.7, 0.3],  # relative heights of each row
        )
        fig.add_trace(
            go.Scatter(
                x=ref["co2"]["x"],
                y=util.normalize(ref["co2"]["y"]) * 75,
                name="Raw capnography trace<br>(arbitrary units)",
                mode="lines",
                line={"color": "rgba(106, 132, 219, 0.432)"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ref["rr capno"]["x"],
                y=ref["rr capno"]["y"],
                name="RR from Capnography",
                mode="lines",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ref["rr monitor"]["x"],
                y=ref["rr monitor"]["y"],
                name="RR from monitor",
                mode="lines",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=aligned_dfs[1]["x"],
                y=ppg_raw,
                name="Raw PPG",
                mode="lines",
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="Resp Rate (breaths per min)", row=1, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_xaxes(range=[min(ref["co2"]["x"]), max(ref["co2"]["x"])])    # Otherwise, Plotly pads both ends because the plot uses markers

        fig.show()

    return ppg_raw, fs_ppg, ref

def align_time_axes(df_list: list[pd.DataFrame], fs_list: list[float]):
    """Use the "relative time" axis to align recorded data.

    For Kapiolani, CO2, pleth, and patient monitor data are not perfectly aligned.
    Here we use the "relative time" column to align the three sets of data.
    We assume there is no break during signal acquisition and the time axis is
    continuous (which may not be necessarily true?).

    Args:
        df_list: A list of dataframes to align. Each must contain columns:
            * 'relative time': integer
            * 'time': pd.timestamp
            * 'obs': observed data
        fs_list: A list of sampling frequencies corresponding to each df.

    Returns:
        list of aligned dataframes.
    """
    target_col = "relative time"
    rel_timestamps = [df[target_col] for df in df_list]

    # largest starting & smallest ending timestamps
    new_t_start = max(
        [t_series[np.isfinite(t_series)].values[0] for t_series in rel_timestamps]
    )
    new_t_end = min(
        [t_series[np.isfinite(t_series)].values[-1] for t_series in rel_timestamps]
    )

    # Align time series by filtering
    aligned_dfs = []
    for df, fs in zip(df_list, fs_list, strict=True):
        aligned_df = df[df[target_col].between(new_t_start, new_t_end)]
        aligned_df = aligned_df.assign(x=np.arange(0, aligned_df.shape[0] / fs, 1 / fs))

        aligned_dfs.append(aligned_df)

    return aligned_dfs
