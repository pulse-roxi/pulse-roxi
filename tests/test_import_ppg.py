import numpy as np
import pandas as pd

from ppg2rr import import_ppg

ROOT = import_ppg.GIT_ROOT


def test_align_time_axes():
    """Makes sure kapiolani sensors are time-aligned before returning x- axis data.

    Tested using files from a single kapiolani patient.
    """
    # TODO: add test for custom probe

    # read files
    this_pt_dir = f"{ROOT}/tests/resources/import"

    column_names = ["time", "relative time", "obs", "unk"]
    co2_df = pd.read_csv(
        f"{this_pt_dir}/NOM_AWAY_CO2WaveExport.csv", names=column_names, header=None
    )
    ppg_df = pd.read_csv(
        f"{this_pt_dir}//NOM_PLETHWaveExport.csv", names=column_names, header=None
    )
    pt_monitor = pd.read_csv(f"{this_pt_dir}/MPDataExport.csv")
    pt_monitor.rename(
        columns={"Time": "time", "RelativeTime": "relative time"}, inplace=True
    )

    # format time columns
    co2_df["time"] = pd.to_datetime(co2_df["time"], format="%d-%m-%Y %H:%M:%S.%f")
    ppg_df["time"] = pd.to_datetime(ppg_df["time"], format="%d-%m-%Y %H:%M:%S.%f")
    pt_monitor["time"] = pd.to_datetime(
        pt_monitor["time"].str.split(" ").str[1], format="%H:%M:%S"
    )

    # align
    fs_list = [62.5, 125, 1]
    sensor_dfs = [co2_df, ppg_df, pt_monitor]
    aligned_dfs = import_ppg.align_time_axes(sensor_dfs, fs_list)

    time_ends = [df["relative time"].iloc[-1] for df in aligned_dfs]
    time_starts = [df["relative time"].iloc[0] for df in aligned_dfs]
    assert sum(np.diff(np.array(time_ends))) == 0
    assert sum(np.diff(np.array(time_starts))) == 0
