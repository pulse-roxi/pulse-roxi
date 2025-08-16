"""Generate RR estimates with estimate_rr() or estimate_rr_dataset()."""

from ppg2rr.config import AlgorithmParams
from ppg2rr.rr_est import estimate_rr

# This will cause all warnings to be treated as exceptions. Set to 'default' to disable
# warnings.filterwarnings('error')

# dataset='kapiolani'
# params = AlgorithmParams(
#     dataset=dataset,
#     probe=1,
#     remove_riv_outliers="segment-wise"
#     )
# trials = [4]
# df, rr_candidates, quality_indices = estimate_rr_dataset(
#     dataset=dataset,
#     trials=trials,
#     params=params,
#     save_df=True,
#     show=True,
#     save_fig=False,
#     file_suffix='test',
# )

dataset = "kapiolani"
trial = 12  # integer number to specify which trial to run
frame_num = None # If None, process all frames; if list of integers, process specified frames
params = AlgorithmParams(
    dataset=dataset,
    probe=1,
    remove_riv_outliers="segment-wise",
    peak_counting_prominence=0.5,
    window_size=30,
)

(
    frame_data,
    rr_candidate_merged,
    all_rr_candidates_list,
    feature_quality_list,
    meta,
) = estimate_rr(
    trial_num=trial,
    frame_num=frame_num,
    params=params,
    dataset=dataset,
    show=True,
    save_fig=False,
    rr_seed=20,
)

print("done")
