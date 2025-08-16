# Overview
For new developers, I recommend reading the notebooks in the following order
1. Get to know the existing dataset with `kapiolani_dataset_eda.ipynb`. This notebook outlines the distributions of heart rate (HR), resp rate (RR), age, and PPG quality of the data.
2. Get to know the differences across PPG waveforms in Capnobase vs Kapiolani datasets by inspecting the data using `evaluate_single_subject.ipynb`. Notice:
   1. The PPG quality in Capnobase is generally high, while the PPG quality in Kapiolani can be very low.
   2. PPG waveform experiences clipping and is gain-adjusted rapidly after clipping occurs. This introduces distortions that hides the RR frequency.
3. The second part of `evaluate_single_subject.ipynb` evaluates single 30-second time frames, and shows information about each processing step in the algorithm.
4. `evaluate_dataset_posthoc.ipynb` inspects the results from `evaluate_dataset.ipynb`.
   1. The first section is meant to experiment with different fusion methods for combining/down-selecting different RR estimates.
   2. In `Performance Summary vs Quality`, we see the tabulated results quantifying the performance of each RR estimate by calculating the Bias, LoA, r2, pct in tolerance, and n (number of datapoints included). See the [wiki](https://github.com/new-horizons/pulseox_pubalgs/wiki) for more information about these metrics.
   3. `Bias and LoA vs Quality` computes a plot to quantify algorithm performance vs PPG quality. On the x-axis, datapoints with quality smaller than x are excluded, and the remaining datapoints are used to calculate performance. The purpose is to evaluate how well quality correlates with confidence in algorithm performance.
   4. `error vs quality` plots percent-error vs PPG quality. It's another way to evaluate the correlation of  PPG quality with performance. Our results so far shows that the upper bound of error is highly correlated with PPG quality.
   5. We can also inspect the performance per specific trials under `Subject-Wise metrics`. This helps us identify trials with poor performance, so we can inspect these troublesome PPG examples and maybe further refine the algorithm using these examples.
5. The algorithm generates multiple sets of RR estimates. To see the performance distribution for each estimate, look at the examples in `riv_error_distributions.ipynb`. I expect we will need to significantly simplify the algorithm in the future. These distributions can help us identify RIVs that may not be contributing to the algorithm.