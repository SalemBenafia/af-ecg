# gbm ablation on VAL (fit on TRAIN n = 5968, scored on VAL n = 1280)

Seeds: [0, 1, 2]. Values are mean ± SD over seeds. The configured model is *all features, balanced, 0.05*. The test split was not used.

| Feature set | # features | class_weight | learning_rate | Challenge F1 | AF sensitivity | AF specificity (mean) |
|---|---|---|---|---|---|---|
| all features | 21 | None | 0.05 | 0.783 ± 0.007 | 0.760 ± 0.019 | 0.981 |
| all features | 21 | None | 0.1 | 0.780 ± 0.018 | 0.751 ± 0.036 | 0.981 |
| all features | 21 | balanced | 0.1 | 0.773 ± 0.006 | 0.802 ± 0.018 | 0.973 |
| all features | 21 | balanced | 0.05 | 0.770 ± 0.003 | 0.808 ± 0.014 | 0.972 |
| RR + P wave + f-wave | 13 | None | 0.05 | 0.752 ± 0.008 | 0.742 ± 0.028 | 0.979 |
| RR + P wave + f-wave | 13 | None | 0.1 | 0.750 ± 0.013 | 0.739 ± 0.032 | 0.979 |
| RR + P wave | 12 | None | 0.05 | 0.746 ± 0.003 | 0.739 ± 0.009 | 0.978 |
| RR + P wave | 12 | None | 0.1 | 0.744 ± 0.009 | 0.721 ± 0.018 | 0.979 |
| RR + P wave | 12 | balanced | 0.1 | 0.738 ± 0.001 | 0.823 ± 0.010 | 0.965 |
| RR + P wave + f-wave | 13 | balanced | 0.05 | 0.734 ± 0.009 | 0.814 ± 0.005 | 0.963 |
| RR + P wave + f-wave | 13 | balanced | 0.1 | 0.734 ± 0.005 | 0.799 ± 0.005 | 0.966 |
| RR only | 11 | None | 0.05 | 0.732 ± 0.001 | 0.685 ± 0.000 | 0.977 |
| RR + P wave | 12 | balanced | 0.05 | 0.730 ± 0.008 | 0.817 ± 0.005 | 0.965 |
| RR only | 11 | None | 0.1 | 0.727 ± 0.004 | 0.673 ± 0.010 | 0.976 |
| RR only | 11 | balanced | 0.1 | 0.715 ± 0.004 | 0.736 ± 0.005 | 0.962 |
| RR only | 11 | balanced | 0.05 | 0.702 ± 0.007 | 0.730 ± 0.018 | 0.958 |

## Permutation importance (best configuration: all features, class_weight None, learning_rate 0.05; seed 0, 10 repeats on VAL)

Drop in VAL challenge F1 when one feature is shuffled. Correlated features share credit, so a low value does not mean the feature is useless.

| Feature | Mean drop | SD |
|---|---|---|
| hr_mean | 0.1330 | 0.0117 |
| pnn50 | 0.0873 | 0.0124 |
| sampen | 0.0551 | 0.0069 |
| tmpl_corr | 0.0490 | 0.0087 |
| p_wave_ratio | 0.0474 | 0.0071 |
| rmssd_norm | 0.0460 | 0.0066 |
| sd1_sd2 | 0.0389 | 0.0068 |
| rr_mad_norm | 0.0168 | 0.0017 |
| rr_cv | 0.0094 | 0.0034 |
| tpr | 0.0094 | 0.0028 |
| kurtosis | 0.0072 | 0.0063 |
| hf_noise_ratio | 0.0069 | 0.0042 |
| rr_entropy | 0.0062 | 0.0043 |
| fwave_ratio | 0.0044 | 0.0050 |
| n_beats | 0.0035 | 0.0025 |
| qrs_amp | 0.0017 | 0.0035 |
| frac_rr_implausible | 0.0007 | 0.0004 |
| nk_quality | 0.0007 | 0.0013 |
| amp_range | 0.0006 | 0.0031 |
| duration_s | -0.0002 | 0.0028 |
| lf_ratio | -0.0008 | 0.0024 |

## Recommendation

Keep all 21 features: in every class_weight/learning_rate setting they beat the best reduced set by at least 0.03 challenge F1, and duration_s (the possible "short = noisy" shortcut) has about zero permutation importance. class_weight None raises challenge F1 by about 0.013 (0.783 vs 0.770) but lowers AF sensitivity by about 0.05 (0.760 vs 0.808), and for a screening aid a missed AF is worse than an extra clinician review, so keep "balanced". learning_rate 0.1 vs 0.05 differs by less than one seed SD (early stopping ends after 28-83 iterations), so keep 0.05: the current configs/train.yaml needs no change, and because all these gaps are smaller than the VAL bootstrap CI of the gbm (0.738-0.796, results/val/summary.json) they are tendencies, not proven differences.
