# The 21 features, in plain words

Every recording is resampled to 300 Hz, cleaned (NeuroKit, method "neurokit") and its heartbeats (R peaks)
are detected. The model then sees only these 21 numbers. Exact definitions: docs/SPEC.md section 6;
code: src/afdetect/features.py.

Rules that apply to all features:
- Gaps between beats (RR intervals) shorter than 0.25 s or longer than 2.5 s are treated as detection errors
  and left out. With fewer than 5 usable gaps, the rhythm features are left empty (NaN).
- The "typical beat" (template) is the median of all complete beats, each cut from 0.25 s before to 0.45 s
  after its R peak. With fewer than 3 complete beats, the beat-shape, P-wave and f-wave features are left empty.
- An empty value (NaN) means "could not be measured". The recording is never dropped; the model handles
  empty values.

## Rhythm (RR intervals)
| Feature | In one sentence | In AF this tends to be |
|---|---|---|
| n_beats | How many heartbeats were detected in the recording. | |
| frac_rr_implausible | The share of gaps between beats that are too short or too long to be real, a sign of missed or extra beats. | |
| hr_mean | The average heart rate in beats per minute. | |
| rr_cv | How much the gaps between beats vary overall, relative to their average length. | high |
| rmssd_norm | How much each gap between beats differs from the one before it, relative to the average gap. | high |
| pnn50 | The share of consecutive gaps that differ by more than 50 milliseconds. | high |
| rr_mad_norm | A robust version of rhythm variability that ignores a few extreme gaps. | high |
| sd1_sd2 | The shape of the cloud when each gap is plotted against the next one: close to 1 means a round, random-looking cloud. | high |
| tpr | How often the gap length changes direction (up then down, or down then up); a random sequence does this about two thirds of the time. | near 2/3 |
| rr_entropy | How evenly the gap lengths spread over 16 bins between the shortest and longest gap, on a scale from 0 to 1. | high (Other: lower) |
| sampen | How unpredictable the sequence of gaps is: higher means repeating patterns are rarer (capped at 3.0 when no pattern repeats at all). | high |

## P wave
| Feature | In one sentence | In AF this tends to be |
|---|---|---|
| p_wave_ratio | The size of the small bump just before each beat (the P wave, from the upper heart chambers), compared with the size of the main spike. | low |

## Beat shape
| Feature | In one sentence | In AF this tends to be |
|---|---|---|
| tmpl_corr | How closely each beat matches the typical beat (1 = identical shapes). | |
| qrs_amp | The height of the main spike (QRS complex) of the typical beat, in millivolts. | |

## Atrial activity
| Feature | In one sentence | In AF this tends to be |
|---|---|---|
| fwave_ratio | After removing the typical beat from every heartbeat, the share of the remaining signal power in the 4-9 Hz band, where the fast "fibrillatory" waves of AF show up. | high |

## Signal quality
| Feature | In one sentence | In AF this tends to be |
|---|---|---|
| duration_s | The length of the recording in seconds. | |
| kurtosis | How "spiky" the cleaned signal is compared with a smooth bell-shaped distribution of values. | |
| nk_quality | NeuroKit's average beat-quality score (higher = beats look more like a clean ECG). | |
| hf_noise_ratio | The share of the raw signal's power above 40 Hz, mostly muscle and electrical noise. | |
| lf_ratio | The share of the raw signal's power below 0.7 Hz, mostly slow baseline drift from breathing or movement. | |
| amp_range | The spread of the cleaned signal between its 1st and 99th percentile, in millivolts (very large values suggest movement artefacts). | |

The "In AF this tends to be" column repeats the expectations from the spec. It is not a measured result; measured
differences between classes come from results/ files only.
