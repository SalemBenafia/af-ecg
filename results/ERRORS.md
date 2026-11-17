# AF Detect: subgroups and failure cases (test split, after the freeze)

This is a description of the frozen gbm's errors on the test split. Nothing was retrained or changed because of it.
Sources: `results/test/predictions.csv` (gbm predictions, made once after tag `v1-frozen`) joined with
`data/features/cinc2017.csv`; produced by `scripts/errors.py`. Numbers are in `results/errors/subgroups.csv`.

## Subgroups

95% CIs are percentile bootstrap intervals over recordings (2,000 resamples, seed 0). **Groups with fewer than 100
recordings or fewer than 20 AF recordings are descriptive only**: their CIs are wide and they must not be read as
proven differences. The quality terciles are computed on the test recordings themselves.

| By | Group | n | n AF | Challenge F1 (95% CI) | AF sensitivity (95% CI) | Note |
|---|---|---|---|---|---|---|
| length | < 20 s | 92 | 12 | 0.719 (0.589-0.821) | 0.667 (0.384-0.923) | descriptive only (small group) |
| length | 20-40 s | 1022 | 82 | 0.785 (0.754-0.813) | 0.890 (0.817-0.955) |  |
| length | >= 40 s | 166 | 17 | 0.781 (0.696-0.848) | 0.765 (0.538-0.941) | descriptive only (small group) |
| heart_rate | < 60 bpm | 194 | 7 | 0.743 (0.622-0.820) | 0.714 (0.333-1.000) | descriptive only (small group) |
| heart_rate | 60-100 bpm | 1009 | 72 | 0.764 (0.728-0.798) | 0.833 (0.741-0.915) |  |
| heart_rate | >= 100 bpm | 74 | 32 | 0.598 (0.548-0.640) | 0.906 (0.794-1.000) | descriptive only (small group) |
| heart_rate | unknown (NaN) | 3 | 0 | 0.222 (0.000-0.333) | n/a (n/a-n/a) | descriptive only (small group) |
| quality | low (< 0.724) | 427 | 36 | 0.778 (0.725-0.826) | 0.806 (0.667-0.930) |  |
| quality | middle (0.724-0.8) | 426 | 34 | 0.764 (0.709-0.808) | 0.824 (0.686-0.938) |  |
| quality | high (> 0.8) | 427 | 41 | 0.794 (0.749-0.832) | 0.902 (0.804-0.978) |  |

What the table suggests (descriptive, not tested):
- Short recordings (< 20 s, 92 recordings, only 12 AF) have the lowest AF sensitivity (0.667, CI 0.384-0.923); both
  "AF missed" examples below are shorter than 12 s. Too few AF recordings to say more.
- Fast rhythms (>= 100 bpm, 74 recordings) have a clearly lower challenge F1 (0.598) while AF sensitivity stays high
  (0.906, 32 AF): the problems at high heart rate are mostly Normal/Other/Noisy confusions, not missed AF.
- AF sensitivity rises from 0.806 (low quality) to 0.902 (high quality), but the CIs overlap; challenge F1 is
  similar in all three quality groups.
- 3 recordings have no heart rate (too few usable beats); with 3 recordings this group says nothing.

## Six misclassified test recordings

Picked deterministically, not by hand: for each error type the 2 recordings where the model was most wrong
(lowest p(AF) for missed AF, highest p(AF) for false alarms). Each plot shows the first 15 s of the cleaned ECG
with the detected R peaks (red) and the RR intervals of the whole recording. The causes below are my reading of
the plots, not a cardiologist's review.

| Plot | Error | Likely cause (one sentence) |
|---|---|---|
| [1](errors/1_af_missed_A01056.png) | AF → Normal (A01056, 11.1 s, p(AF) 0.02) | The beats are detected correctly but the RR intervals change smoothly and slowly over only 16 intervals, so in this short recording there is no beat-to-beat irregularity for the model to see (either AF with an unusually regular response or a disputed label). |
| [2](errors/2_af_missed_A00648.png) | AF → Too noisy (A00648, 11.6 s, p(AF) 0.05) | Strong baseline wander and a large artefact near 11 s put several R-peak marks in the wrong place (low quality score 0.51), so the quality features dominate and "too noisy" is an understandable, if wrong, answer. |
| [3](errors/3_false_af_alarm_A05639.png) | Other → AF (A05639, p(AF) 0.97) | Clean signal, but frequent premature beats (the deep, wider beats near 8.2 s and 14.3 s) and changing RR intervals make the rhythm look randomly irregular, which the RR features cannot tell apart from AF. |
| [4](errors/4_false_af_alarm_A05696.png) | Other → AF (A05696, p(AF) 0.95) | The RR intervals swing widely (roughly 0.5-1.3 s) in waves, and a large movement artefact around 8.5-9.5 s plus small QRS complexes make the P-wave evidence weak, so the irregularity is read as AF. |
| [5](errors/5_af_called_other_A01325.png) | AF → Other (A01325, p(AF) 0.02) | The rhythm is fast and almost perfectly regular with fast, regular atrial waves between the beats, which looks more like atrial flutter than AF; two missed beats near an artefact at about 13 s add long RR intervals that fit "Other". |
| [6](errors/6_af_called_other_A03979.png) | AF → Other (A03979, p(AF) 0.02) | The beats are detected correctly and the rhythm is regular except for two pauses of about 1.2 s, a pattern that looks like a regular rhythm with dropped beats rather than AF, so either the AF is not visible in this recording or the label is disputed. |

Common themes: (a) AF that is not irregular in the recording (plots 1, 5, 6) is called Normal or Other, because the
RR features carry most of the AF evidence; (b) irregular non-AF rhythms with premature beats (plots 3, 4) raise false
alarms; (c) noise and baseline wander break R-peak detection (plot 2), the risk already seen in P3. These match the
limitations in docs/SPEC.md section 11 (AF vs Other confusion, label noise) and are reported, not fixed.
