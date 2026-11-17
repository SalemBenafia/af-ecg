# Results on test (n = 1280)

| Model | AF sensitivity (95% CI) | AF specificity (95% CI) | AF PPV | AF AUROC | Challenge F1 (95% CI) | F1 N | F1 A | F1 O | F1 ~ |
|---|---|---|---|---|---|---|---|---|---|
| rr_rule | 0.820 (0.745-0.888) | 0.713 (0.688-0.738) | 0.213 | 0.766 | 0.378 (0.358-0.396) | 0.796 | 0.338 | 0.000 | 0.000 |
| logreg_rr | 0.802 (0.724-0.873) | 0.941 (0.926-0.955) | 0.563 | 0.944 | 0.640 (0.608-0.671) | 0.767 | 0.662 | 0.490 | 0.368 |
| gbm | 0.847 (0.773-0.910) | 0.979 (0.970-0.986) | 0.790 | 0.971 | 0.780 (0.753-0.805) | 0.851 | 0.817 | 0.671 | 0.576 |

gbm with abstain threshold 0.50 (chosen on VAL): keeps 92.7% (1187 of 1280); on the kept ones challenge F1 0.792, AF sensitivity 0.855, AF specificity 0.980.
