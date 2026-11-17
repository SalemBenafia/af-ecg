# Results on afdb (n = 2466, 23 source records)

Main numbers: AF sensitivity and specificity. AFDB "(N" means all non-AF rhythms, not "normal", so the 4-class challenge F1 is shown for completeness only. CIs resample source records.

| Model | AF sensitivity (95% CI) | AF specificity (95% CI) | AF PPV | AF AUROC | Challenge F1 (95% CI) | F1 N | F1 A | F1 O | F1 ~ |
|---|---|---|---|---|---|---|---|---|---|
| rr_rule | 0.889 (0.840-0.934) | 0.786 (0.701-0.869) | 0.765 | 0.838 | 0.559 (0.528-0.589) | 0.856 | 0.823 | 0.000 | 0.000 |
| logreg_rr | 0.820 (0.700-0.927) | 0.944 (0.888-0.982) | 0.919 | 0.955 | 0.578 (0.518-0.624) | 0.759 | 0.867 | 0.107 | 0.000 |
| gbm | 0.825 (0.708-0.932) | 0.963 (0.936-0.987) | 0.946 | 0.967 | 0.639 (0.551-0.708) | 0.754 | 0.881 | 0.281 | 0.000 |

gbm with abstain threshold 0.50 (chosen on VAL): keeps 94.0% (2317 of 2466); on the kept ones challenge F1 0.655, AF sensitivity 0.856, AF specificity 0.975.
