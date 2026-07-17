# Chapter IV Draft - Results and Discussion

> **Data provenance.** Every primary value in this draft is generated from the full-data artifacts in `results/metrics`. Quick-run artifacts are explicitly rejected by `src/reporting.py` and must not be reported as thesis results.

## 4.1 Experimental Data and Preprocessing

The raw dataset contained 5,351,760 rows. Preprocessing removed 1,177 rows with invalid numeric values, leaving 5,350,583 valid observations and 69 model features. The stratified 80:20 split produced 4,280,466 training rows and 1,070,117 test rows. The maximum class-percentage difference between the two splits was 0.000047 percentage points.

The scaler was fitted only on training rows (`scaler_fit_scope = train_only`) and then applied unchanged to both splits. Resampling and class weighting were applied only to training artifacts, while the classifier integrity report verified that test artifacts remained unchanged. These controls support H1 methodologically: they demonstrate the intended ordering and reduce leakage risk, but they do not claim that leakage prevention is itself an empirical performance gain.

See [Table 4.1](tables/table_4_1_dataset_pipeline.csv).

## 4.2 Autoencoder Training and Latent Representation

The Autoencoder reduced 69 normalized features to 16 latent features, a 76.81% dimension reduction. The selected checkpoint was epoch 48 with validation MSE 0.00010914. Full-array reconstruction MSE was 0.00011259 on train and 0.00011357 on test, a relative gap of 0.87%. The small gap indicates no coarse reconstruction overfitting.

![Autoencoder loss](figures/figure_4_1_autoencoder_loss.png)

## 4.3 Imbalance-Handling Scenario Results

| Scenario | Accuracy | Macro precision | Macro recall | Macro F1 | Training seconds |
|---|---|---|---|---|---|
| S1 - No handling | 0.8436 | 0.3204 | 0.3376 | 0.3142 | 104.9755 |
| S2 - Class weight | 0.7285 | 0.4106 | 0.5620 | 0.4249 | 104.9128 |
| S3 - Upsampling | 0.7278 | 0.4103 | 0.5580 | 0.4233 | 532.1799 |
| S4 - Downsampling | 0.6155 | 0.3050 | 0.4900 | 0.2923 | 1.6365 |

S1 achieved the highest accuracy (0.8436) but only 0.3142 macro F1. S2 achieved the best macro recall (0.5620) and macro F1 (0.4249). S3 was nearly equivalent to S2, differing by only 0.0017 macro F1 while requiring 5.07 times the classifier training time. This similarity is consistent with random duplication and class weighting producing equivalent relative class contributions for tree training. S4 produced the lowest aggregate macro F1 (0.2923) after discarding nearly all majority-class training rows.

![Scenario metrics](figures/figure_4_2_scenario_metrics.png)

## 4.4 Per-Class Effects and Confusion Patterns

The imbalance strategies improved several minority classes but did not improve every class. Backdoor and Ransomware benefited strongly under S2/S3, while DoS and DDoS remained unreliable in all four scenarios. The complete focus-class values are provided in [Table 4.4](tables/table_4_4_minority_class_metrics.csv).

The trade-off was substantial for XSS: recall fell from 0.9241 under S1 to 0.4898 under S2. Therefore, S2 is the preferred scenario under the pre-specified macro-metric objective, not a universal winner for every class or operating requirement.

![Minority F1](figures/figure_4_3_minority_f1.png)

![S2 confusion matrix](figures/figure_4_4_s2_confusion_matrix.png)

## 4.5 Optional Original-Feature Baseline and H2

| Representation | Scenario | Accuracy | Macro recall | Macro F1 | Classifier seconds |
|---|---|---|---|---|---|
| latent_16 | s1_none | 0.8436 | 0.3376 | 0.3142 | 104.9755 |
| latent_16 | s2_class_weight | 0.7285 | 0.5620 | 0.4249 | 104.9128 |
| original_69 | s1_none | 0.8531 | 0.3401 | 0.3373 | 243.9785 |
| original_69 | s2_class_weight | 0.7881 | 0.6483 | 0.5406 | 208.4351 |

Under S2, the original 69-feature baseline improved macro F1 by 0.1157 (0.4249 to 0.5406) and improved macro recall from 0.5620 to 0.6483. In contrast, the latent representation reduced LightGBM training time by 49.7% for S2. This timing comparison covers the classifier only and must not be interpreted as an end-to-end runtime advantage because Autoencoder training is an additional cost.

H2 is therefore partially supported. The Autoencoder clearly provides a compact representation from which LightGBM can distinguish several classes, but the controlled baseline shows that the 16-dimensional representation does not preserve all useful discriminative information. It should not be claimed that the Autoencoder improves classification performance over the original features.

For DoS and DDoS, the original-feature baseline did not improve the best F1 for either class. This supports the diagnostic conclusion that their failure is not primarily an Autoencoder artifact: The original 69 features improve S2 macro F1 from 0.4249 to 0.5406, but they do not improve the best attainable F1-score for either DoS or DDoS. Their mutual confusion is therefore more consistent with intrinsic flow similarity and very small class supports than with Autoencoder information loss as the primary cause.

![Representation baseline](figures/figure_4_5_representation_baseline.png)

## 4.6 Hypothesis Assessment

| hypothesis | assessment | evidence |
|---|---|---|
| H1 | Supported methodologically | Split precedes normalization; scaler scope is train_only; imbalance handling uses training artifacts; test integrity hashes remain unchanged. |
| H2 | Partially supported | Latent-16 reduces dimension by 76.81% and remains classifiable, but original-69 S2 macro F1 is 0.5406 versus 0.4249 for latent-16. |
| H3 | Supported | S1 has the best accuracy, while S2 has the best macro recall/F1; per-class recall and confusion change materially across scenarios. |

H3 is supported because imbalance handling materially changed macro recall, macro F1, and class-level confusion. The contrast between S1 accuracy and S2 macro performance also confirms that accuracy alone is insufficient for this dataset.

## 4.7 Supplementary Diagnostics

A post-hoc data-quality audit found 145 byte-identical DoS/DDoS feature groups with contradictory labels. All 145 DoS rows were involved, together with 71.78% of DDoS rows. This label contradiction provides a direct explanation for persistent mutual confusion and must be stated as a dataset limitation.

The exploratory Autoencoder sensitivity study reached its best macro F1 (0.4395) with v4_linear_32_weighted. This remains below the original-69 S2 baseline and is supplementary rather than a replacement for the pre-registered latent-16 pipeline.

Within the proposal's LightGBM sensitivity grid, the strongest latent-16 point was learning_rate=0.05 and n_estimators=200 with macro F1 0.4249. Because these sensitivity values were evaluated on the held-out test set, they are descriptive and must not be used as post-hoc test-set model selection.

A separate nine-class diagnostic merged DoS and DDoS to remove the contradictory decision boundary. Its S2 macro F1 was 0.5136. This value is not directly comparable with the primary ten-class metrics and must remain a supplementary dataset-correction experiment.

## 4.8 Limitations

- DoS and DDoS have extremely small support and contradictory duplicate labels.
- S3 uses random duplication, so its effective class contribution is mathematically close to S2 while requiring substantially more storage and training time.
- S4 removes 99.97% of training rows and is intentionally an extreme comparison.
- The original-feature baseline improves predictive metrics, so dimensionality reduction should be justified by compactness and classifier cost, not accuracy gain.
- Exploratory variant and tuning results are sensitivity analyses, not confirmation from an independent second test set.
- Quick-run metrics are smoke-test outputs and are excluded from every table and conclusion in this chapter.

## 4.9 Chapter Conclusion

The experiment supports the leakage-aware preprocessing procedure in H1 and the imbalance-sensitivity claim in H3. H2 is partially supported: latent-16 is compact and useful, but the optional original-69 baseline is stronger under the primary macro metrics. S2 class weighting is the preferred ten-class latent scenario for macro recall and macro F1, while S1 remains preferable only when aggregate accuracy is prioritized over balanced class performance.

## Generated Materials

- `tables/`: CSV tables for direct import into the thesis source.
- `figures/`: full-data PNG figures at 180 DPI.
- `report_manifest.json`: machine-readable provenance and hypothesis conclusions.
