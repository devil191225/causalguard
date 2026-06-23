# Verification proof bundle (automated appendix)

Generated: `2026-05-07T16:38:52.850378`


## Theoretical scaffolding (Gaussian / independence approximations)

**Theorem sketch (bounded FPR under Gaussian normal operation).**
Let each standardized residual $\epsilon_i$ behave approximately as $\mathcal{N}(0, \sigma_i^2)$ under 
normal operation. For threshold $t_i = k \sigma_i$, the per-sample per-feature probability of exceeding 
$t_i$ is approximately $2\Phi(-k)$ for two-sided deviations, yielding a controllable nominal false-alert rate 
when residuals are calibrated on normal data.

**Theorem sketch (structural SCM consistency).**
If each structural equation is learned with RMS residual bounded by $\eta_i$ over the training distribution 
of parents, then for draws from that distribution, $|X_i - \hat f_i(\mathrm{Pa}(X_i))| \leq \rho_i$
with empirical probability $\geq \hat p_i$ estimated on held-out normal data — the quantity our 
verification measures.

These statements are **not** machine-checked proofs about the ICS plant — they summarize standard 
Gaussian tail bounds and SCM regression residuals used as assurance evidence.


## Empirical verdicts

### CAUSAL_CONSISTENCY
- **Verified**: True
- property: `CAUSAL_CONSISTENCY`
- consistency_rate: `0.9923444444444445`
- total_predictions: `180000`
- within_bounds: `178622`
- confidence_level: `0.95`
- training_backed_bounds: `True`

### BOUNDED_FPR
- **Verified**: True
- property: `BOUNDED_FPR`
- fpr: `0.0`
- alpha_threshold: `0.1`
- false_positives: `0`
- true_negatives: `0`

### BOUNDED_FNR
- **Verified**: True
- property: `BOUNDED_FNR`
- fnr: `0.035`
- beta_threshold: `0.2`
- false_negatives: `35`
- true_positives: `965`

### CAUSAL_PATH_VALIDITY
- **Verified**: True
- property: `CAUSAL_PATH_VALIDITY`
- validity_rate: `0.9260204081632653`
- multi_feature_anomaly_samples: `784`
- structurally_supported: `726`

### ANOMALY_MONOTONICITY
- **Verified**: False
- property: `ANOMALY_MONOTONICITY`
- spearman_aggregate_deviation: `0.8906811506811508`
- pairwise_strict_rate: `0.8559986651092941`
- pairs_tested: `5993`
- pairwise_violations: `863`
- aggregate_score: `sum(abs residuals); deviation=max(abs residuals)`

## Adversarial robustness (finite-difference FGSM/PGD)
- **detection_rate_clean_under_attack_windows**: `0.965`
- **detection_rate_fgsm_under_attack_windows**: `0.964`
- **detection_rate_pgd_under_attack_windows**: `0.965`
- **detection_rate_model_informed_under_attack_windows**: `0.964`
- **evasion_flip_rate_fgsm**: `0.001`
- **evasion_flip_rate_pgd**: `0.0`
- **epsilon_l_inf_scaled_space**: `0.15`
- **mean_l_inf_perturbation_fgsm_approx**: `4.064911250799411`

*End of appendix.*