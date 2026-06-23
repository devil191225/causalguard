# PHASE 2: FORMAL VERIFICATION - ROADMAP

**Start Date:** October 19, 2025  
**Status:** Implementation complete (code + integration + tests; interpret results per `HONEST_LIMITATIONS.md`)  
**Last code update:** Phase 2 pipeline operational

---

## 🎯 PHASE 2 OBJECTIVES

Add formal verification and mathematical guarantees to CausalGuard's causal anomaly detection system.

### Goals:
1. ✅ **Property Encoding**: Formalize detection properties
2. ✅ **Interval Analysis**: Training-calibrated residual bounds (90/95/99 via `IntervalAnalyzer`)
3. ✅ **FP/FN Guarantees**: Empirical FPR/FNR checks vs configured α, β
4. ✅ **Adversarial Testing**: FGSM / PGD / model-informed (`src/adversarial_testing.py`, finite-difference gradients)
5. ✅ **Proof Generation**: `verification_proof_appendix.md` + `verification_certificate.txt`

---

##  COMPLETED (Day 1)

### 1. Formal Property Definitions
- **CAUSAL_CONSISTENCY**: Predictions within interval bounds
- **BOUNDED_FPR**: False positive rate ≤ 10%
- **BOUNDED_FNR**: False negative rate ≤ 20%
- **CAUSAL_PATH_VALIDITY**: Anomalies follow DAG structure
- **ANOMALY_MONOTONICITY**: Larger deviations → higher scores

### 2. Foundation Code
- `src/formal_verification.py` (470 lines)
- `FormalProperty` class for property representation
- `IntervalAnalyzer` for prediction bounds
- `FormalVerifier` main verification engine

### 3. Initial Capabilities
- Property 1 verification (causal consistency)
- Property 2 & 3 verification (FPR/FNR bounds)
- JSON result export

---

## 🚧 IN PROGRESS

### 1. Property 4: Causal Path Validity
**Task**: Verify that detected anomalies follow causal DAG structure

**Implementation Plan:**
```python
def verify_property_4(self, detector, anomaly_flags, causal_paths):
    """
    Check if detected anomalies respect causal structure.
    
    For each anomaly:
    1. Identify affected features
    2. Check if path exists in DAG from root causes to effects
    3. Verify no violations of causal ordering
    """
    violations = []
    for anomaly in detected_anomalies:
        root_causes = identify_roots(anomaly, detector.causal_graph)
        effects = identify_effects(anomaly, detector.causal_graph)
        
        for effect in effects:
            if not has_path(detector.causal_graph, root_causes, effect):
                violations.append({
                    'effect': effect,
                    'roots': root_causes,
                    'violation': 'no_causal_path'
                })
    
    return len(violations) == 0
```

**Status**: Design complete, implementation pending

---

### 2. Property 5: Anomaly Monotonicity
**Task**: Verify that larger deviations result in higher anomaly scores

**Implementation Plan:**
```python
def verify_property_5(self, anomaly_scores, deviations):
    """
    Test monotonicity: |dev1| > |dev2| ⇒ score1 > score2
    """
    violations = 0
    for i in range(len(deviations) - 1):
        for j in range(i + 1, len(deviations)):
            if abs(deviations[i]) > abs(deviations[j]):
                if not (anomaly_scores[i] > anomaly_scores[j]):
                    violations += 1
    
    return violations == 0
```

**Status**: Design complete, implementation pending

---

## ⏳ PLANNED (This Week)

### Day 2-3: Interval Analysis Enhancement
**Objectives:**
- Precise interval computation using training data
- Confidence interval adjustment (90%, 95%, 99%)
- Per-feature interval bounds
- Temporal interval evolution

**Deliverables:**
- Enhanced `IntervalAnalyzer` with training data access
- Configurable confidence levels
- Visualization of prediction intervals

---

### Day 3-4: Adversarial Robustness Testing
**Objectives:**
- Implement FGSM (Fast Gradient Sign Method) attacks
- Implement PGD (Projected Gradient Descent) attacks
- Test detector robustness to adversarial perturbations
- Measure degradation under attack

**Attack Types:**
1. **FGSM**: Single-step gradient-based perturbation
2. **PGD**: Multi-step iterative attack
3. **Model-Informed**: Adversary knows causal graph

**Implementation:**
```python
class AdversarialTester:
    def fgsm_attack(self, detector, X, epsilon=0.1):
        """Generate adversarial examples using FGSM."""
        # Compute gradient of detection score w.r.t. input
        gradients = compute_gradients(detector, X)
        
        # Generate perturbation
        perturbation = epsilon * np.sign(gradients)
        X_adv = X + perturbation
        
        return X_adv
    
    def pgd_attack(self, detector, X, epsilon=0.1, alpha=0.01, iterations=10):
        """Generate adversarial examples using PGD."""
        X_adv = X.copy()
        
        for _ in range(iterations):
            gradients = compute_gradients(detector, X_adv)
            X_adv = X_adv + alpha * np.sign(gradients)
            
            # Project back into epsilon ball
            X_adv = np.clip(X_adv, X - epsilon, X + epsilon)
        
        return X_adv
```

**Deliverables:**
- `src/adversarial_testing.py`
- Robustness metrics (accuracy under attack)
- Defense recommendations

---

### Day 4-5: Mathematical Guarantees
**Objectives:**
- Formal proof of FPR/FNR bounds
- Confidence bounds on predictions
- Robustness certificates

**Theoretical Foundation:**
```
Theorem 1 (Bounded FPR):
  Given confidence level α and normal data distribution N(μ, σ²),
  the FPR is bounded by:
  
  FPR ≤ P(|X - μ| > kσ) = 2Φ(-k)
  
  where k = 1.96 for 95% confidence, Φ is standard normal CDF.

Theorem 2 (Causal Consistency):
  If f(Pa(X)) is an ε-approximation of X, then:
  
  P(|X - f(Pa(X))| > ε) ≤ δ
  
  where δ is the model's generalization error.
```

**Deliverables:**
- Mathematical proofs document
- Automated proof generation code
- Verification certificates

---

### Day 5: Integration & Testing
**Objectives:**
- Integrate verification with main detector
- Run full verification suite on all datasets
- Generate verification reports

**Integration:**
```python
# In causal_detector.py
from formal_verification import FormalVerifier

class SWaTCausalDetector:
    def run_full_pipeline(self, verify=True):
        # ... existing code ...
        
        if verify:
            verifier = FormalVerifier()
            verification_results = verifier.run_verification_suite(
                detector=self,
                test_data=attack_data,
                true_labels=true_labels,
                predictions=predictions,
                feature_cols=feature_columns
            )
            
            print(f"\nVerification: {verification_results['properties_verified']}/{verification_results['properties_tested']} properties passed")
        
        return results
```

**Deliverables:**
- Integrated verification pipeline
- Comprehensive test results
- Verification report document

---

## 📊 SUCCESS CRITERIA

### Minimum Viable Phase 2:
- [ ] All 5 properties formally defined ✅
- [ ] Properties 1-3 verified ✅ (partially)
- [ ] Properties 4-5 verified ⏳
- [ ] Adversarial testing implemented ⏳
- [ ] Integration with main detector ⏳

### Ideal Phase 2:
- [ ] Mathematical proofs generated
- [ ] Robustness certificates issued
- [ ] Automated verification on every run
- [ ] Adversarial defense mechanisms
- [ ] Formal verification paper draft

---

## 🚫 KNOWN BLOCKERS

### 1. Training Data Access
**Problem**: Interval analysis needs training data, but current API doesn't expose it

**Solution**: Refactor `causal_detector.py` to store training data or statistics

**Priority**: HIGH

### 2. Gradient Computation
**Problem**: Adversarial attacks need gradients, but Random Forest isn't differentiable

**Solutions**:
- Use gradient-free attacks (e.g., boundary search)
- Focus on Linear Regression component (differentiable)
- Approximate gradients using finite differences

**Priority**: MEDIUM

### 3. Dimension Mismatch Bug
**Problem**: `detect_anomalies` returns tuple, not 1D array

**Impact**: Blocks batch verification testing

**Solution**: Fix return format or add wrapper function

**Priority**: HIGH

---

## 🎯 NEXT IMMEDIATE STEPS

1. **TODAY**: 
   - Fix dimension mismatch bug
   - Implement Property 4 verification
   - Implement Property 5 verification

2. **TOMORROW**:
   - Enhance interval analysis with training data
   - Begin adversarial testing implementation
   - Run verification on realistic attacks

3. **DAY 3-5**:
   - Complete adversarial testing
   - Write mathematical proofs
   - Integrate with main pipeline
   - Generate Phase 2 completion report

---

## 📈 ESTIMATED PROGRESS

**Day 1 (Today)**: ████████░░ 80% Foundation
**Day 2**: ░░░░░░░░░░ 0% Properties 4-5
**Day 3**: ░░░░░░░░░░ 0% Adversarial
**Day 4**: ░░░░░░░░░░ 0% Math Proofs
**Day 5**: ░░░░░░░░░░ 0% Integration

**Overall Phase 2**: ████░░░░░░ 40% Complete

---

## 📝 DELIVERABLES CHECKLIST

### Code:
- [x] `src/formal_verification.py` (full suite)
- [x] `src/adversarial_testing.py`
- [x] Integration with `src/causal_detector.py` (`run_full_pipeline(verify=True, run_adversarial=...)`)
- [x] Unit tests: `tests/test_phase2_verification.py`

### Documentation:
- [x] This roadmap (`PHASE_2_ROADMAP.md`)
- [ ] Mathematical proofs document
- [ ] Verification report template
- [ ] API documentation

### Results:
- [ ] Verification results on synthetic data
- [ ] Verification results on realistic attacks
- [ ] Adversarial robustness metrics
- [ ] Formal certificates (if passed)

---

**BOTTOM LINE**: Phase 2 foundation is solid. Core verification framework ready. Need 4 more days of focused work to complete adversarial testing, mathematical proofs, and full integration.

**Risk Level**: LOW - Foundation proven, remaining tasks well-defined

**Blocker Risk**: MEDIUM - Dimension mismatch bug needs immediate fix

---

**Last Updated**: October 19, 2025  
**Next Milestone**: Properties 4-5 verified + dimension bug fixed (Tomorrow)



