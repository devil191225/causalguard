# CAUSALGUARD: HONEST LIMITATIONS & STATUS
**Date:** October 19, 2025  
**Version:** 1.0-alpha  
**Status:** Research Prototype

---

## EXECUTIVE SUMMARY

CausalGuard is a **research prototype** demonstrating the application of true causal discovery (PC/GES algorithms) to ICS anomaly detection. The core methodology is novel and shows promise, but **significant limitations exist** that prevent production deployment.

**Key Achievement:** Implemented and validated PC algorithm for causal structure learning on known synthetic structures (F1=1.0 recovery).

**Critical Limitation:** All testing performed on **synthetic data**. Real-world performance is **unknown** and likely significantly lower.

---

## WHAT WORKS (VALIDATED)

### 1. **True Causal Discovery** ✅
- **PC Algorithm**: Implemented from scratch, validated on known causal structures
- **Test Case**: Synthetic data P1→P2→P3→P4, P1→P3
- **Result**: Perfect recovery (F1=1.0, all 4 edges correct, 0 false positives/negatives)
- **Conclusion**: Algorithm implementation is **correct**

### 2. **Structural Causal Models** ✅  
- **Ensemble Models**: Random Forest + Linear Regression
- **Precision-Recall Optimization**: Dynamic threshold selection
- **Adaptive Learning**: Model persistence and incremental updates
- **Status**: **Implemented and functional**

### 3. **Counterfactual Explanations** ✅
- **Root Cause Identification**: Distinguishes primary failures from cascades
- **Causal Path Tracing**: Tracks anomaly propagation through DAG
- **Human-Readable Output**: Text format with actual vs. expected values
- **Status**: **Implemented**

### 4. **Performance Benchmarking** ✅
- **Latency**: 53ms mean, 65ms P95 (acceptable for ICS @ 1-10Hz)
- **Throughput**: 19 samples/second
- **Status**: **Real-time capable**

---

## CRITICAL BUGS FIXED

### Bug #1: Label Leakage (FIXED) ✅
**Problem:** Attack labels (`attack`, `attack_P1`, `attack_P2`, `attack_P3`) were included in causal discovery, appearing as nodes in the graph.

**Impact:** Catastrophic - system was learning "attack label causes sensor reading" instead of true physical causation.

**Fix:** Aggressive filtering of all columns containing 'attack' in any form.

**Validation:** Graph now shows only process variables (P1-P4), no attack labels.

### Bug #2: Dense Causal Graphs (PARTIALLY FIXED) ⚠️
**Problem:** PC algorithm produced 80-94 edges for 20-23 features (4x too dense).

**Root Cause:** alpha=0.05 too permissive, max_cond_set_size=3 too complex.

**Fix:** alpha→0.01, max_cond_set_size→2

**Result:** 16 edges for 20 features (0.80 edges/feature)

**Status:** Improved but still above target (0.5 edges/feature). May indicate:
- Noisy data
- Complex interdependencies (realistic for ICS)
- Need further tuning

---

## WHAT DOESN'T WORK (KNOWN ISSUES)

###  1. **Dimension Mismatch in `detect_anomalies`** ❌ **CRITICAL BUG**

**Symptom:**
```
ValueError: Found input variables with inconsistent numbers of samples: [3000, 3]
```

**Root Cause:** Method returns `(anomaly_scores, anomaly_flags, causal_paths)` tuple, but callers expect 1D prediction array.

**Impact:** Cannot run batch evaluations on test data.

**Workaround:** Manual single-sample testing works.

**Status:** **BLOCKS REALISTIC ATTACK TESTING**

### 2. **No Real Attack Data Testing** ❌ **CRITICAL LIMITATION**

**Current Status:**
- All testing on **synthetic data** (our own generation)
- Real SWaT dataset: Not tested
- Real HAI dataset: Git LFS failed, not tested

**Performance Claims:**
- Precision: 100%
- Recall: 96.5%
- F1-Score: 0.982

**BRUTAL HONESTY:** These metrics are **MEANINGLESS** without real attack validation.

**Realistic Estimates** (based on literature review):
- Expected Precision: 60-70%
- Expected Recall: 75-85%
- Expected F1-Score: 0.65-0.75
- Expected FPR: 5-10%

### 3. **Edge Case Generator Broken** ❌
**Problem:** Generates 100% attack data, 0% normal baseline.

**Impact:** Cannot test on:
- Gradual drift attacks
- Intermittent attacks
- Mimicry attacks
- Cascade failures
- Adversarial attacks

**Status:** **BLOCKS ROBUSTNESS TESTING**

### 4. **Noise Robustness Unproven** ❌
**Problem:** All noise tests (10%-80% corruption) fail with dimension errors.

**Impact:** No evidence system handles:
- Sensor noise
- Communication errors
- Data corruption

**Status:** **UNKNOWN ROBUSTNESS**

---

## HONEST PERFORMANCE ASSESSMENT

### What We CAN Claim:
1. ✅ PC algorithm **correctly implemented** (validated on known structures)
2. ✅ System is **real-time capable** (53ms latency)
3. ✅ Causal graphs are **more sparse** after tuning (0.80 edges/feature)
4. ✅ **No label leakage** in current version
5. ✅ **Counterfactual explanations** generated

### What We CANNOT Claim:
1. ❌ "Production-ready"
2. ❌ "F1=0.984 on real attacks" (only on synthetic)
3. ❌ "Robust to noise" (untested)
4. ❌ "Handles edge cases" (tests broken)
5. ❌ "Better than baselines" (no comparison)

### What We SHOULD Claim:
**"CausalGuard is a research prototype demonstrating the feasibility of applying true causal discovery (PC algorithm) to ICS anomaly detection. Core algorithms are validated on synthetic structures. Preliminary results on synthetic attacks show F1=0.982, but real-world performance is expected to be significantly lower (estimated 0.65-0.75). System requires further validation on real attack traces and adversarial scenarios before deployment consideration."**

---

## COMPARISON TO STATE-OF-THE-ART

| Feature | CausalGuard | LSTM Autoencoder | Isolation Forest | Statistical (3σ) |
|---------|-------------|------------------|------------------|------------------|
| **True Causality** | ✅ Yes (PC) | ❌ No | ❌ No | ❌ No |
| **Explainability** | ✅ Counterfactual | ⚠️ Attention | ❌ No | ✅ Threshold |
| **Real-time** | ✅ 53ms | ✅ <100ms | ✅ <10ms | ✅ <1ms |
| **Tested on Real Data** | ❌ **NO** | ✅ Yes | ✅ Yes | ✅ Yes |
| **F1-Score** | 0.98 (synthetic) | 0.65-0.75 (real) | 0.50-0.65 | 0.40-0.55 |
| **False Positives** | Unknown | 10-20% | 15-30% | 20-40% |

**Honest Verdict:** CausalGuard has a **novel approach** (true causality) but **weaker validation** than established methods.

---

## ROADMAP TO PRODUCTION

### Phase 1: Fix Critical Bugs (1 week)
- [x] Fix label leakage
- [x] Tune PC for sparsity
- [ ] Fix `detect_anomalies` dimension mismatch
- [ ] Fix edge case generator
- [ ] Fix noise robustness tests

### Phase 2: Real Data Validation (2 weeks)
- [ ] Obtain real SWaT attack traces
- [ ] Test on HAI dataset (actual attacks)
- [ ] Achieve F1 > 0.65 on real data
- [ ] Document failure cases

### Phase 3: Baseline Comparison (1 week)
- [ ] Implement LSTM autoencoder
- [ ] Compare vs. Isolation Forest
- [ ] Compare vs. Statistical (3σ)
- [ ] Fair head-to-head evaluation

### Phase 4: Hardening (2 weeks)
- [ ] Adversarial attack testing (FGSM, PGD)
- [ ] Temporal modeling (LSTM integration)
- [ ] Online adaptation optimization
- [ ] Deployment readiness checklist

**Total Time to Production:** **6 weeks** (optimistic)

---

## KNOWN LIMITATIONS

### Theoretical Limitations:
1. **Causal Discovery Assumptions:**
   - Requires i.i.d. data (ICS data is time-series)
   - Assumes no hidden confounders
   - Requires sufficient data (we use 1,000 samples, may need more)

2. **Attack Detection Limits:**
   - **Cannot detect** attacks that maintain causal consistency
   - **Struggles with** slow, gradual changes (drift vs. attack)
   - **Vulnerable to** adversaries who know the learned graph

3. **Computational Constraints:**
   - Training: 29s for 9,000 samples (too slow for frequent retraining)
   - Sparsity: 0.80 edges/feature (target: <0.5)

### Practical Limitations:
1. **Data Quality:**
   - Sensitive to sensor noise
   - Requires clean training data
   - No handling of missing data streams

2. **Deployment:**
   - No real-time graph update mechanism
   - No distributed deployment
   - No failover/redundancy

3. **Validation:**
   - Tested only on synthetic data
   - No real ICS deployment
   - No long-term stability testing

---

## HONEST RECOMMENDATION

### For Research/Academia: ✅ **PROCEED**
- Core idea is novel
- PC algorithm correctly implemented
- Good foundation for paper/thesis
- **Caveat:** Be transparent about synthetic data limitation

### For Production/Industry: ❌ **NOT READY**
- Critical bugs still present
- No real attack validation
- Unknown robustness
- **Timeline:** 6+ weeks to deployment-ready

### For Patent Application: ⚠️ **MAYBE**
- Novel: True causal discovery for ICS security
- Novel: Cascade-aware counterfactuals
- Novel: PR-optimized SCM thresholding
- **But:** Implementation gaps weaken claims
- **Recommendation:** Fix critical bugs first, then file

---

## BRUTALLY HONEST SELF-ASSESSMENT

**What I Did Right:**
- Implemented PC algorithm from scratch correctly
- Fixed label leakage bug immediately when caught
- Tuned for sparsity
- Generated comprehensive documentation
- **Self-awareness**: Acknowledged synthetic data limitation upfront

**What I Did Wrong:**
- Tested on own synthetic data (circular validation)
- Claimed F1=0.984 without real attack testing
- Didn't catch label leakage sooner
- Dimension mismatch bug still blocking tests
- Edge case generator broken

**What I Learned:**
- Perfect scores (F1=1.0, Precision=100%) are RED FLAGS
- Synthetic data is NOT sufficient for security systems
- Label leakage is **catastrophic** but easy to miss
- Sparsity is critical for causal graphs
- Honest documentation is more valuable than inflated metrics

**Rating:** **6.5/10**
- Core technology: 9/10
- Implementation: 7/10
- Testing: **3/10**
- Validation: **2/10**
- Production-readiness: **4/10**

---

**BOTTOM LINE:**

CausalGuard demonstrates a **novel and promising approach** to ICS anomaly detection using true causal discovery. The PC algorithm implementation is **correct** and validated on known structures. However, **significant limitations exist**:

1. All testing on synthetic data
2. Critical bugs blocking realistic evaluations
3. Unknown performance on real attacks
4. No baseline comparisons

**System is suitable for research/academic work with transparent disclosure of limitations. NOT suitable for production deployment without 6+ weeks of hardening and real-world validation.**

---

**Prepared by:** Aditya Srikar Konduri  
**Date:** October 19, 2025  
**Contact:** [Redacted for security]  

**Acknowledgment:** This assessment was prepared with assistance from Claude (Anthropic) in adherence to brutal honesty and transparency standards expected at ETH Zurich.



