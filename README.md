> [!NOTE]
> **Authorized Use Only** — This is a security research project. Use only on systems you own or have explicit written authorization to access/monitor. Read [DISCLAIMER.md](./DISCLAIMER.md) and [LICENSE](./LICENSE) before use.

---

# Certifiable Anomaly Detection (CausalGuard)

Research software for **causal, explainable anomaly detection** on industrial control telemetry, with optional **formal verification** of detection properties and **stress testing** via synthetic edge cases and realistic attack scenarios.

---

## Role in the Cybersecurity Landscape

Industrial and operational technology (OT) environments increasingly rely on **behavioral anomaly detection** to spot process manipulation, stealthy actuator changes, and sensor spoofing when signatures are insufficient or unknown. Typical offerings include statistical baselines, black-box ML, and vendor-specific analytics layered on SIEM/XDR workflows.

This repository occupies a narrower slice of that landscape:


| Layer                | Common practice                    | What this project adds                                                                                                                                                                                                                            |
| -------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Detection signal** | Correlation-heavy or opaque scores | **Structural causal modeling (SCM)** over process variables: anomalies are inferred relative to assumed cause-effect structure, enabling path-level explanations                                                                                   |
| **Assurance**        | Retrospective red-teaming only     | **FormalVerifier** encodes stated properties (residual consistency on normal data, empirical FPR/FNR, DAG coherence of multi-feature alerts, monotonicity diagnostics) and exports structured JSON, a narrative proof appendix, and a short certificate |
| **Evaluation**       | Fixed benchmarks                   | **EdgeCaseTester**, **RealisticAttackGenerator**, and **comprehensive_stress_test** generate challenging synthetic regimes (drift, mimicry, pulses, ramps), similar in spirit to adversarial and testbed-style evaluation in ML and ICS research |


It is **not** a turnkey OT firewall, ICS protocol parser, or production SOC product. It is best understood as a **research prototype** for studying how causal structure and verification hooks can complement conventional anomaly detectors on **tabular ICS-style time series** (here centered on the public SWaT dataset format).

---

## Repository Contents

### Core detection and causal structure


| Component                       | Module                         | Purpose                                                                                                                                                                                |
| ------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SWaT causal detector**        | `src/causal_detector.py`       | Loads and preprocesses SWaT-format CSVs, learns or loads a causal graph, trains structural models (e.g., ensemble regressors), scores anomalies, and produces causal-path explanations |
| **Causal discovery (optional)** | `src/true_causal_discovery.py` | PC / GES-style discovery and validation helpers when `causalnex` / `dowhy` are available                                                                                                |


### Verification and certification-style reporting


| Component                  | Module                       | Purpose                                                                                                                                                       |
| -------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Formal verification**    | `src/formal_verification.py` | Five properties, training-calibrated residual bounds, DAG path checks, monotonicity diagnostics, JSON bundle plus narrative appendix and plaintext certificate |
| **Adversarial robustness** | `src/adversarial_testing.py` | FGSM / PGD / graph-biased perturbations using finite-difference gradients on a scalar residual objective (ensemble includes non-differentiable tree models)    |


The five encoded properties are: **CAUSAL_CONSISTENCY** (normal-operation residual bounds), **BOUNDED_FPR** and **BOUNDED_FNR** (empirical thresholds), **CAUSAL_PATH_VALIDITY** (multi-feature anomalies align with directed paths in the learned graph), and **ANOMALY_MONOTONICITY** (rank agreement between aggregated and max per-sample residuals).

### Evaluation, stress testing, and data generation


| Component                     | Module                              | Purpose                                                                                                                                            |
| ----------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Integrated demo**           | `src/run_verified_detection.py`     | End-to-end train, detect, and verification suite; optional `--adversarial` FGSM/PGD battery                                                           |
| **Baseline analysis**         | `src/baseline_swat_analysis.py`     | Exploratory and comparative baseline runs on SWaT-oriented workflows                                                                               |
| **Comprehensive stress test** | `src/comprehensive_stress_test.py`  | Aggregates stress scenarios for robustness characterization                                                                                        |
| **Edge case synthesis**       | `src/edge_case_tester.py`           | Generates difficult synthetic regimes (adversarial, drift, cascades, mimicry, intermittent behavior) under `outputs/edge_cases/`                   |
| **Realistic attack CSVs**     | `src/realistic_attack_generator.py` | Builds subtle ICS-motivated scenarios (setpoint creep, spoofing, slow ramps, pulses, model-informed perturbations) under `data/realistic_attacks/` |
| **Attack evaluation harness** | `src/test_realistic_attacks.py`     | Exercises the detector against generated realistic attack data                                                                                     |
| **Adaptive learning demo**    | `src/demo_adaptive_learning.py`     | Demonstrates persistence and incremental adjustment behavior                                                                                       |

### Tests

| Component              | Path                         | Purpose                                      |
| ---------------------- | ---------------------------- | -------------------------------------------- |
| **Phase 2 smoke tests** | `tests/test_phase2_verification.py` | Pytest coverage for formal verification helpers and suite wiring |

---

## Requirements

### Python

Use Python 3.9+ (3.10+ recommended). Core dependencies:

```
pandas, numpy, scikit-learn, networkx, matplotlib, seaborn, plotly, tqdm, joblib
```

Install dependencies (for example):

```bash
pip install pandas numpy scikit-learn networkx matplotlib seaborn plotly tqdm joblib
```

### Optional causal-discovery stacks

Install when you want full integration with external structure-learning libraries:

- **causalnex**: NOTEARS-style structure from data (when you install the package).
- **dowhy**: causal modeling utilities.

If those packages are absent, the code falls back to simplified structure logic where applicable.

Optional **SciPy** is useful for parts of edge-case tooling; install with `pip install scipy` if imports fail.

---

## Data

Default workflows expect SWaT-style CSV files under `data/`:

- `data/SWaT_Dataset_Normal_v1.csv`: predominantly normal operation.
- `data/SWaT_Dataset_Attack_v0.csv`: periods containing attacks (with labels).

If these files are missing, several entry points fall back to **synthetic sample data** inside the detector for smoke execution only.

Synthetic attack corpora produced by this repository live under `data/realistic_attacks/`.

---

## Usage

Run all commands from the **repository root** so relative paths to `data/` and `outputs/` resolve correctly.


| Goal                                       | Command                                                 |
| ------------------------------------------ | ------------------------------------------------------- |
| Full detection + formal verification demo  | `python src/run_verified_detection.py`                  |
| Verification + FGSM/PGD robustness battery | `python src/run_verified_detection.py --adversarial`   |
| Run causal detector module directly        | `python src/causal_detector.py`                         |
| Run formal verification entrypoint         | `python src/formal_verification.py`                     |
| Generate realistic attack CSVs             | `python src/realistic_attack_generator.py`              |
| Test detector on realistic attacks         | `python src/test_realistic_attacks.py`                  |
| Edge-case generation and tests             | `python src/edge_case_tester.py`                        |
| Stress test suite                          | `python src/comprehensive_stress_test.py`               |
| Baseline SWaT analysis                     | `python src/baseline_swat_analysis.py`                  |
| Adaptive learning demo                     | `python src/demo_adaptive_learning.py`                  |
| Causal discovery experiments               | `python src/true_causal_discovery.py`                   |
| Phase 2 verification tests (pytest)        | `python -m pytest tests/test_phase2_verification.py -v` |


Calling `SWaTCausalDetector().run_full_pipeline(verify=True, run_adversarial=False)` runs detection, then the same formal suite (`run_adversarial=True` includes the adversarial battery).

### Outputs

Typical artifacts are written under `outputs/`, including:

- `outputs/verification/`: numerical and summary JSON, optional adversarial run JSON, a human-readable proof appendix export, and a short certificate file.
- `outputs/stress_tests/`: stress test metrics.
- `outputs/edge_cases/`: generated edge-case datasets and plots.
- `outputs/results/`: explanations, metrics, counterfactual examples.
- `outputs/baseline_analysis/`: baseline comparison JSON and plots.
- `outputs/adaptive/`: adaptive run history where applicable.

Exact filenames may vary by script version and run configuration.

---

## Design Notes

- **Causal graph + structural models**: Residuals are interpreted using a directed structure over process variables, which supports root-cause-style narratives that pure distance-based detectors do not provide natively.
- **Formal properties**: The verification layer states explicitly what is checked (residual calibration on normal data, confusion-matrix FPR/FNR, structural coherence, monotonicity statistics). These are **empirical checks and informal proof sketches**, not machine-verified theorems about a physical plant or a guaranteed safety case.
- **Evaluation scope**: Strong claims about deployment need operational data, domain review, and independent red-teaming. Metrics from this repository reflect behavior under the chosen datasets and generators.

---

## Status and Limitations

- **Research prototype**: Not hardened for production OT deployment; expect to add proper data governance, access control, monitoring, and plant-specific validation.
- **Learned graph**: Recovered structure is an estimate; wrong edges undermine explanations and path-based checks. Sparsity and discovery hyperparameters materially affect graphs.
- **Labels and leakage**: Attack or scenario columns must be excluded from feature learning; overly dense graphs weaken interpretability.
- **Adversarial testing**: FGSM/PGD use **finite-difference** gradients of a surrogate scalar score because tree ensembles are not smoothly differentiable; results characterize this procedure, not all possible perturbations.
- **Synthetic and benchmark bias**: Fallback synthetic data and fixed benchmarks do not substitute for field evidence. Verification pass/fail is **configuration- and dataset-dependent**.
- **Performance claims**: Latency or accuracy figures from notebook-style runs vary with hardware and data subset; interpret as indicative, not audited certifications.

---

## Steps still ahead for a commercial product

Today this project is a **research and evaluation codebase**, not a packaged OT product. Turning it into something sellable usually means closing gaps in **scope, engineering, data rights, and operations**:

1. **Define the SKU.** Choose a narrow product boundary (e.g. historian-based process anomaly + causal explanations + assurance reports) versus chasing full OT IDS scope; that drives ingest, UX, and compliance.
2. **Production hardening.** Long-running service or appliance packaging, configuration and secrets management, structured logging, upgrades and rollback, health checks, and secure defaults (not only CLI scripts).
3. **Plant connectivity.** Replace CSV-centric flows with supported paths to real telemetry (e.g. OPC UA, Modbus, PI/historian exports, Kafka); normalize time zones, tag naming, and train/score separation.
4. **Operational ML.** Scheduled retraining, drift monitoring, versioning of models and causal graphs, and clear operator controls when graphs or thresholds change.
5. **Operator and SOC workflow.** Dashboards for alert triage, graph and narrative views, exports for incident response, role-based access, and audit trails.
6. **Go-to-market and liability.** Contracts, support tiers, on-prem or private-cloud deployment, and careful wording so "verification" is sold as **documented empirical testing**, not a mathematical guarantee of physical safety.
7. **Evidence that buyers trust.** Strengthen the evaluation ladder: public process-oriented benchmarks and stress suites for R&D, **owned testbed** data for IP-clear demos and regressions, and **partner or customer telemetry** (under agreement) to show behavior on real operating conditions.

Nothing in this list is a single weekend of work; it is the usual distance between a strong prototype and something operators can run, patch, and defend in procurement.

---

## Acknowledgments

SWaT (Secure Water Treatment) datasets are widely used in the ICS security research community; this project uses them as a **public benchmark format** rather than asserting affiliation with any testbed operator. Module-level source comments retain author attribution where present.

---

## Citation

If you use this software in academic work, cite the repository URL and commit hash, and cite the SWaT dataset and any third-party causal-discovery libraries you enable.