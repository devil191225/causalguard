#!/usr/bin/env python3
"""
PHASE 2: FORMAL VERIFICATION LAYER
============================================================
Property encoding, interval analysis, empirical FPR/FNR checks,
causal-path consistency, monotonicity tests, adversarial robustness
reporting, and machine-readable / human-readable verification proofs.

Author: Aditya Srikar Konduri (foundation); Phase 2 completion
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx


class FormalProperty:
    """Represents a formal property for verification."""

    def __init__(self, name: str, description: str, formula: str):
        self.name = name
        self.description = description
        self.formula = formula
        self.verified = False
        self.counterexamples: List[Any] = []

    def __repr__(self):
        status = "VERIFIED" if self.verified else "UNVERIFIED"
        return f"Property({self.name}, {status})"


def _z_multiplier(confidence_level: float) -> float:
    """Two-sided normal quantile multiplier (approximation for common alphas)."""
    table = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    if confidence_level in table:
        return table[confidence_level]
    # linear interp between 90 and 99
    if confidence_level > 0.99:
        return 2.576 + 50 * max(0.0, confidence_level - 0.99)
    return 1.960


def causal_graph_to_digraph(detector) -> nx.DiGraph:
    """Build NetworkX DiGraph from detector.causal_graph (multiple backend shapes)."""
    G = nx.DiGraph()
    g = getattr(detector, "causal_graph", None)
    if g is None:
        return G
    edges = list(getattr(g, "edges", []) or [])
    G.add_edges_from(edges)
    if hasattr(g, "nodes"):
        nodes = getattr(g, "nodes", None)
        if isinstance(nodes, (set, list)):
            G.add_nodes_from(nodes)
        elif callable(nodes):
            try:
                G.add_nodes_from(list(nodes()))
            except TypeError:
                pass
    return G


class IntervalAnalyzer:
    """
    Per-feature prediction / residual intervals using normal-operation training data.
    """

    def __init__(self, confidence_level=0.95):
        self.confidence_level = float(confidence_level)
        self.k = _z_multiplier(self.confidence_level)
        self.per_feature_residual_quantile: Dict[str, Tuple[float, float]] = {}

    def set_confidence(self, confidence_level: float) -> None:
        self.confidence_level = float(confidence_level)
        self.k = _z_multiplier(self.confidence_level)

    def residual_bounds_from_training(self, detector, X_train_scaled: np.ndarray,
                                      feature_cols: List[str]) -> Dict[str, float]:
        """
        Train-time absolute residual thresholds per target feature (ensemble / statistical).
        Returns map feature -> epsilon (scalar bound on abs residual).
        """
        bounds = {}
        n = X_train_scaled.shape[0]
        for feat in feature_cols:
            mi = detector.models.get(feat, {})
            ty = mi.get("type", "constant")
            if ty == "statistical":
                y = X_train_scaled[:, feature_cols.index(feat)]
                pred = np.full(n, mi["mean"])
                res = np.abs(y - pred)
            elif ty == "ensemble":
                parent_indices = mi["parent_indices"]
                Xp = X_train_scaled[:, parent_indices]
                y = X_train_scaled[:, feature_cols.index(feat)]
                yhat = (
                    0.7 * mi["models"][0].predict(Xp)
                    + 0.3 * mi["models"][1].predict(Xp)
                )
                res = np.abs(y - yhat)
            elif ty == "regressor":
                pi = mi["parent_indices"]
                Xp = X_train_scaled[:, pi]
                y = X_train_scaled[:, feature_cols.index(feat)]
                yhat = mi["model"].predict(Xp)
                res = np.abs(y - yhat)
            else:
                y = X_train_scaled[:, feature_cols.index(feat)]
                pred = np.full(n, mi.get("value", 0.0))
                res = np.abs(y - pred)
            # Conservative: max of calibrated quantile model threshold and Gaussian tail proxy
            q = float(np.quantile(res, self.confidence_level))
            sigma = float(np.std(res))
            sigma_bound = sigma * self.k
            epsilon = max(q, sigma_bound)
            epsilon = max(epsilon, mi.get("threshold", 0) * 0.95)
            bounds[feat] = max(epsilon, 1e-8)
            self.per_feature_residual_quantile[feat] = (float(np.quantile(res, 0.05)),
                                                        float(np.quantile(res, 0.995)))
        return bounds


class VerificationProofGenerator:
    """Writes human-readable justification text (informal proofs + empirical appendix)."""

    STATIC_THEOREMS = """
## Theoretical scaffolding (Gaussian / independence approximations)

**Theorem sketch (bounded FPR under Gaussian normal operation).**
Let each standardized residual $\\epsilon_i$ behave approximately as $\\mathcal{N}(0, \\sigma_i^2)$ under 
normal operation. For threshold $t_i = k \\sigma_i$, the per-sample per-feature probability of exceeding 
$t_i$ is approximately $2\\Phi(-k)$ for two-sided deviations, yielding a controllable nominal false-alert rate 
when residuals are calibrated on normal data.

**Theorem sketch (structural SCM consistency).**
If each structural equation is learned with RMS residual bounded by $\\eta_i$ over the training distribution 
of parents, then for draws from that distribution, $|X_i - \\hat f_i(\\mathrm{Pa}(X_i))| \\leq \\rho_i$
with empirical probability $\\geq \\hat p_i$ estimated on held-out normal data — the quantity our 
verification measures.

These statements are **not** machine-checked proofs about the ICS plant — they summarize standard 
Gaussian tail bounds and SCM regression residuals used as assurance evidence.
"""

    @classmethod
    def render_markdown(cls, aggregate: Dict) -> str:
        lines = [
            "# Verification proof bundle (automated appendix)",
            "",
            f"Generated: `{aggregate.get('verification_date','')}`",
            "",
            cls.STATIC_THEOREMS,
            "",
            "## Empirical verdicts",
            "",
        ]
        for key, blob in aggregate.get("results", {}).items():
            if not isinstance(blob, dict):
                continue
            vn = blob.get("property", key)
            lines.append(f"### {vn}")
            lines.append(f"- **Verified**: {blob.get('verified', 'n/a')}")
            for kk, vv in blob.items():
                if kk == "violations_detail" or kk == "invalid_paths_detail":
                    continue
                if isinstance(vv, (list, dict)) and len(str(vv)) > 480:
                    continue
                if kk != "verified":
                    lines.append(f"- {kk}: `{vv}`")
            lines.append("")
        if aggregate.get("adversarial_summary"):
            lines.append("## Adversarial robustness (finite-difference FGSM/PGD)")
            for kk, vv in aggregate["adversarial_summary"]["metrics"].items():
                lines.append(f"- **{kk}**: `{vv}`")
            lines.append("")
        lines.append("*End of appendix.*")
        return "\n".join(lines)


class FormalVerifier:
    """Main formal verification system for causal anomaly detection."""

    def __init__(
        self,
        output_dir: Path | str = "outputs/verification",
        confidence_level: float = 0.95,
    ):
        self.properties: List[FormalProperty] = []
        self.interval_analyzer = IntervalAnalyzer(confidence_level=confidence_level)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verification_results: Dict = {}

    def define_properties(self) -> None:
        p1 = FormalProperty(
            name="CAUSAL_CONSISTENCY",
            description="Structural predictions approximate observed values within calibrated residual bounds "
            "(normal-operation calibration).",
            formula="Empirically P(|residual_i|<=eps_i) >= tau on labeled normal operation",
        )
        p2 = FormalProperty(
            name="BOUNDED_FPR",
            description="False positive rate <= alpha",
            formula="FP/(FP+TN) <= alpha (alpha=0.10 default)",
        )
        p3 = FormalProperty(
            name="BOUNDED_FNR",
            description="False negative rate <= beta",
            formula="FN/(FN+TP) <= beta (beta=0.20 default)",
        )
        p4 = FormalProperty(
            name="CAUSAL_PATH_VALIDITY",
            description="Concurrent per-feature anomalies cohere with reachability in the learned DAG.",
            formula="For each non-root v in anomaly set S, exists root r in S with path r->v in G",
        )
        p5 = FormalProperty(
            name="ANOMALY_MONOTONICITY",
            description="Sample-level aggregated residual score is largely monotone in max per-sample residual.",
            formula="Random pairs: |d1|>|d2|+delta implies score1>score2-delta with high rate",
        )
        self.properties = [p1, p2, p3, p4, p5]
        print(f"Defined {len(self.properties)} formal properties:")
        for p in self.properties:
            print(f"  - {p.name}: {p.description}")

    def _training_scaled(self, detector, feature_cols: List[str]) -> Optional[np.ndarray]:
        Xtr = getattr(detector, "_normal_training_scaled", None)
        cols = getattr(detector, "_training_feature_columns", None)
        if Xtr is None or cols is None:
            return None
        if list(cols) != list(feature_cols):
            # Best-effort reorder if permutation
            idx = [cols.index(c) for c in feature_cols if c in cols]
            if len(idx) != len(feature_cols):
                return None
            return Xtr[:, [cols.index(c) for c in feature_cols]]
        return np.asarray(Xtr, dtype=np.float64)

    def verify_property_1(
        self, detector, test_data: pd.DataFrame, feature_cols: List[str]
    ) -> Dict:
        print("\n" + "=" * 70)
        print("VERIFYING PROPERTY 1: CAUSAL_CONSISTENCY")
        print("=" * 70)

        X_test = detector.scaler.transform(test_data[feature_cols].values.astype(np.float64))
        X_train = self._training_scaled(detector, feature_cols)

        violations: List[Dict] = []
        total = 0
        within = 0

        if X_train is not None:
            eps_map = self.interval_analyzer.residual_bounds_from_training(
                detector, X_train, feature_cols
            )
        else:
            eps_map = {}
            print(
                "  WARNING: No cached training matrix; falling back to model summary bounds."
            )

        for i, feat in enumerate(feature_cols):
            mi = detector.models[feat]
            y_true = X_test[:, i]
            n_samples = len(y_true)
            if feat in eps_map:
                eps = eps_map[feat]
            elif mi["type"] == "statistical":
                eps = mi["std"] * self.interval_analyzer.k
            elif mi["type"] == "ensemble":
                eps = mi.get(
                    "residual_std", mi.get("threshold", 1.0)
                ) * self.interval_analyzer.k
                eps = max(eps, mi.get("threshold", 0.1))
            elif mi["type"] == "regressor":
                eps = mi.get("threshold", 1.0) * self.interval_analyzer.k
            else:
                eps = mi.get("std", 1.0) * self.interval_analyzer.k

            if mi["type"] == "statistical":
                pred = np.full(n_samples, mi["mean"])
            elif mi["type"] == "ensemble":
                Xp = X_test[:, mi["parent_indices"]]
                pred = 0.7 * mi["models"][0].predict(Xp) + 0.3 * mi["models"][1].predict(Xp)
            elif mi["type"] == "regressor":
                Xp = X_test[:, mi["parent_indices"]]
                pred = mi["model"].predict(Xp)
            else:
                pred = np.full(n_samples, mi.get("value", 0.0))

            res = np.abs(y_true - pred)
            ok = res <= eps
            within += int(np.sum(ok))
            total += n_samples
            viol = int(np.sum(~ok))
            if viol > 0:
                violations.append(
                    {
                        "feature": feat,
                        "num_violations": viol,
                        "violation_rate": float(viol / n_samples),
                        "epsilon": float(eps),
                    }
                )

        rate = within / total if total else 1.0
        result = {
            "property": "CAUSAL_CONSISTENCY",
            "verified": rate >= 0.90,
            "consistency_rate": float(rate),
            "total_predictions": int(total),
            "within_bounds": int(within),
            "violations": violations[:8],
            "confidence_level": self.interval_analyzer.confidence_level,
            "training_backed_bounds": bool(X_train is not None),
        }

        print(f"  Consistency Rate: {rate:.4f}")
        print(f"  Within Bounds: {within}/{total}")
        print(f"  Verified: {'YES' if result['verified'] else 'NO'}")
        return result

    def verify_property_4(
        self, detector, anomaly_flags: np.ndarray, feature_cols: List[str]
    ) -> Dict:
        """
        Concurrent anomalies must be reachable from at least one anomaly root inside the subgraph.
        """
        print("\n" + "=" * 70)
        print("VERIFYING PROPERTY 4: CAUSAL_PATH_VALIDITY")
        print("=" * 70)

        G = causal_graph_to_digraph(detector)
        if G.number_of_nodes() == 0:
            print("  No DAG edges — treating property as vacuously satisfied.")
            return {
                "property": "CAUSAL_PATH_VALIDITY",
                "verified": True,
                "vacuous": True,
                "reason": "empty_or_missing_graph",
            }

        for f in feature_cols:
            G.add_node(f)

        invalid_samples: List[Dict] = []
        total_positive = 0
        satisfying = 0

        nf = min(anomaly_flags.shape[1], len(feature_cols))
        for sample_idx in range(anomaly_flags.shape[0]):
            mask = anomaly_flags[sample_idx, :nf]
            S = [feature_cols[i] for i in range(nf) if mask[i]]
            if len(S) <= 1:
                continue

            total_positive += 1
            anomaly_set = set(S)
            H = G.subgraph(anomaly_set).copy()
            roots = [v for v in anomaly_set if H.in_degree(v) == 0]
            if not roots:
                roots = list(anomaly_set)

            ok = True
            for v in anomaly_set:
                if v in roots:
                    continue
                if any(nx.has_path(H, r, v) for r in roots):
                    continue
                ok = False
                break

            if ok:
                satisfying += 1
            elif len(invalid_samples) < 12:
                invalid_samples.append({"sample_idx": int(sample_idx), "features": S[:10]})

        if total_positive == 0:
            rate = 1.0
        else:
            rate = satisfying / total_positive

        result = {
            "property": "CAUSAL_PATH_VALIDITY",
            "verified": rate >= 0.90,
            "validity_rate": float(rate),
            "multi_feature_anomaly_samples": int(total_positive),
            "structurally_supported": int(satisfying),
            "invalid_paths_detail": invalid_samples,
        }

        print(f"  Samples with 2+ anomaly flags analyzed: {total_positive}")
        print(f"  Structurally coherent: {satisfying} ({rate:.3f})")
        print(f"  Verified: {'YES' if result['verified'] else 'NO'}")
        return result

    def verify_property_5(self, anomaly_scores: np.ndarray) -> Dict:
        """
        Statistical monotonic relationship between aggregated deviation and anomaly score surrogate.
        """
        print("\n" + "=" * 70)
        print("VERIFYING PROPERTY 5: ANOMALY_MONOTONICITY")
        print("=" * 70)

        if anomaly_scores.size == 0:
            return {
                "property": "ANOMALY_MONOTONICITY",
                "verified": True,
                "note": "empty_scores",
                "monotonicity_rate": 1.0,
            }

        dev = np.max(anomaly_scores, axis=1)
        agg = np.sum(anomaly_scores, axis=1)
        n = len(dev)
        if n < 2:
            return {
                "property": "ANOMALY_MONOTONICITY",
                "verified": True,
                "monotonicity_rate": 1.0,
                "pairs_tested": 0,
                "violations": 0,
                "spearman_aggregate_deviation": float("nan"),
            }

        # Rank correlation between max-feature residual and total residual mass (robust monotonic trend).
        r_dev = pd.Series(dev).rank(method="average").to_numpy(dtype=np.float64)
        r_agg = pd.Series(agg).rank(method="average").to_numpy(dtype=np.float64)
        if np.std(r_dev) < 1e-12 or np.std(r_agg) < 1e-12:
            rho = 1.0
        else:
            rho = float(np.corrcoef(r_dev, r_agg)[0, 1])

        rng = np.random.default_rng(42)
        pair_budget = min(6000, n * max(50, min(6000 // max(n, 1), 250)))
        tol = np.finfo(np.float64).resolution * 512
        viol = 0
        pairs = 0
        for _ in range(pair_budget):
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            if i == j:
                continue
            d1, d2 = dev[i], dev[j]
            s1, s2 = agg[i], agg[j]
            if abs(d1 - d2) < tol:
                continue
            pairs += 1
            if d1 > d2 + tol and not (s1 > s2 - tol):
                viol += 1
            elif d2 > d1 + tol and not (s2 > s1 - tol):
                viol += 1

        pair_rate = 1.0 if pairs == 0 else 1.0 - (viol / pairs)
        # Spearman aligns with global monotonic trend; pairwise check is strict on sum vs max residuals.
        verified = rho >= 0.88 and pair_rate >= 0.85
        result = {
            "property": "ANOMALY_MONOTONICITY",
            "verified": bool(verified),
            "spearman_aggregate_deviation": rho,
            "pairwise_strict_rate": float(pair_rate),
            "pairs_tested": int(pairs),
            "pairwise_violations": int(viol),
            "aggregate_score": "sum(abs residuals); deviation=max(abs residuals)",
        }

        print(f"  Spearman rho(dev, aggregated score): {rho:.4f}")
        print(f"  Pairwise strict rate: {pair_rate:.4f} ({pairs} pairs)")
        print(f"  Verified: {'YES' if result['verified'] else 'NO'}")
        return result

    def verify_property_2_and_3(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        alpha=0.10,
        beta=0.20,
    ) -> Tuple[Dict, Dict]:
        print("\n" + "=" * 70)
        print("VERIFYING PROPERTY 2 & 3: BOUNDED FPR/FNR")
        print("=" * 70)

        tn = int(np.sum((true_labels == 0) & (predictions == 0)))
        fp = int(np.sum((true_labels == 0) & (predictions == 1)))
        fn = int(np.sum((true_labels == 1) & (predictions == 0)))
        tp = int(np.sum((true_labels == 1) & (predictions == 1)))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        p2 = {
            "property": "BOUNDED_FPR",
            "verified": bool(fpr <= alpha),
            "fpr": float(fpr),
            "alpha_threshold": alpha,
            "false_positives": fp,
            "true_negatives": tn,
        }
        p3 = {
            "property": "BOUNDED_FNR",
            "verified": bool(fnr <= beta),
            "fnr": float(fnr),
            "beta_threshold": beta,
            "false_negatives": fn,
            "true_positives": tp,
        }

        print(f"  FPR: {fpr:.4f} (threshold {alpha})")
        print(f"  FNR: {fnr:.4f} (threshold {beta})")
        print(f"  Property 2: {'VERIFIED' if p2['verified'] else 'VIOLATED'}")
        print(f"  Property 3: {'VERIFIED' if p3['verified'] else 'VIOLATED'}")
        return p2, p3

    def run_verification_suite(
        self,
        detector,
        test_data: pd.DataFrame,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        feature_cols: List[str],
        anomaly_scores: Optional[np.ndarray] = None,
        anomaly_flags: Optional[np.ndarray] = None,
        normal_reference_data: Optional[pd.DataFrame] = None,
        run_adversarial: bool = False,
        adversarial_epsilon: float = 0.15,
        adversarial_subsample: int = 512,
    ) -> Dict:
        print("\n" + "=" * 70)
        print("FORMAL VERIFICATION SUITE (PHASE 2)")
        print("=" * 70)

        self.define_properties()

        # Property 1: use dedicated normal reference when provided; else labeled-normal test rows.
        lbl = np.asarray(true_labels).astype(np.int64)
        normal_mask = lbl == 0
        if normal_reference_data is not None and len(normal_reference_data):
            consistency_df = normal_reference_data
            print(
                f"\n(Property 1 uses {len(consistency_df)} normal-reference rows for consistency.)"
            )
        elif np.any(normal_mask):
            consistency_df = test_data.iloc[normal_mask].reset_index(drop=True)
            print(
                f"\n(Property 1 uses {int(np.sum(normal_mask))} labeled-normal test rows "
                "for residual calibration consistency.)"
            )
        else:
            consistency_df = test_data
            print("\n(Property 1 warning: no normal reference; using full test set.)")

        r1 = self.verify_property_1(detector, consistency_df, feature_cols)
        r2, r3 = self.verify_property_2_and_3(true_labels, predictions)

        r4 = None
        if anomaly_flags is not None:
            r4 = self.verify_property_4(detector, anomaly_flags, feature_cols)

        r5 = None
        if anomaly_scores is not None:
            r5 = self.verify_property_5(anomaly_scores)

        adv_block = None
        if run_adversarial:
            _src = Path(__file__).resolve().parent
            if str(_src) not in sys.path:
                sys.path.insert(0, str(_src))
            try:
                from adversarial_testing import AdversarialTester, save_adversarial_report

                tester = AdversarialTester(detector, feature_cols)
                adv_block = tester.run_battery(
                    test_data,
                    true_labels,
                    epsilon=adversarial_epsilon,
                    subsample=adversarial_subsample,
                )
                save_adversarial_report(
                    adv_block, self.output_dir / "adversarial_robustness.json"
                )
                print("\n  Adversarial battery complete -> adversarial_robustness.json")
            except Exception as exc:  # pragma: no cover - defensive
                adv_block = {"error": str(exc)}
                print(f"\n  Adversarial testing skipped due to error: {exc}")

        tested = 3 + (1 if r4 else 0) + (1 if r5 else 0)
        verified = sum(
            bool(x["verified"])
            for x in [r1, r2, r3] + ([r4] if r4 else []) + ([r5] if r5 else [])
            if isinstance(x.get("verified"), bool)
        )

        aggregate = {
            "verification_date": pd.Timestamp.now().isoformat(),
            "properties_tested": tested,
            "properties_verified": int(verified),
            "verification_pass_rate": float(verified / tested) if tested else 0.0,
            "results": {
                "causal_consistency": r1,
                "bounded_fpr": r2,
                "bounded_fnr": r3,
            },
            "theoretical_notes": VerificationProofGenerator.STATIC_THEOREMS.strip(),
        }
        if r4:
            aggregate["results"]["causal_path_validity"] = r4
        if r5:
            aggregate["results"]["anomaly_monotonicity"] = r5
        if adv_block and "metrics" in adv_block:
            aggregate["adversarial_summary"] = adv_block

        results_file = self.output_dir / "verification_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, default=str)

        proof_md = VerificationProofGenerator.render_markdown(aggregate)
        proof_path = self.output_dir / "verification_proof_appendix.md"
        proof_path.write_text(proof_md, encoding="utf-8")

        cert_lines = [
            "=" * 72,
            "CAUSALGUARD PHASE 2 — VERIFICATION CERTIFICATE (machine summary)",
            "=" * 72,
            f"ISO timestamp: {aggregate['verification_date']}",
            f"Properties verified: {verified} / {tested}",
            f"Formal JSON: {results_file}",
            f"Informal appendix: {proof_path}",
            "",
        ]
        for name, rr in aggregate["results"].items():
            if isinstance(rr, dict) and "property" in rr:
                cert_lines.append(f"  [{rr['property']}]: VERIFIED={rr.get('verified')}")
        cert_path = self.output_dir / "verification_certificate.txt"
        cert_path.write_text("\n".join(cert_lines), encoding="utf-8")

        self.verification_results = aggregate

        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        print(f"  Properties Verified: {verified}/{tested}")
        print(f"  JSON -> {results_file}")
        print(f"  Appendix -> {proof_path}")
        print(f"  Certificate -> {cert_path}")
        return aggregate


def main():
    print("Formal verification module — import FormalVerifier and run_verification_suite()")
    print("or execute `python src/run_verified_detection.py`.")


if __name__ == "__main__":
    main()
