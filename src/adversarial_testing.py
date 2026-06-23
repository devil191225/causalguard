#!/usr/bin/env python3
"""
Phase 2: Adversarial robustness testing for SCM-based anomaly detection.

Random forests are not differentiable; we use finite-difference gradients of a
smooth scalar objective (maximum per-sample residual across features), matching
how detection thresholds are applied per feature.

Includes model-informed perturbations biased along parent variables in the
learned causal graph (structural prioritization).

Author: Phase 2 implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx


@dataclass
class RobustnessMetrics:
    detection_rate_clean: float
    detection_rate_fgsm: float
    detection_rate_pgd: float
    detection_rate_informed: float
    attack_flip_rate_fgsm: float
    attack_flip_rate_pgd: float
    mean_linf_fgsm: float
    mean_linf_pgd: float


class AdversarialTester:
    """
    FGSM / PGD in scaled input space plus graph-biased model-informed attacks.
    """

    def __init__(self, detector: Any, feature_columns: List[str]):
        self.detector = detector
        self.feature_columns = feature_columns

    def _scaled_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return self.detector.scaler.transform(df[self.feature_columns].values.astype(np.float64))

    def _df_from_scaled(self, df_template: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
        raw = self.detector.scaler.inverse_transform(X_scaled)
        out = df_template.copy()
        out[self.feature_columns] = raw
        return out

    def aggregate_residual_score(self, X_scaled: np.ndarray) -> np.ndarray:
        """Per-sample scalar: max residual (same family as structural anomaly scoring)."""
        R = self.detector.compute_residuals_matrix(X_scaled, self.feature_columns)
        return np.max(R, axis=1)

    def numerical_gradient_max_residual(
        self, X_scaled: np.ndarray, h: float = 0.05
    ) -> np.ndarray:
        """
        Finite-difference gradient of max residual w.r.t. inputs (evasion pushes score down).
        """
        base = self.aggregate_residual_score(X_scaled)
        n, d = X_scaled.shape
        grad = np.zeros_like(X_scaled, dtype=np.float64)
        for j in range(d):
            Xp = np.array(X_scaled, copy=True)
            Xp[:, j] += h
            up = self.aggregate_residual_score(Xp)
            grad[:, j] = (up - base) / h
        return grad

    def _causal_boost_mask(self) -> np.ndarray:
        """Prefer perturbing sink / effect variables that have incoming edges."""
        d = len(self.feature_columns)
        boost = np.ones(d, dtype=np.float64)
        g = getattr(self.detector, "causal_graph", None)
        if g is None or not hasattr(g, "edges"):
            return boost
        G = nx.DiGraph()
        G.add_edges_from(list(g.edges))
        for j, name in enumerate(self.feature_columns):
            if G.has_node(name) and G.in_degree(name) > 0:
                boost[j] = 1.75
            else:
                boost[j] = 0.85
        return boost

    def fgsm_attack(
        self,
        df: pd.DataFrame,
        epsilon: float = 0.15,
        h: float = 0.05,
        subsample: int = 512,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Single-step L_inf attack minimizing detection score."""
        rng = np.random.default_rng(seed)
        X = self._scaled_matrix(df)
        n = X.shape[0]
        idx = np.arange(n)
        if subsample and n > subsample:
            idx = rng.choice(n, size=subsample, replace=False)
            X_sub = X[idx].copy()
        else:
            X_sub = X.copy()
        grad = self.numerical_gradient_max_residual(X_sub, h=h)
        delta = epsilon * np.sign(grad)
        X_adv_sub = np.clip(X_sub - delta, X_sub - epsilon, X_sub + epsilon)
        X_out = np.array(X, copy=True)
        if subsample and n > subsample:
            X_out[idx] = X_adv_sub
            return self._df_from_scaled(df, X_out)
        return self._df_from_scaled(df, X_adv_sub)

    def pgd_attack(
        self,
        df: pd.DataFrame,
        epsilon: float = 0.15,
        alpha: float = 0.03,
        iterations: int = 12,
        h: float = 0.05,
        subsample: int = 512,
        seed: int = 1,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        X_ref = self._scaled_matrix(df)
        n = X_ref.shape[0]
        idx = np.arange(n)
        if subsample and n > subsample:
            idx = rng.choice(n, size=subsample, replace=False)
        X0 = np.array(X_ref[idx], copy=True)
        X_adv = np.array(X0, copy=True)
        for _ in range(iterations):
            grad = self.numerical_gradient_max_residual(X_adv, h=h)
            X_adv = X_adv - alpha * np.sign(grad)
            X_adv = np.minimum(np.maximum(X_adv, X0 - epsilon), X0 + epsilon)
        X_out = np.array(X_ref, copy=True)
        X_out[idx] = X_adv
        return self._df_from_scaled(df, X_out)

    def model_informed_attack(
        self,
        df: pd.DataFrame,
        epsilon: float = 0.12,
        subsample: int = 512,
        seed: int = 2,
    ) -> pd.DataFrame:
        """
        Perturb predominantly along causal descendants (prioritizes actuator/sensor hops).
        """
        rng = np.random.default_rng(seed)
        grad = self.numerical_gradient_max_residual(self._scaled_matrix(df), h=0.05)
        mask = self._causal_boost_mask()
        boosted = np.sign(grad) * mask[np.newaxis, :]
        X = self._scaled_matrix(df)
        n = X.shape[0]
        idx = np.arange(n)
        if subsample and n > subsample:
            idx = rng.choice(n, size=subsample, replace=False)
        X_sub = np.array(X[idx], copy=True)
        boosted_sub = boosted[idx]
        delta = epsilon * np.sign(boosted_sub)
        adv_sub = np.clip(X_sub - delta, X_sub - epsilon, X_sub + epsilon)
        X_out = np.array(X, copy=True)
        X_out[idx] = adv_sub
        return self._df_from_scaled(df, X_out)

    def run_battery(
        self,
        df: pd.DataFrame,
        true_labels: np.ndarray,
        epsilon: float = 0.15,
        subsample: int = 512,
    ) -> Dict[str, Any]:
        """Compare detection behavior on clean vs attacked data (labeled subset)."""
        preds_clean = self.detector.detect_anomalies_simple(df, self.feature_columns)
        attack_rows = np.asarray(true_labels, dtype=np.int64) == 1
        df_fgsm = self.fgsm_attack(df, epsilon=epsilon, subsample=subsample)
        df_pgd = self.pgd_attack(df, epsilon=epsilon, subsample=subsample)
        df_inf = self.model_informed_attack(df, epsilon=max(0.05, epsilon * 0.8), subsample=subsample)

        preds_fgsm = self.detector.detect_anomalies_simple(df_fgsm, self.feature_columns)
        preds_pgd = self.detector.detect_anomalies_simple(df_pgd, self.feature_columns)
        preds_inf = self.detector.detect_anomalies_simple(df_inf, self.feature_columns)

        attack_pos = attack_rows.astype(bool)

        def _rate(preds: np.ndarray) -> float:
            if not np.any(attack_pos):
                return float(np.mean(preds))
            return float(np.mean(preds[attack_pos]))

        X0 = self._scaled_matrix(df)
        Xf = self._scaled_matrix(df_fgsm)

        flip_fgsm = 0.0
        flip_pgd = 0.0
        if np.any(attack_pos):
            flip_fgsm = float(
                np.mean((preds_clean[attack_pos] == 1) & (preds_fgsm[attack_pos] == 0))
            )
            flip_pgd = float(
                np.mean((preds_clean[attack_pos] == 1) & (preds_pgd[attack_pos] == 0))
            )

        pert_fgsm_row = np.max(np.abs(Xf - X0), axis=1)
        pert_pgd_row = np.max(np.abs(self._scaled_matrix(df_pgd) - X0), axis=1)
        sub_fgsm = np.any(Xf != X0, axis=1)
        mean_linf_fgsm = float(np.mean(pert_fgsm_row[sub_fgsm])) if np.any(sub_fgsm) else 0.0
        metrics = RobustnessMetrics(
            detection_rate_clean=_rate(preds_clean),
            detection_rate_fgsm=_rate(preds_fgsm),
            detection_rate_pgd=_rate(preds_pgd),
            detection_rate_informed=_rate(preds_inf),
            attack_flip_rate_fgsm=flip_fgsm,
            attack_flip_rate_pgd=flip_pgd,
            mean_linf_fgsm=mean_linf_fgsm,
            mean_linf_pgd=float(np.mean(pert_pgd_row)),
        )

        return {
            "metrics": {
                "detection_rate_clean_under_attack_windows": metrics.detection_rate_clean,
                "detection_rate_fgsm_under_attack_windows": metrics.detection_rate_fgsm,
                "detection_rate_pgd_under_attack_windows": metrics.detection_rate_pgd,
                "detection_rate_model_informed_under_attack_windows": metrics.detection_rate_informed,
                "evasion_flip_rate_fgsm": metrics.attack_flip_rate_fgsm,
                "evasion_flip_rate_pgd": metrics.attack_flip_rate_pgd,
                "epsilon_l_inf_scaled_space": epsilon,
                "mean_l_inf_perturbation_fgsm_approx": metrics.mean_linf_fgsm,
            },
            "subsample_rows": subsample,
        }


def save_adversarial_report(payload: Dict, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
