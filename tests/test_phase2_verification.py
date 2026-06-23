"""
Smoke tests for Phase 2 formal verification and adversarial helpers.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def test_z_multiplier():
    from formal_verification import _z_multiplier

    assert abs(_z_multiplier(0.95) - 1.96) < 1e-9
    assert _z_multiplier(0.90) < _z_multiplier(0.99)


def test_causal_graph_to_digraph():

    class G:
        edges = [("a", "b"), ("b", "c")]
        nodes = {"a", "b", "c"}

    class Det:
        causal_graph = G()

    from formal_verification import causal_graph_to_digraph

    nxg = causal_graph_to_digraph(Det())
    assert nxg.number_of_edges() == 2
    assert nxg.has_edge("a", "b")


def test_verification_suite_properties_count():
    from formal_verification import FormalVerifier

    class EG:
        edges = []

    class Det:
        causal_graph = EG()
        models = {}

    det = Det()
    det.scaler = StandardScaler()

    feats = ["f0", "f1"]
    n = 40
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    labs = np.zeros(n, dtype=int)
    preds = np.zeros(n, dtype=int)

    for f in feats:
        det.models[f] = {"type": "statistical", "mean": 0.0, "std": 1.0, "threshold": 3.0, "parents": []}

    det._normal_training_scaled = det.scaler.fit_transform(df.values)
    det._training_feature_columns = list(feats)

    # Nearly monotone increasing max and sum of residuals for Property 5
    t = np.linspace(0.1, 2.0, n)
    anomaly_scores = np.column_stack([t, 0.2 * t + 1e-4 * rng.standard_normal(n)])

    with tempfile.TemporaryDirectory() as tmp:
        vf = FormalVerifier(output_dir=tmp)
        out = vf.run_verification_suite(
            det,
            df,
            labs,
            preds,
            feats,
            anomaly_scores=anomaly_scores,
            anomaly_flags=np.zeros((n, 2), dtype=bool),
            normal_reference_data=df,
            run_adversarial=False,
        )

        assert out["properties_tested"] == 5
        assert (Path(tmp) / "verification_proof_appendix.md").is_file()


def test_adversarial_tester_import():
    from adversarial_testing import AdversarialTester

    assert AdversarialTester is not None
