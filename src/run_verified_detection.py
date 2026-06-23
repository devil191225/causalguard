#!/usr/bin/env python3
"""
INTEGRATED VERIFICATION TEST
============================================================
Run causal anomaly detection with formal verification.

This demonstrates Phase 2 integration:
1. Train causal detector
2. Detect anomalies
3. Run formal verification suite
4. Generate verification certificate

Author: Aditya Srikar Konduri
Date: October 19, 2025
Status: PHASE 2 - INTEGRATED TEST
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from causal_detector import SWaTCausalDetector
from formal_verification import FormalVerifier

def main():
    parser = argparse.ArgumentParser(description="CausalGuard with Phase 2 formal verification")
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Run FGSM/PGD/model-informed robustness battery (finite-difference gradients)",
    )
    parser.add_argument(
        "--adversarial-epsilon",
        type=float,
        default=0.15,
        help="L_inf perturbation radius in standardized feature space",
    )
    parser.add_argument("--adversarial-subsample", type=int, default=512,
                        help="Max rows attacked (speed / memory tradeoff)")
    args = parser.parse_args()

    print("=" * 70)
    print("CAUSALGUARD: VERIFIED ANOMALY DETECTION")
    print("=" * 70)
    print("\nPhase 1: Causal Detection")
    print("Phase 2: Formal Verification")
    print("=" * 70)
    
    # Initialize detector
    detector = SWaTCausalDetector(adaptive_learning=False)
    
    # Load data
    print("\nLoading data...")
    try:
        normal_path = Path('data/SWaT_Dataset_Normal_v1.csv')
        attack_path = Path('data/SWaT_Dataset_Attack_v0.csv')
        
        if normal_path.exists() and attack_path.exists():
            normal_data = pd.read_csv(normal_path)
            attack_data = pd.read_csv(attack_path)
            print(f"  Normal: {len(normal_data)} samples")
            print(f"  Attack: {len(attack_data)} samples")
        else:
            print("  Using synthetic data (real data not found)")
            normal_data, attack_data = detector._create_sample_data()
    
    except Exception as e:
        print(f"  Error loading data: {e}")
        print("  Using synthetic data")
        normal_data, attack_data = detector._create_sample_data()
    
    # Preprocess
    print("\nPreprocessing...")
    normal_proc, attack_proc, feature_cols = detector.load_and_preprocess_data()
    print(f"  Features: {len(feature_cols)}")
    
    # Learn causal graph
    print("\nPhase 1.1: Learning Causal Graph...")
    detector.learn_causal_graph(normal_proc, feature_cols)
    
    # Train models
    print("\nPhase 1.2: Training Structural Models...")
    detector.train_structural_models(normal_proc, feature_cols)
    
    # Detect anomalies (get full output)
    print("\nPhase 1.3: Detecting Anomalies...")
    anomaly_scores, anomaly_flags, causal_paths = detector.detect_anomalies(
        attack_proc, feature_cols
    )
    
    # Sample-level predictions (same logic as detector; avoid redundant detect pass)
    predictions = np.any(anomaly_flags, axis=1).astype(int)
    true_labels = attack_proc['Label'].values
    
    print(f"  Detected: {np.sum(predictions)} / {len(predictions)} samples")
    
    # Compute basic metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)
    
    print(f"\n  Detection Performance:")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall: {recall:.3f}")
    print(f"    F1-Score: {f1:.3f}")
    
    # Initialize verifier
    print("\n" + "=" * 70)
    print("PHASE 2: FORMAL VERIFICATION")
    print("=" * 70)
    
    verifier = FormalVerifier()
    
    # Run complete verification suite
    verification_results = verifier.run_verification_suite(
        detector=detector,
        test_data=attack_proc,
        true_labels=true_labels,
        predictions=predictions,
        feature_cols=feature_cols,
        anomaly_scores=anomaly_scores,
        anomaly_flags=anomaly_flags,
        normal_reference_data=normal_proc,
        run_adversarial=args.adversarial,
        adversarial_epsilon=args.adversarial_epsilon,
        adversarial_subsample=args.adversarial_subsample,
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("VERIFICATION CERTIFICATE")
    print("=" * 70)
    
    props_tested = verification_results['properties_tested']
    props_verified = verification_results['properties_verified']
    
    print(f"\nProperties Tested: {props_tested}")
    print(f"Properties Verified: {props_verified}")
    print(f"Verification Rate: {props_verified/props_tested*100:.1f}%")
    
    print("\nProperty Status:")
    for prop_name, prop_result in verification_results['results'].items():
        status = "[PASS] VERIFIED" if prop_result['verified'] else "[FAIL] VIOLATED"
        print(f"  {prop_name}: {status}")
    
    # Generate certificate
    if verification_results.get("adversarial_summary") and verification_results["adversarial_summary"].get(
        "metrics"
    ):
        print("\nAdversarial summary (attack-labeled rows):")
        for k, v in verification_results["adversarial_summary"]["metrics"].items():
            print(f"  {k}: {v}")

    if props_verified == props_tested:
        print("\n" + "=" * 70)
        print("CERTIFICATION: PASS")
        print("=" * 70)
        print("All formal properties verified.")
        print("System meets safety requirements for deployment.")
    else:
        print("\n" + "=" * 70)
        print("CERTIFICATION: CONDITIONAL")
        print("=" * 70)
        print(f"{props_tested - props_verified} properties failed verification.")
        print("Review failures before deployment.")
    
    print(f"\nResults saved to: outputs/verification/")
    print("=" * 70)

if __name__ == "__main__":
    main()



