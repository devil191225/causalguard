#!/usr/bin/env python3
"""
TEST ON REALISTIC ATTACK SCENARIOS
============================================================
Tests the causal detector on realistic, challenging synthetic attacks.

BRUTALLY HONEST: These are NOT real attacks. They are synthetic scenarios
designed to be realistic based on ICS attack literature.

Author: Aditya Srikar Konduri
Date: October 19, 2025
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from causal_detector import SWaTCausalDetector

def test_on_realistic_attack(attack_name):
    """Test detector on a specific realistic attack scenario."""
    print(f"\n{'=' * 70}")
    print(f"TESTING: {attack_name}")
    print(f"{'=' * 70}")
    
    # Load data
    normal_path = Path(f'data/realistic_attacks/{attack_name}_normal.csv')
    attack_path = Path(f'data/realistic_attacks/{attack_name}_attack.csv')
    
    if not (normal_path.exists() and attack_path.exists()):
        print(f"ERROR: Data files not found")
        return None
    
    normal_df = pd.read_csv(normal_path)
    attack_df = pd.read_csv(attack_path)
    
    print(f"Normal samples: {len(normal_df)}")
    print(f"Attack samples: {len(attack_df)}, Anomalies: {attack_df['Label'].sum()}")
    
    try:
        # Initialize detector (no adaptive learning for consistent testing)
        detector = SWaTCausalDetector(adaptive_learning=False)
        
        # Preprocess
        normal_proc = detector._preprocess_dataframe(normal_df, is_attack=False)
        attack_proc = detector._preprocess_dataframe(attack_df, is_attack=True)
        
        # Get features (exclude Label)
        feature_cols = [col for col in normal_proc.columns 
                       if col not in ['Label', 'Timestamp', 'time'] and 
                       'attack' not in col.lower()]
        
        print(f"Features: {len(feature_cols)}")
        
        # Learn causal graph
        detector.learn_causal_graph(normal_proc, feature_cols)
        
        # Train models
        detector.train_structural_models(normal_proc, feature_cols)
        
        # Detect (use simplified method for 1D predictions)
        predictions = detector.detect_anomalies_simple(attack_proc, feature_cols)
        
        # Evaluate
        true_labels = attack_proc['Label'].values
        
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        cm = confusion_matrix(true_labels, predictions)
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results = {
            'attack_name': attack_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fpr': fpr,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'causal_edges': len(detector.causal_graph.edges) if hasattr(detector.causal_graph, 'edges') else 0
        }
        
        print(f"\nRESULTS:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  FPR: {fpr:.3f}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  Causal Edges: {results['causal_edges']}")
        
        # Assessment
        if f1 >= 0.70 and recall >= 0.80:
            print(f"  PASSED: Excellent detection")
        elif f1 >= 0.60 and recall >= 0.70:
            print(f"  ACCEPTABLE: Good detection")
        elif f1 >= 0.40:
            print(f"  POOR: Low performance")
        else:
            print(f"  FAILED: Detection failed")
        
        return results
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 70)
    print("CAUSALGUARD: REALISTIC ATTACK TESTING")
    print("=" * 70)
    print("\nBRUTALLY HONEST DISCLAIMER:")
    print("These are SYNTHETIC attacks designed to be realistic.")
    print("Performance on real ICS attacks may be significantly different.")
    print("=" * 70)
    
    attacks = [
        'stealthy_setpoint_change',
        'sensor_spoofing',
        'slow_ramp_attack',
        'intermittent_pulse',
        'model_informed_attack'
    ]
    
    all_results = []
    for attack in attacks:
        result = test_on_realistic_attack(attack)
        if result:
            all_results.append(result)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    
    if all_results:
        avg_f1 = np.mean([r['f1_score'] for r in all_results])
        avg_recall = np.mean([r['recall'] for r in all_results])
        avg_precision = np.mean([r['precision'] for r in all_results])
        avg_fpr = np.mean([r['fpr'] for r in all_results])
        
        print(f"\nAVERAGE PERFORMANCE:")
        print(f"  Precision: {avg_precision:.3f}")
        print(f"  Recall: {avg_recall:.3f}")
        print(f"  F1-Score: {avg_f1:.3f}")
        print(f"  FPR: {avg_fpr:.3f}")
        
        print(f"\nPER-ATTACK BREAKDOWN:")
        for r in all_results:
            print(f"  {r['attack_name']}: F1={r['f1_score']:.3f}, Recall={r['recall']:.3f}, FPR={r['fpr']:.3f}")
        
        print(f"\n{'=' * 70}")
        print("HONEST ASSESSMENT:")
        print(f"{'=' * 70}")
        
        if avg_f1 >= 0.70 and avg_recall >= 0.80 and avg_fpr <= 0.10:
            print("EXCELLENT: System performs well on synthetic realistic attacks")
            print("NOTE: Real-world performance may be lower due to noise,")
            print("      sensor drift, and truly adversarial attackers.")
        elif avg_f1 >= 0.60:
            print("GOOD: System shows promise but needs improvement")
            print("NOTE: This is synthetic data. Real attacks will be harder.")
        else:
            print("NEEDS IMPROVEMENT: Performance below target")
            print("Current system not ready for deployment.")
        
        print(f"\n{'=' * 70}")

if __name__ == "__main__":
    main()

