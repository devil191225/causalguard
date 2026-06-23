"""
Demo script to show adaptive learning in action.
This script runs the detector multiple times to show how it improves.
"""

import os
import sys
import numpy as np
import pandas as pd
from causal_detector import SWaTCausalDetector

def create_varying_data(n_samples=1000, run_number=0):
    """
    Create data that varies slightly between runs to simulate real-world conditions.
    """
    np.random.seed(42 + run_number)  # Different seed for each run
    
    # Generate correlated features with slight variations
    base_features = np.random.randn(n_samples, 5)
    
    # Create causal relationships with some variation
    features = np.zeros((n_samples, 15))
    features[:, 0] = base_features[:, 0]  # Independent variable
    features[:, 1] = (0.7 + 0.1 * np.sin(run_number)) * features[:, 0] + 0.3 * base_features[:, 1]
    features[:, 2] = 0.5 * features[:, 0] + 0.4 * features[:, 1] + 0.1 * base_features[:, 2]
    features[:, 3] = 0.6 * features[:, 1] + 0.4 * base_features[:, 3]
    features[:, 4] = 0.3 * features[:, 2] + 0.7 * base_features[:, 4]
    
    # Add more independent features with slight variations
    for i in range(5, 15):
        features[:, i] = base_features[:, (i-5) % 5] + 0.1 * np.random.randn(n_samples)
    
    # Create normal data
    normal_data = pd.DataFrame(features, columns=[f'Feature_{i}' for i in range(15)])
    normal_data['Label'] = 0
    normal_data['Timestamp'] = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
    
    # Create attack data with varying anomaly patterns
    attack_data = normal_data.copy()
    attack_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    
    for idx in attack_indices:
        # Vary anomaly intensity based on run number
        intensity = 1.0 + 0.2 * np.sin(run_number)
        
        if np.random.random() < 0.3:
            attack_data.iloc[idx, 0] += intensity * 3 * np.random.randn()
        if np.random.random() < 0.3:
            attack_data.iloc[idx, 2] += intensity * 2 * np.random.randn()
        if np.random.random() < 0.2:
            attack_data.iloc[idx, 4] += intensity * 4 * np.random.randn()
    
    attack_data['Label'] = 0
    attack_data.loc[attack_indices, 'Label'] = 1
    
    return normal_data, attack_data

def run_adaptive_demo():
    """
    Run the adaptive learning demo.
    """
    print("🤖 ADAPTIVE LEARNING DEMO")
    print("="*50)
    print("This demo shows how the model improves over multiple runs")
    print("Each run uses slightly different data to simulate real-world conditions")
    print()
    
    # Initialize detector with adaptive learning
    detector = SWaTCausalDetector(adaptive_learning=True)
    
    performance_history = []
    
    # Run multiple iterations
    for run in range(5):
        print(f"🔄 RUN {run + 1}/5")
        print("-" * 30)
        
        # Create varying data for this run
        normal_data, attack_data = create_varying_data(n_samples=2000, run_number=run)
        
        # Save data temporarily for this run
        os.makedirs('data', exist_ok=True)
        normal_data.to_csv('data/SWaT_Dataset_Normal_v1.csv', index=False)
        attack_data.to_csv('data/SWaT_Dataset_Attack_v0.csv', index=False)
        
        # Run the pipeline
        results = detector.run_full_pipeline()
        
        # Store performance metrics
        metrics = results['metrics']['overall']
        performance_history.append({
            'run': run + 1,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
        
        print(f"Run {run + 1} Results:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1']:.3f}")
        print()
    
    # Analyze performance trends
    print("📊 PERFORMANCE TREND ANALYSIS")
    print("="*50)
    
    precisions = [p['precision'] for p in performance_history]
    recalls = [p['recall'] for p in performance_history]
    f1_scores = [p['f1'] for p in performance_history]
    
    print(f"Precision: {precisions[0]:.3f} → {precisions[-1]:.3f} (Δ: {precisions[-1] - precisions[0]:+.3f})")
    print(f"Recall:    {recalls[0]:.3f} → {recalls[-1]:.3f} (Δ: {recalls[-1] - recalls[0]:+.3f})")
    print(f"F1-Score:  {f1_scores[0]:.3f} → {f1_scores[-1]:.3f} (Δ: {f1_scores[-1] - f1_scores[0]:+.3f})")
    
    # Calculate improvement trends
    precision_trend = np.polyfit(range(len(precisions)), precisions, 1)[0]
    recall_trend = np.polyfit(range(len(recalls)), recalls, 1)[0]
    f1_trend = np.polyfit(range(len(f1_scores)), f1_scores, 1)[0]
    
    print()
    print("📈 TREND ANALYSIS:")
    if precision_trend > 0.01:
        print("✓ Precision is improving over time")
    elif precision_trend < -0.01:
        print("⚠ Precision is declining")
    else:
        print("→ Precision is stable")
    
    if recall_trend > 0.01:
        print("✓ Recall is improving over time")
    elif recall_trend < -0.01:
        print("⚠ Recall is declining")
    else:
        print("→ Recall is stable")
    
    if f1_trend > 0.01:
        print("✓ F1-Score is improving over time")
    elif f1_trend < -0.01:
        print("⚠ F1-Score is declining")
    else:
        print("→ F1-Score is stable")
    
    print()
    print("🎯 ADAPTIVE LEARNING SUMMARY:")
    print(f"• Model has learned from {len(performance_history)} runs")
    print(f"• Performance history saved for future runs")
    print(f"• Models will continue to improve with more data")
    print(f"• Adaptive state saved in outputs/adaptive/")
    
    # Clean up temporary data
    if os.path.exists('data/SWaT_Dataset_Normal_v1.csv'):
        os.remove('data/SWaT_Dataset_Normal_v1.csv')
    if os.path.exists('data/SWaT_Dataset_Attack_v0.csv'):
        os.remove('data/SWaT_Dataset_Attack_v0.csv')

if __name__ == "__main__":
    run_adaptive_demo()


