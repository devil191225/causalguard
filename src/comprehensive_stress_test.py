#!/usr/bin/env python3
"""
 COMPREHENSIVE STRESS TEST SUITE FOR CAUSALGUARD
============================================================
Tests the causal anomaly detection system under extreme conditions:
1. All edge cases (gradual drift, intermittent, mimicry, cascade, adversarial)
2. Large-scale data (10K+ samples)
3. Extreme noise (up to 80% corruption)
4. Real-time performance benchmarking
5. Causal graph validation
6. Adversarial robustness

Author: Aditya Srikar Konduri
Date: October 19, 2025
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Import the causal detector
sys.path.insert(0, str(Path(__file__).parent))
from causal_detector import SWaTCausalDetector

class StressTestSuite:
    """Comprehensive stress testing for CausalGuard system."""
    
    def __init__(self):
        self.results = {
            'edge_cases': {},
            'noise_tests': {},
            'performance_tests': {},
            'causal_validation': {}
        }
        self.output_dir = Path('outputs/stress_tests')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_all_tests(self):
        """Run all stress tests."""
        print("=" * 80)
        print("CAUSALGUARD COMPREHENSIVE STRESS TEST SUITE")
        print("=" * 80)
        print()
        
        # Test 1: Edge cases
        print("=" * 80)
        print("TEST 1: EDGE CASE SCENARIOS")
        print("=" * 80)
        self.test_edge_cases()
        
        # Test 2: Noise robustness
        print("\n" + "=" * 80)
        print("TEST 2: NOISE ROBUSTNESS")
        print("=" * 80)
        self.test_noise_robustness()
        
        # Test 3: Performance benchmarking
        print("\n" + "=" * 80)
        print("TEST 3: REAL-TIME PERFORMANCE")
        print("=" * 80)
        self.test_performance()
        
        # Test 4: Causal graph validation
        print("\n" + "=" * 80)
        print("TEST 4: CAUSAL GRAPH VALIDATION")
        print("=" * 80)
        self.test_causal_graph_validation()
        
        # Generate final report
        self.generate_report()
        
    def test_edge_cases(self):
        """Test all edge case scenarios."""
        edge_case_dir = Path('outputs/edge_cases/data')
        
        if not edge_case_dir.exists():
            print(" Edge case data not found. Run edge_case_tester.py first.")
            return
        
        edge_cases = ['gradual_drift', 'intermittent', 'mimicry', 'cascade', 'adversarial']
        
        for case_name in edge_cases:
            print(f"\n Testing: {case_name}")
            print("-" * 60)
            
            data_file = edge_case_dir / f"{case_name}_data.csv"
            if not data_file.exists():
                print(f"   Data file not found: {data_file}")
                continue
            
            try:
                # Load data
                data = pd.read_csv(data_file)
                
                # Split into normal and attack
                if 'attack' in data.columns:
                    normal_mask = (data['attack'] == 'Normal')
                else:
                    # For edge cases, first 20% is normal, rest is attack
                    split_idx = int(len(data) * 0.2)
                    normal_mask = pd.Series([True] * split_idx + [False] * (len(data) - split_idx))
                
                normal_data = data[normal_mask].copy()
                attack_data = data[~normal_mask].copy()
                
                print(f"  - Normal samples: {len(normal_data)}")
                print(f"  - Attack samples: {len(attack_data)}")
                
                if len(normal_data) < 100 or len(attack_data) < 10:
                    print(f"   Insufficient data for testing")
                    continue
                
                # Initialize detector (without adaptive learning for consistent testing)
                detector = SWaTCausalDetector(adaptive_learning=False)
                
                # Manually preprocess data
                start_time = time.time()
                normal_proc = detector._preprocess_dataframe(normal_data, is_attack=False)
                attack_proc = detector._preprocess_dataframe(attack_data, is_attack=True)
                
                # Get feature columns (exclude non-numeric)
                feature_cols = [col for col in normal_proc.columns 
                               if col not in ['attack', 'time', 'Label', 'Timestamp'] 
                               and normal_proc[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
                
                # Learn causal graph
                detector.learn_causal_graph(normal_proc, feature_cols)
                
                # Train structural models
                detector.train_structural_models(normal_proc, feature_cols)
                
                # Detect anomalies
                predictions = detector.detect_anomalies(attack_proc, feature_cols)
                total_time = time.time() - start_time
                
                # Evaluate
                true_labels = np.ones(len(attack_proc))
                precision = precision_score(true_labels, predictions)
                recall = recall_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions)
                
                # Store results
                self.results['edge_cases'][case_name] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'causal_edges': len(detector.causal_graph.edges) if hasattr(detector.causal_graph, 'edges') else 0,
                    'test_samples': len(attack_proc),
                    'execution_time_seconds': float(total_time)
                }
                
                print(f"   Precision: {precision:.3f}")
                print(f"   Recall: {recall:.3f}")
                print(f"   F1-Score: {f1:.3f}")
                print(f"   Causal edges: {self.results['edge_cases'][case_name]['causal_edges']}")
                print(f"   Execution time: {total_time:.2f}s")
                
                # Pass/Fail criteria
                if f1 >= 0.6 and recall >= 0.7:
                    print(f"   PASSED (F10.6, Recall0.7)")
                else:
                    print(f"   WARNING: Performance below target")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
                self.results['edge_cases'][case_name] = {'error': str(e)}
    
    def test_noise_robustness(self):
        """Test robustness to various noise levels."""
        noise_levels = [0.1, 0.2, 0.3, 0.5, 0.8]  # 10% to 80% noise
        
        # Load base data
        normal_file = Path('data/SWaT_Dataset_Normal_v1.csv')
        attack_file = Path('data/SWaT_Dataset_Attack_v0.csv')
        
        if not (normal_file.exists() and attack_file.exists()):
            print(" Base data files not found")
            return
        
        normal_data = pd.read_csv(normal_file)
        attack_data = pd.read_csv(attack_file)
        
        for noise_level in noise_levels:
            print(f"\n Testing with {int(noise_level*100)}% noise corruption")
            print("-" * 60)
            
            try:
                # Add noise to data
                normal_noisy = self._add_noise(normal_data.copy(), noise_level)
                attack_noisy = self._add_noise(attack_data.copy(), noise_level)
                
                # Initialize detector
                detector = SWaTCausalDetector(adaptive_learning=False)
                
                # Process
                start_time = time.time()
                normal_proc = detector._preprocess_dataframe(normal_noisy, is_attack=False)
                attack_proc = detector._preprocess_dataframe(attack_noisy, is_attack=True)
                
                feature_cols = [col for col in normal_proc.columns 
                               if col not in ['attack', 'time', 'Label', 'Timestamp'] 
                               and normal_proc[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
                
                # Learn and train
                detector.learn_causal_graph(normal_proc, feature_cols)
                detector.train_structural_models(normal_proc, feature_cols)
                
                # Detect
                predictions = detector.detect_anomalies(attack_proc, feature_cols)
                total_time = time.time() - start_time
                
                # Evaluate
                true_labels = np.ones(len(attack_proc))
                precision = precision_score(true_labels, predictions)
                recall = recall_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions)
                
                # Store results
                self.results['noise_tests'][f'{int(noise_level*100)}%'] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'execution_time_seconds': float(total_time)
                }
                
                print(f"   Precision: {precision:.3f}")
                print(f"   Recall: {recall:.3f}")
                print(f"   F1-Score: {f1:.3f}")
                
                # Pass/Fail
                if noise_level <= 0.3 and f1 >= 0.5:
                    print(f"   PASSED (Acceptable for 30% noise)")
                elif noise_level <= 0.5 and f1 >= 0.3:
                    print(f"   PASSED (Acceptable for 50% noise)")
                else:
                    print(f"   Performance degraded as expected")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
                self.results['noise_tests'][f'{int(noise_level*100)}%'] = {'error': str(e)}
    
    def _add_noise(self, data, noise_level):
        """Add Gaussian noise to numeric columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['attack', 'time']:
                noise = np.random.normal(0, noise_level * data[col].std(), len(data))
                data[col] = data[col] + noise
        
        return data
    
    def test_performance(self):
        """Benchmark real-time inference performance."""
        print("\n Benchmarking inference latency...")
        print("-" * 60)
        
        # Load data
        normal_file = Path('data/SWaT_Dataset_Normal_v1.csv')
        attack_file = Path('data/SWaT_Dataset_Attack_v0.csv')
        
        if not (normal_file.exists() and attack_file.exists()):
            print(" Data files not found")
            return
        
        normal_data = pd.read_csv(normal_file)
        attack_data = pd.read_csv(attack_file)
        
        try:
            # Initialize and train detector
            detector = SWaTCausalDetector(adaptive_learning=False)
            normal_proc = detector._preprocess_dataframe(normal_data, is_attack=False)
            attack_proc = detector._preprocess_dataframe(attack_data, is_attack=True)
            
            feature_cols = [col for col in normal_proc.columns 
                           if col not in ['attack', 'time', 'Label', 'Timestamp'] 
                           and normal_proc[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
            
            print("  - Training models...")
            train_start = time.time()
            detector.learn_causal_graph(normal_proc, feature_cols)
            detector.train_structural_models(normal_proc, feature_cols)
            train_time = time.time() - train_start
            
            print(f"   Training time: {train_time:.2f}s")
            
            # Benchmark inference on individual samples
            print("  - Benchmarking single-sample inference...")
            latencies = []
            
            for i in range(min(100, len(attack_proc))):
                sample = attack_proc.iloc[i:i+1]
                
                start = time.perf_counter()
                pred = detector.detect_anomalies(sample, feature_cols)
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)  # Convert to ms
            
            latencies = np.array(latencies)
            
            self.results['performance_tests'] = {
                'training_time_seconds': float(train_time),
                'inference_latency_ms': {
                    'mean': float(np.mean(latencies)),
                    'median': float(np.median(latencies)),
                    'p95': float(np.percentile(latencies, 95)),
                    'p99': float(np.percentile(latencies, 99)),
                    'max': float(np.max(latencies))
                },
                'throughput_samples_per_second': float(1000 / np.mean(latencies))
            }
            
            print(f"   Mean latency: {np.mean(latencies):.2f}ms")
            print(f"   Median latency: {np.median(latencies):.2f}ms")
            print(f"   P95 latency: {np.percentile(latencies, 95):.2f}ms")
            print(f"   P99 latency: {np.percentile(latencies, 99):.2f}ms")
            print(f"   Max latency: {np.max(latencies):.2f}ms")
            print(f"   Throughput: {1000/np.mean(latencies):.0f} samples/second")
            
            # Pass/Fail criteria
            if np.mean(latencies) < 10:
                print(f"   EXCELLENT: <10ms latency (real-time capable)")
            elif np.mean(latencies) < 50:
                print(f"   GOOD: <50ms latency (near real-time)")
            elif np.mean(latencies) < 100:
                print(f"   ACCEPTABLE: <100ms latency")
            else:
                print(f"   WARNING: High latency (>100ms)")
                
        except Exception as e:
            print(f"   Error: {str(e)}")
            self.results['performance_tests'] = {'error': str(e)}
    
    def test_causal_graph_validation(self):
        """Validate causal graph against known relationships."""
        print("\n Validating causal graph structure...")
        print("-" * 60)
        
        # Create synthetic data with KNOWN causal structure
        np.random.seed(42)
        n_samples = 5000
        
        # True causal structure: P1  P2  P3  P4, P1  P3
        P1 = np.random.normal(0, 1, n_samples)
        P2 = 0.8 * P1 + np.random.normal(0, 0.2, n_samples)
        P3 = 0.7 * P2 + 0.3 * P1 + np.random.normal(0, 0.2, n_samples)
        P4 = 0.9 * P3 + np.random.normal(0, 0.2, n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4,
            'attack': ['Normal'] * n_samples
        })
        
        # Expected edges
        expected_edges = {
            ('P1', 'P2'), ('P2', 'P3'), ('P3', 'P4'), ('P1', 'P3')
        }
        
        try:
            # Initialize detector
            detector = SWaTCausalDetector(adaptive_learning=False)
            
            # Learn causal graph
            feature_cols = ['P1', 'P2', 'P3', 'P4']
            detector.learn_causal_graph(data, feature_cols)
            
            # Get learned edges
            if hasattr(detector.causal_graph, 'edges'):
                learned_edges = set(detector.causal_graph.edges)
            else:
                learned_edges = set()
            
            print(f"  - Expected edges: {expected_edges}")
            print(f"  - Learned edges: {learned_edges}")
            
            # Calculate precision/recall for edge recovery
            true_positives = len(expected_edges & learned_edges)
            false_positives = len(learned_edges - expected_edges)
            false_negatives = len(expected_edges - learned_edges)
            
            edge_precision = true_positives / len(learned_edges) if len(learned_edges) > 0 else 0
            edge_recall = true_positives / len(expected_edges) if len(expected_edges) > 0 else 0
            edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0
            
            self.results['causal_validation'] = {
                'expected_edges': list(map(str, expected_edges)),
                'learned_edges': list(map(str, learned_edges)),
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives),
                'edge_precision': float(edge_precision),
                'edge_recall': float(edge_recall),
                'edge_f1_score': float(edge_f1)
            }
            
            print(f"   Edge precision: {edge_precision:.3f}")
            print(f"   Edge recall: {edge_recall:.3f}")
            print(f"   Edge F1-score: {edge_f1:.3f}")
            print(f"   True positives: {true_positives}")
            print(f"   False positives: {false_positives}")
            print(f"   False negatives: {false_negatives}")
            
            # Pass/Fail
            if edge_f1 >= 0.7:
                print(f"   EXCELLENT: High causal structure recovery (F10.7)")
            elif edge_f1 >= 0.5:
                print(f"   GOOD: Reasonable causal structure recovery (F10.5)")
            elif edge_f1 >= 0.3:
                print(f"   ACCEPTABLE: Partial causal structure recovery (F10.3)")
            else:
                print(f"   WARNING: Low causal structure recovery")
                
        except Exception as e:
            print(f"   Error: {str(e)}")
            self.results['causal_validation'] = {'error': str(e)}
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 80)
        print(" FINAL STRESS TEST REPORT")
        print("=" * 80)
        
        # Save results to JSON
        results_file = self.output_dir / 'stress_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n Detailed results saved to: {results_file}")
        
        # Summary statistics
        print("\n SUMMARY:")
        print("-" * 80)
        
        # Edge cases summary
        if self.results['edge_cases']:
            print("\n Edge Cases:")
            edge_f1_scores = [v['f1_score'] for v in self.results['edge_cases'].values() if 'f1_score' in v]
            if edge_f1_scores:
                print(f"  - Mean F1-score: {np.mean(edge_f1_scores):.3f}")
                print(f"  - Min F1-score: {np.min(edge_f1_scores):.3f}")
                print(f"  - Max F1-score: {np.max(edge_f1_scores):.3f}")
                
                passed = sum(1 for f1 in edge_f1_scores if f1 >= 0.6)
                print(f"  - Passed cases: {passed}/{len(edge_f1_scores)}")
        
        # Noise robustness summary
        if self.results['noise_tests']:
            print("\n Noise Robustness:")
            for noise_level, metrics in self.results['noise_tests'].items():
                if 'f1_score' in metrics:
                    print(f"  - {noise_level} noise: F1={metrics['f1_score']:.3f}")
        
        # Performance summary
        if self.results['performance_tests'] and 'inference_latency_ms' in self.results['performance_tests']:
            print("\n Performance:")
            perf = self.results['performance_tests']
            print(f"  - Mean latency: {perf['inference_latency_ms']['mean']:.2f}ms")
            print(f"  - Throughput: {perf['throughput_samples_per_second']:.0f} samples/sec")
        
        # Causal validation summary
        if self.results['causal_validation'] and 'edge_f1_score' in self.results['causal_validation']:
            print("\n Causal Structure Recovery:")
            cv = self.results['causal_validation']
            print(f"  - Edge F1-score: {cv['edge_f1_score']:.3f}")
            print(f"  - True positives: {cv['true_positives']}")
            print(f"  - False positives: {cv['false_positives']}")
            print(f"  - False negatives: {cv['false_negatives']}")
        
        # Overall verdict
        print("\n" + "=" * 80)
        print(" OVERALL VERDICT:")
        print("=" * 80)
        
        # Determine pass/fail
        passed_tests = []
        failed_tests = []
        
        # Check edge cases
        if self.results['edge_cases']:
            edge_f1_scores = [v['f1_score'] for v in self.results['edge_cases'].values() if 'f1_score' in v]
            if edge_f1_scores and np.mean(edge_f1_scores) >= 0.6:
                passed_tests.append("Edge Cases")
            elif edge_f1_scores:
                failed_tests.append("Edge Cases")
        
        # Check noise robustness
        if self.results['noise_tests']:
            noise_30 = self.results['noise_tests'].get('30%', {})
            if 'f1_score' in noise_30 and noise_30['f1_score'] >= 0.5:
                passed_tests.append("Noise Robustness")
            elif 'f1_score' in noise_30:
                failed_tests.append("Noise Robustness")
        
        # Check performance
        if self.results['performance_tests'] and 'inference_latency_ms' in self.results['performance_tests']:
            if self.results['performance_tests']['inference_latency_ms']['mean'] < 100:
                passed_tests.append("Performance")
            else:
                failed_tests.append("Performance")
        
        # Check causal validation
        if self.results['causal_validation'] and 'edge_f1_score' in self.results['causal_validation']:
            if self.results['causal_validation']['edge_f1_score'] >= 0.5:
                passed_tests.append("Causal Validation")
            else:
                failed_tests.append("Causal Validation")
        
        print(f"\n PASSED: {', '.join(passed_tests) if passed_tests else 'None'}")
        print(f" FAILED: {', '.join(failed_tests) if failed_tests else 'None'}")
        
        if len(passed_tests) >= 3:
            print("\n SYSTEM IS PRODUCTION-READY!")
        elif len(passed_tests) >= 2:
            print("\n System shows promise, minor improvements needed")
        else:
            print("\n System needs significant improvements")
        
        print("\n" + "=" * 80)
        print(" STRESS TEST COMPLETE")
        print("=" * 80)

if __name__ == "__main__":
    suite = StressTestSuite()
    suite.run_all_tests()

