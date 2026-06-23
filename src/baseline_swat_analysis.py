"""
Baseline SWaT Analysis Script
============================

This script provides comprehensive analysis capabilities for the SWaT dataset,
including data exploration, baseline anomaly detection methods, and comparison
with the SCM-based approach.

Author: Aditya Srikar Konduri
Date: 10/19/2025
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm
import json
from datetime import datetime
import argparse

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from causal_detector import SWaTCausalDetector

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SWaTBaselineAnalyzer:
    """
    Baseline analyzer for SWaT dataset with multiple anomaly detection methods.
    """
    
    def __init__(self, data_dir='data', output_dir='outputs'):
        """
        Initialize the analyzer.
        
        Args:
            data_dir (str): Directory containing SWaT data files
            output_dir (str): Directory to save outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.baseline_models = {}
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'baseline_analysis'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'baseline_analysis', 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'baseline_analysis', 'results'), exist_ok=True)
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess SWaT dataset.
        
        Returns:
            tuple: (normal_data, attack_data, feature_columns)
        """
        print("Loading SWaT dataset for baseline analysis...")
        
        # Load normal data
        normal_file = os.path.join(self.data_dir, 'SWaT_Dataset_Normal_v1.csv')
        attack_file = os.path.join(self.data_dir, 'SWaT_Dataset_Attack_v0.csv')
        
        if not os.path.exists(normal_file):
            print(f"Warning: {normal_file} not found. Creating sample data for demonstration.")
            return self._create_sample_data()
        
        # Load CSV files
        normal_data = pd.read_csv(normal_file)
        attack_data = pd.read_csv(attack_file)
        
        print(f"Normal data shape: {normal_data.shape}")
        print(f"Attack data shape: {attack_data.shape}")
        
        # Preprocess data
        normal_data = self._preprocess_dataframe(normal_data, is_attack=False)
        attack_data = self._preprocess_dataframe(attack_data, is_attack=True)
        
        # Get feature names (exclude timestamp and label columns)
        feature_columns = [col for col in normal_data.columns 
                          if col not in ['Timestamp', 'Normal/Attack', 'Label']]
        
        print(f"Features: {len(feature_columns)}")
        print(f"Feature names: {feature_columns[:10]}...")  # Show first 10
        
        return normal_data, attack_data, feature_columns
    
    def _create_sample_data(self):
        """
        Create sample data for demonstration when real SWaT data is not available.
        """
        print("Creating sample data for demonstration...")
        
        # Create synthetic SWaT-like data
        np.random.seed(42)
        n_samples = 10000
        
        # Generate correlated features (simulating industrial process variables)
        base_features = np.random.randn(n_samples, 5)
        
        # Create causal relationships
        features = np.zeros((n_samples, 15))
        features[:, 0] = base_features[:, 0]  # Independent variable
        features[:, 1] = 0.7 * features[:, 0] + 0.3 * base_features[:, 1]  # Depends on 0
        features[:, 2] = 0.5 * features[:, 0] + 0.4 * features[:, 1] + 0.1 * base_features[:, 2]  # Depends on 0,1
        features[:, 3] = 0.6 * features[:, 1] + 0.4 * base_features[:, 3]  # Depends on 1
        features[:, 4] = 0.3 * features[:, 2] + 0.7 * base_features[:, 4]  # Depends on 2
        
        # Add more independent features
        for i in range(5, 15):
            features[:, i] = base_features[:, (i-5) % 5] + 0.1 * np.random.randn(n_samples)
        
        # Create normal data
        normal_data = pd.DataFrame(features, columns=[f'Feature_{i}' for i in range(15)])
        normal_data['Label'] = 0
        normal_data['Timestamp'] = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
        
        # Create attack data with anomalies
        attack_data = normal_data.copy()
        attack_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        
        for idx in attack_indices:
            # Introduce anomalies in different features
            if np.random.random() < 0.3:
                attack_data.iloc[idx, 0] += 3 * np.random.randn()  # Large deviation
            if np.random.random() < 0.3:
                attack_data.iloc[idx, 2] += 2 * np.random.randn()  # Medium deviation
            if np.random.random() < 0.2:
                attack_data.iloc[idx, 4] += 4 * np.random.randn()  # Large deviation
        
        attack_data['Label'] = 0
        attack_data.loc[attack_indices, 'Label'] = 1
        
        feature_columns = [f'Feature_{i}' for i in range(15)]
        
        return normal_data, attack_data, feature_columns
    
    def _preprocess_dataframe(self, df, is_attack=False):
        """
        Preprocess a single dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            is_attack (bool): Whether this is attack data
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Convert timestamp if present
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Handle label column
        if 'Normal/Attack' in df.columns:
            df['Label'] = (df['Normal/Attack'] == 'Attack').astype(int)
        elif 'Label' not in df.columns and is_attack:
            # If no label column in attack data, assume all are attacks
            df['Label'] = 1
        elif 'Label' not in df.columns:
            df['Label'] = 0
        
        return df
    
    def explore_data(self, normal_data, attack_data, feature_columns):
        """
        Perform comprehensive data exploration.
        
        Args:
            normal_data (pd.DataFrame): Normal operation data
            attack_data (pd.DataFrame): Attack data
            feature_columns (list): List of feature column names
        """
        print("Performing data exploration...")
        
        # Basic statistics
        print("\n" + "="*50)
        print("DATA EXPLORATION SUMMARY")
        print("="*50)
        
        print(f"Normal data shape: {normal_data.shape}")
        print(f"Attack data shape: {attack_data.shape}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Normal samples: {len(normal_data)}")
        print(f"Attack samples: {len(attack_data)}")
        print(f"Attack percentage: {len(attack_data) / (len(normal_data) + len(attack_data)) * 100:.2f}%")
        
        # Feature statistics
        print("\nFeature Statistics (Normal Data):")
        normal_stats = normal_data[feature_columns].describe()
        print(normal_stats)
        
        print("\nFeature Statistics (Attack Data):")
        attack_stats = attack_data[feature_columns].describe()
        print(attack_stats)
        
        # Create visualizations
        self._create_exploration_plots(normal_data, attack_data, feature_columns)
        
        # Save exploration results
        exploration_results = {
            'normal_data_shape': normal_data.shape,
            'attack_data_shape': attack_data.shape,
            'feature_count': len(feature_columns),
            'normal_samples': len(normal_data),
            'attack_samples': len(attack_data),
            'attack_percentage': len(attack_data) / (len(normal_data) + len(attack_data)) * 100,
            'normal_stats': normal_stats.to_dict(),
            'attack_stats': attack_stats.to_dict()
        }
        
        results_path = os.path.join(self.output_dir, 'baseline_analysis', 'results', 'exploration_results.json')
        with open(results_path, 'w') as f:
            json.dump(exploration_results, f, indent=2, default=str)
        
        print(f"\nExploration results saved to: {results_path}")
    
    def _create_exploration_plots(self, normal_data, attack_data, feature_columns):
        """
        Create data exploration visualizations.
        """
        print("Creating exploration plots...")
        
        # 1. Feature distribution comparison
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_columns[:15]):  # Show first 15 features
            if i < len(axes):
                # Normal data
                axes[i].hist(normal_data[feature], bins=50, alpha=0.7, label='Normal', color='blue')
                # Attack data
                axes[i].hist(attack_data[feature], bins=50, alpha=0.7, label='Attack', color='red')
                axes[i].set_title(f'{feature}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(feature_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Distribution Comparison (Normal vs Attack)', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'baseline_analysis', 'plots', 'feature_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = normal_data[feature_columns].corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix (Normal Data)', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'baseline_analysis', 'plots', 'correlation_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. PCA visualization
        self._create_pca_plot(normal_data, attack_data, feature_columns)
        
        print(f"Exploration plots saved to: {os.path.join(self.output_dir, 'baseline_analysis', 'plots')}")
    
    def _create_pca_plot(self, normal_data, attack_data, feature_columns):
        """
        Create PCA visualization of the data.
        """
        # Combine data for PCA
        combined_data = pd.concat([normal_data[feature_columns], attack_data[feature_columns]], ignore_index=True)
        combined_labels = np.concatenate([np.zeros(len(normal_data)), np.ones(len(attack_data))])
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_data)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot normal data
        normal_mask = combined_labels == 0
        plt.scatter(pca_result[normal_mask, 0], pca_result[normal_mask, 1], 
                   c='blue', alpha=0.6, label='Normal', s=20)
        
        # Plot attack data
        attack_mask = combined_labels == 1
        plt.scatter(pca_result[attack_mask, 0], pca_result[attack_mask, 1], 
                   c='red', alpha=0.6, label='Attack', s=20)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Visualization of Normal vs Attack Data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'baseline_analysis', 'plots', 'pca_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_baseline_models(self, normal_data, feature_columns):
        """
        Train baseline anomaly detection models.
        
        Args:
            normal_data (pd.DataFrame): Normal operation data
            feature_columns (list): List of feature column names
        """
        print("Training baseline anomaly detection models...")
        
        # Prepare data
        X_normal = normal_data[feature_columns].values
        X_scaled = self.scaler.fit_transform(X_normal)
        
        # 1. Isolation Forest
        print("Training Isolation Forest...")
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X_scaled)
        self.baseline_models['isolation_forest'] = iso_forest
        
        # 2. One-Class SVM
        print("Training One-Class SVM...")
        oc_svm = OneClassSVM(nu=0.1, kernel='rbf')
        oc_svm.fit(X_scaled)
        self.baseline_models['one_class_svm'] = oc_svm
        
        # 3. DBSCAN
        print("Training DBSCAN...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(X_scaled)
        self.baseline_models['dbscan'] = dbscan
        
        # 4. Statistical method (Z-score based)
        print("Training Statistical method...")
        self.baseline_models['statistical'] = {
            'means': np.mean(X_scaled, axis=0),
            'stds': np.std(X_scaled, axis=0),
            'threshold': 3.0  # 3-sigma rule
        }
        
        # 5. Neural Network Autoencoder
        print("Training Neural Network Autoencoder...")
        try:
            from sklearn.neural_network import MLPRegressor
            # Create autoencoder-like architecture
            autoencoder = MLPRegressor(
                hidden_layer_sizes=(len(feature_columns)//2, len(feature_columns)//4, len(feature_columns)//2),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42
            )
            autoencoder.fit(X_scaled, X_scaled)  # Autoencoder: input = output
            self.baseline_models['autoencoder'] = autoencoder
        except Exception as e:
            print(f"Autoencoder training failed: {e}")
            self.baseline_models['autoencoder'] = None
        
        print(f"Trained {len(self.baseline_models)} baseline models")
    
    def evaluate_baseline_models(self, test_data, feature_columns):
        """
        Evaluate baseline models on test data.
        
        Args:
            test_data (pd.DataFrame): Test data (attack data)
            feature_columns (list): List of feature column names
            
        Returns:
            dict: Evaluation results for all models
        """
        print("Evaluating baseline models...")
        
        # Prepare test data
        X_test = test_data[feature_columns].values
        X_test_scaled = self.scaler.transform(X_test)
        true_labels = test_data['Label'].values if 'Label' in test_data.columns else np.zeros(len(test_data))
        
        results = {}
        
        # Evaluate each model
        for model_name, model in self.baseline_models.items():
            print(f"Evaluating {model_name}...")
            
            if model_name == 'isolation_forest':
                predictions = model.predict(X_test_scaled)
                # Convert to binary (1 for anomaly, 0 for normal)
                predictions = (predictions == -1).astype(int)
                
            elif model_name == 'one_class_svm':
                predictions = model.predict(X_test_scaled)
                # Convert to binary (1 for anomaly, 0 for normal)
                predictions = (predictions == -1).astype(int)
                
            elif model_name == 'dbscan':
                predictions = model.fit_predict(X_test_scaled)
                # Convert to binary (1 for anomaly, 0 for normal)
                predictions = (predictions == -1).astype(int)
                
            elif model_name == 'statistical':
                # Calculate Z-scores
                z_scores = np.abs((X_test_scaled - model['means']) / model['stds'])
                # Flag as anomaly if any feature exceeds threshold
                predictions = (np.max(z_scores, axis=1) > model['threshold']).astype(int)
                
            elif model_name == 'autoencoder':
                if model is not None:
                    # Reconstruct data using autoencoder
                    reconstructed = model.predict(X_test_scaled)
                    # Calculate reconstruction error
                    reconstruction_error = np.mean((X_test_scaled - reconstructed) ** 2, axis=1)
                    # Use 95th percentile as threshold
                    threshold = np.percentile(reconstruction_error, 95)
                    predictions = (reconstruction_error > threshold).astype(int)
                else:
                    predictions = np.zeros(len(X_test_scaled), dtype=int)
            
            # Calculate metrics
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            results[model_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': predictions.tolist()
            }
        
        # Save results
        results_path = os.path.join(self.output_dir, 'baseline_analysis', 'results', 'baseline_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create comparison plot
        self._create_baseline_comparison_plot(results, true_labels)
        
        return results
    
    def _create_baseline_comparison_plot(self, results, true_labels):
        """
        Create comparison plot for baseline models.
        """
        # Extract metrics
        model_names = list(results.keys())
        precisions = [results[name]['precision'] for name in model_names]
        recalls = [results[name]['recall'] for name in model_names]
        f1_scores = [results[name]['f1'] for name in model_names]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Precision
        axes[0].bar(model_names, precisions, color='skyblue', alpha=0.7)
        axes[0].set_title('Precision Comparison')
        axes[0].set_ylabel('Precision')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[1].bar(model_names, recalls, color='lightcoral', alpha=0.7)
        axes[1].set_title('Recall Comparison')
        axes[1].set_ylabel('Recall')
        axes[1].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[2].bar(model_names, f1_scores, color='lightgreen', alpha=0.7)
        axes[2].set_title('F1-Score Comparison')
        axes[2].set_ylabel('F1-Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'baseline_analysis', 'plots', 'baseline_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Baseline comparison plot saved to: {plot_path}")
    
    def compare_with_scm(self, test_data, feature_columns):
        """
        Compare baseline methods with SCM-based approach.
        
        Args:
            test_data (pd.DataFrame): Test data (attack data)
            feature_columns (list): List of feature column names
        """
        print("Comparing with SCM-based approach...")
        
        # Initialize SCM detector
        scm_detector = SWaTCausalDetector(self.data_dir, self.output_dir)
        
        # Load and preprocess data for SCM
        normal_data, _, _ = scm_detector.load_and_preprocess_data()
        
        # Train SCM models
        scm_detector.learn_causal_graph(normal_data, feature_columns)
        scm_detector.train_structural_models(normal_data, feature_columns)
        
        # Detect anomalies with SCM
        anomaly_scores, anomaly_flags, causal_paths = scm_detector.detect_anomalies(test_data, feature_columns)
        
        # Evaluate SCM performance
        scm_metrics = scm_detector.evaluate_performance(test_data, anomaly_flags, feature_columns)
        
        # Get baseline results
        baseline_results = self.results.get('baseline', {})
        
        # Create comparison
        comparison = {
            'scm': {
                'precision': scm_metrics['overall']['precision'],
                'recall': scm_metrics['overall']['recall'],
                'f1': scm_metrics['overall']['f1']
            },
            'baseline': baseline_results
        }
        
        # Print comparison
        print("\n" + "="*60)
        print("SCM vs BASELINE COMPARISON")
        print("="*60)
        
        print(f"{'Method':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        # SCM results
        print(f"{'SCM-based':<20} {scm_metrics['overall']['precision']:<10.3f} {scm_metrics['overall']['recall']:<10.3f} {scm_metrics['overall']['f1']:<10.3f}")
        
        # Baseline results
        for model_name, metrics in baseline_results.items():
            print(f"{model_name:<20} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f}")
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, 'baseline_analysis', 'results', 'scm_baseline_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"\nComparison results saved to: {comparison_path}")
        
        return comparison
    
    def run_full_analysis(self):
        """
        Run the complete baseline analysis pipeline.
        """
        print("Starting SWaT Baseline Analysis Pipeline")
        print("="*50)
        
        # Load and preprocess data
        normal_data, attack_data, feature_columns = self.load_and_preprocess_data()
        
        # Explore data
        self.explore_data(normal_data, attack_data, feature_columns)
        
        # Train baseline models
        self.train_baseline_models(normal_data, feature_columns)
        
        # Evaluate baseline models
        baseline_results = self.evaluate_baseline_models(attack_data, feature_columns)
        self.results['baseline'] = baseline_results
        
        # Compare with SCM
        comparison = self.compare_with_scm(attack_data, feature_columns)
        self.results['comparison'] = comparison
        
        # Print summary
        print("\n" + "="*50)
        print("BASELINE ANALYSIS SUMMARY")
        print("="*50)
        print(f"Features analyzed: {len(feature_columns)}")
        print(f"Normal samples: {len(normal_data)}")
        print(f"Attack samples: {len(attack_data)}")
        print(f"Baseline models trained: {len(self.baseline_models)}")
        print(f"Results saved to: {os.path.join(self.output_dir, 'baseline_analysis')}")
        
        return self.results


def main():
    """
    Main function to run the baseline analysis.
    """
    parser = argparse.ArgumentParser(description='SWaT Baseline Analysis')
    parser.add_argument('--data_dir', default='data', help='Directory containing SWaT data files')
    parser.add_argument('--output_dir', default='outputs', help='Directory to save outputs')
    parser.add_argument('--compare_scm', action='store_true', help='Compare with SCM-based approach')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SWaTBaselineAnalyzer(args.data_dir, args.output_dir)
    
    # Run analysis
    results = analyzer.run_full_analysis()
    
    return results


if __name__ == "__main__":
    results = main()
