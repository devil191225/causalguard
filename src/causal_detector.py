"""
SCM-based Anomaly Detector for SWaT Dataset
===========================================

This module implements a Structural Causal Model (SCM) based anomaly detector
for the SWaT industrial control dataset. It learns causal relationships between
process variables and uses them to detect anomalies.

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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import networkx as nx

# Try to import causal discovery libraries
try:
    from causalnex.structure import StructureModel
    from causalnex.structure.notears import from_pandas
    CAUSALNEX_AVAILABLE = True
except ImportError:
    CAUSALNEX_AVAILABLE = False

try:
    from dowhy import CausalModel
    from dowhy.causal_learners import PC, GES
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

# Fallback StructureModel implementation
class StructureModel:
    """Fallback StructureModel implementation."""
    def __init__(self):
        self.edges = []
        self.nodes = set()
    
    def add_edge(self, source, target):
        self.edges.append((source, target))
        self.nodes.add(source)
        self.nodes.add(target)
    
    def __len__(self):
        return len(self.edges)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm
import joblib
import json
from datetime import datetime

# Import true causal discovery
try:
    from true_causal_discovery import TrueCausalDiscovery
    TRUE_CAUSAL_AVAILABLE = True
except ImportError:
    TRUE_CAUSAL_AVAILABLE = False
    print(" Warning: True causal discovery not available, using fallback")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SWaTCausalDetector:
    """
    SCM-based anomaly detector for SWaT dataset.
    """
    
    def __init__(self, data_dir='data', output_dir='outputs', adaptive_learning=True):
        """
        Initialize the detector with adaptive learning capabilities.
        
        Args:
            data_dir (str): Directory containing SWaT data files
            output_dir (str): Directory to save outputs
            adaptive_learning (bool): Enable adaptive learning from previous runs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.adaptive_learning = adaptive_learning
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.causal_graph = None
        self.models = {}
        self.feature_names = None
        self.anomaly_thresholds = {}
        self.performance_history = []
        self.learning_rate = 0.1  # How much to adapt from new data
        # Phase 2: retain scaled normal training matrix for verification / intervals
        self._normal_training_scaled = None
        self._training_feature_columns = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'adaptive'), exist_ok=True)
        
        # Load previous model state if adaptive learning is enabled
        if self.adaptive_learning:
            self._load_previous_state()
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess SWaT dataset.
        
        Returns:
            tuple: (normal_data, attack_data, feature_names)
        """
        print("Loading SWaT dataset...")
        
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
        
        # Get feature names (exclude timestamp, label, and any attack-related columns)
        # CRITICAL FIX: Exclude ALL columns containing 'attack' to prevent label leakage
        feature_columns = [col for col in normal_data.columns 
                          if col not in ['Timestamp', 'Normal/Attack', 'Label', 'time'] and 
                          'attack' not in col.lower()]  # More aggressive filtering
        self.feature_names = feature_columns
        
        excluded_cols = [col for col in normal_data.columns if col not in feature_columns and col.startswith(('P', 'attack'))]
        if excluded_cols:
            print(f"WARNING: EXCLUDED FROM CAUSAL DISCOVERY: {excluded_cols}")
        
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

        # Convert timestamp if present (before numeric fill — median() is numeric-only)
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        # Handle missing values (numeric columns only)
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        # Handle label column
        if 'Normal/Attack' in df.columns:
            df['Label'] = (df['Normal/Attack'] == 'Attack').astype(int)
        elif 'Label' not in df.columns and is_attack:
            # If no label column in attack data, assume all are attacks
            df['Label'] = 1
        elif 'Label' not in df.columns:
            df['Label'] = 0
        
        return df
    
    def learn_causal_graph(self, normal_data, feature_columns):
        """
        Learn causal graph from normal data using NOTEARS algorithm.
        
        Args:
            normal_data (pd.DataFrame): Normal operation data
            feature_columns (list): List of feature column names
        """
        print("Learning causal graph from normal data...")
        
        # Prepare data for causal discovery
        X = normal_data[feature_columns].values
        
        # Use first week of data for graph learning (or first 1000 samples if less)
        n_samples = min(1000, len(X))
        X_sample = X[:n_samples]
        
        # Try TRUE CAUSAL DISCOVERY first (PC/GES algorithms)
        if TRUE_CAUSAL_AVAILABLE:
            try:
                print(" Using TRUE CAUSAL DISCOVERY (PC Algorithm)...")
                # TUNED FOR SPARSITY: alpha=0.01 (stricter), max_cond_set_size=2 (simpler)
                # Target: <0.5 edges per node (e.g., <10 edges for 20 features)
                discoverer = TrueCausalDiscovery(method='pc', alpha=0.01, max_cond_set_size=2)
                result = discoverer.discover(
                    pd.DataFrame(X_sample, columns=feature_columns),
                    feature_names=feature_columns
                )
                
                # Convert to our graph format
                self.causal_graph = StructureModel()
                for source, target in result['edges']:
                    self.causal_graph.add_edge(source, target)
                
                edge_count = len(result['edges'])
                sparsity_ratio = edge_count / len(feature_columns) if len(feature_columns) > 0 else 0
                print(f" TRUE CAUSAL DISCOVERY: Learned {edge_count} causal edges")
                print(f" Sparsity ratio: {sparsity_ratio:.2f} edges/feature (target: <0.5)")
                
                if sparsity_ratio > 1.0:
                    print(f"WARNING: Graph may be too dense ({edge_count} edges for {len(feature_columns)} features)")
            except Exception as e:
                print(f" PC Algorithm failed: {e}, trying GES...")
                try:
                    discoverer = TrueCausalDiscovery(method='ges', score_type='bic')
                    result = discoverer.discover(
                        pd.DataFrame(X_sample, columns=feature_columns),
                        feature_names=feature_columns
                    )
                    
                    self.causal_graph = StructureModel()
                    for source, target in result['edges']:
                        self.causal_graph.add_edge(source, target)
                    
                    print(f" GES ALGORITHM: Learned {len(result['edges'])} causal edges")
                except Exception as e2:
                    print(f" GES also failed: {e2}")
                    self._try_alternative_causal_discovery(normal_data, feature_columns)
        # Try NOTEARS if available
        elif CAUSALNEX_AVAILABLE:
            try:
                print("Attempting NOTEARS causal discovery...")
                sm = from_pandas(
                    pd.DataFrame(X_sample, columns=feature_columns),
                    max_iter=100,
                    h_tol=1e-8,
                    w_threshold=0.1
                )
                self.causal_graph = sm
                print(f" NOTEARS learned causal graph with {len(sm.edges)} edges")
            except Exception as e:
                print(f" NOTEARS failed: {e}")
                self._try_alternative_causal_discovery(normal_data, feature_columns)
        elif DOWHY_AVAILABLE:
            print("Attempting DoWhy PC algorithm...")
            self._try_alternative_causal_discovery(normal_data, feature_columns)
        else:
            print(" FALLBACK: Using correlation-based graph (NOT true causal discovery)...")
            self._create_correlation_based_graph(normal_data, feature_columns)
        
        # Visualize the graph
        self._visualize_causal_graph(feature_columns)
    
    def _try_alternative_causal_discovery(self, normal_data, feature_columns):
        """
        Try alternative causal discovery methods when NOTEARS fails.
        """
        if DOWHY_AVAILABLE:
            try:
                print("Attempting DoWhy PC algorithm...")
                self._dowhy_pc_discovery(normal_data, feature_columns)
                return
            except Exception as e:
                print(f" DoWhy PC failed: {e}")
        
        # Fallback to correlation-based method
        print("Falling back to correlation-based causal graph...")
        self._create_correlation_based_graph(normal_data, feature_columns)
    
    def _dowhy_pc_discovery(self, normal_data, feature_columns):
        """
        Use DoWhy's PC algorithm for causal discovery.
        """
        from dowhy.causal_learners import PC
        
        # Prepare data
        df = normal_data[feature_columns].copy()
        
        # Run PC algorithm
        pc_learner = PC()
        pc_learner.learn(df)
        
        # Convert to our structure model
        self.causal_graph = StructureModel()
        
        # Get edges from PC result
        if hasattr(pc_learner, 'graph') and pc_learner.graph is not None:
            edges = pc_learner.graph.edges()
            for edge in edges:
                self.causal_graph.add_edge(edge[0], edge[1])
        
        print(f" DoWhy PC learned causal graph with {len(self.causal_graph.edges)} edges")
    
    def _create_correlation_based_graph(self, normal_data, feature_columns):
        """
        Create a simple causal graph based on correlation when NOTEARS fails.
        """
        print("Creating correlation-based causal graph...")
        
        # Calculate correlation matrix
        corr_matrix = normal_data[feature_columns].corr().abs()
        
        # Create graph with edges for correlations > 0.3
        self.causal_graph = StructureModel()
        
        for i, feature1 in enumerate(feature_columns):
            for j, feature2 in enumerate(feature_columns):
                if i != j and corr_matrix.loc[feature1, feature2] > 0.3:
                    self.causal_graph.add_edge(feature1, feature2)
        
        print(f"Created correlation-based graph with {len(self.causal_graph.edges)} edges")
    
    def _visualize_causal_graph(self, feature_columns):
        """
        Visualize the learned causal graph.
        """
        if self.causal_graph is None:
            return
        
        print("Visualizing causal graph...")
        
        # Create NetworkX graph for visualization
        G = nx.DiGraph()
        G.add_edges_from(self.causal_graph.edges)
        
        # Plot using matplotlib
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title('Learned Causal Graph', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'plots', 'causal_graph.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Causal graph saved to: {plot_path}")
    
    def train_structural_models(self, normal_data, feature_columns):
        """
        Train regression/classification models for each variable using its parents.
        Uses improved thresholding and residual analysis for better performance.
        
        Args:
            normal_data (pd.DataFrame): Normal operation data
            feature_columns (list): List of feature column names
        """
        print("Training structural models with improved thresholding...")
        
        if self.causal_graph is None:
            print("No causal graph available. Training independent models.")
            self._train_independent_models(normal_data, feature_columns)
            self._cache_training_matrix(normal_data, feature_columns)
            return
        
        # Prepare data
        X = normal_data[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model for each variable
        for i, target_var in enumerate(tqdm(feature_columns, desc="Training models")):
            # Get parents of this variable
            parents = [edge[0] for edge in self.causal_graph.edges if edge[1] == target_var]
            
            if not parents:
                # No parents - use statistical model with better thresholding
                y_target = X_scaled[:, i]
                mean_val = np.mean(y_target)
                std_val = np.std(y_target)
                
                # Use 3-sigma rule for better precision-recall balance
                threshold = 3 * std_val
                
                self.models[target_var] = {
                    'type': 'statistical',
                    'mean': mean_val,
                    'std': std_val,
                    'threshold': threshold,
                    'parents': []
                }
                continue
            
            # Get parent indices
            parent_indices = [feature_columns.index(p) for p in parents if p in feature_columns]
            
            if not parent_indices:
                # Parents not in feature list - use statistical model
                y_target = X_scaled[:, i]
                mean_val = np.mean(y_target)
                std_val = np.std(y_target)
                threshold = 3 * std_val
                
                self.models[target_var] = {
                    'type': 'statistical',
                    'mean': mean_val,
                    'std': std_val,
                    'threshold': threshold,
                    'parents': []
                }
                continue
            
            # Prepare training data
            X_parents = X_scaled[:, parent_indices]
            y_target = X_scaled[:, i]
            
            # Train model with ensemble approach
            try:
                # Use ensemble of models for better prediction
                models_ensemble = []
                
                # Random Forest
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                rf_model.fit(X_parents, y_target)
                models_ensemble.append(rf_model)
                
                # Linear Regression for linear relationships
                lr_model = LinearRegression()
                lr_model.fit(X_parents, y_target)
                models_ensemble.append(lr_model)
                
                # Calculate ensemble predictions
                y_pred_rf = rf_model.predict(X_parents)
                y_pred_lr = lr_model.predict(X_parents)
                y_pred = 0.7 * y_pred_rf + 0.3 * y_pred_lr  # Weighted ensemble
                
                # Calculate residuals and improved thresholding
                residuals = np.abs(y_target - y_pred)
                
                # Use precision-recall optimized thresholding
                threshold_95 = np.percentile(residuals, 95)
                threshold_99 = np.percentile(residuals, 99)
                
                # Find optimal threshold using precision-recall curve
                try:
                    from sklearn.metrics import precision_recall_curve
                    # Create synthetic labels for threshold optimization
                    synthetic_labels = np.zeros(len(residuals))
                    # Mark top 5% as "anomalies" for threshold optimization
                    top_5_percent = int(0.05 * len(residuals))
                    anomaly_indices = np.argsort(residuals)[-top_5_percent:]
                    synthetic_labels[anomaly_indices] = 1
                    
                    # Find threshold that gives good precision-recall balance
                    precisions, recalls, thresholds = precision_recall_curve(synthetic_labels, residuals)
                    
                    # Find threshold where precision >= 0.6 and recall >= 0.8
                    valid_indices = (precisions >= 0.6) & (recalls >= 0.8)
                    if np.any(valid_indices):
                        optimal_idx = np.argmax(recalls[valid_indices])
                        threshold = thresholds[valid_indices][optimal_idx]
                    else:
                        # Fallback to 99th percentile if no good threshold found
                        threshold = threshold_99
                except:
                    # Fallback to 99th percentile
                    threshold = threshold_99
                
                # Calculate additional statistics for better anomaly detection
                residual_std = np.std(residuals)
                residual_mean = np.mean(residuals)
                
                self.models[target_var] = {
                    'type': 'ensemble',
                    'models': models_ensemble,
                    'parents': parents,
                    'parent_indices': parent_indices,
                    'threshold': threshold,
                    'threshold_95': threshold_95,
                    'threshold_99': threshold_99,
                    'residual_mean': residual_mean,
                    'residual_std': residual_std
                }
                
            except Exception as e:
                print(f"Error training model for {target_var}: {e}")
                # Fallback to statistical model
                y_target = X_scaled[:, i]
                mean_val = np.mean(y_target)
                std_val = np.std(y_target)
                threshold = 3 * std_val
                
                self.models[target_var] = {
                    'type': 'statistical',
                    'mean': mean_val,
                    'std': std_val,
                    'threshold': threshold,
                    'parents': []
                }
        
        self._cache_training_matrix(normal_data, feature_columns)
        # Save models
        self._save_models()
        print(f"Trained {len(self.models)} structural models with improved thresholding")
    
    def _cache_training_matrix(self, normal_data, feature_columns):
        """Store scaled normal-operation data for formal verification / interval analysis."""
        X = normal_data[feature_columns].values
        self._normal_training_scaled = np.asarray(
            self.scaler.transform(X), dtype=np.float64
        ).copy()
        self._training_feature_columns = list(feature_columns)
    
    def compute_residuals_matrix(self, X_scaled, feature_columns):
        """
        Forward pass: per-sample, per-feature absolute residual (same as detect_anomalies scores).
        
        Args:
            X_scaled (np.ndarray): Scaled feature matrix (n_samples, n_features)
            feature_columns (list): Ordered feature names
            
        Returns:
            np.ndarray: Shape (n_samples, n_features) residuals
        """
        n_samples, _ = X_scaled.shape
        n_features = len(feature_columns)
        anomaly_scores = np.zeros((n_samples, n_features))
        for i, target_var in enumerate(feature_columns):
            model_info = self.models[target_var]
            if model_info['type'] == 'statistical':
                y_pred = np.full(n_samples, model_info['mean'])
                y_true = X_scaled[:, i]
                anomaly_scores[:, i] = np.abs(y_true - y_pred)
            elif model_info['type'] == 'ensemble':
                parent_indices = model_info['parent_indices']
                X_parents = X_scaled[:, parent_indices]
                y_pred_rf = model_info['models'][0].predict(X_parents)
                y_pred_lr = model_info['models'][1].predict(X_parents)
                y_pred = 0.7 * y_pred_rf + 0.3 * y_pred_lr
                y_true = X_scaled[:, i]
                anomaly_scores[:, i] = np.abs(y_true - y_pred)
            elif model_info['type'] == 'regressor':
                parent_indices = model_info['parent_indices']
                X_parents = X_scaled[:, parent_indices]
                y_pred = model_info['model'].predict(X_parents)
                y_true = X_scaled[:, i]
                anomaly_scores[:, i] = np.abs(y_true - y_pred)
            else:
                y_pred = np.full(n_samples, model_info.get('value', 0))
                y_true = X_scaled[:, i]
                anomaly_scores[:, i] = np.abs(y_true - y_pred)
        return anomaly_scores
    
    def _train_independent_models(self, normal_data, feature_columns):
        """
        Train independent models when no causal graph is available.
        """
        X = normal_data[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        for i, target_var in enumerate(feature_columns):
            # Use all other features as predictors
            other_indices = [j for j in range(len(feature_columns)) if j != i]
            X_other = X_scaled[:, other_indices]
            y_target = X_scaled[:, i]
            
            try:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_other, y_target)
                
                y_pred = model.predict(X_other)
                errors = np.abs(y_target - y_pred)
                threshold = np.mean(errors) + 3 * np.std(errors)
                
                self.models[target_var] = {
                    'type': 'regressor',
                    'model': model,
                    'parents': [feature_columns[j] for j in other_indices],
                    'parent_indices': other_indices,
                    'threshold': threshold
                }
            except Exception as e:
                print(f"Error training model for {target_var}: {e}")
                self.models[target_var] = {
                    'type': 'constant',
                    'value': np.mean(X_scaled[:, i]),
                    'parents': []
                }
    
    def _save_models(self):
        """
        Save trained models to disk.
        """
        models_path = os.path.join(self.output_dir, 'models', 'structural_models.joblib')
        
        # Prepare models for saving (remove non-serializable objects)
        save_models = {}
        for var, model_info in self.models.items():
            save_models[var] = {
                'type': model_info['type'],
                'parents': model_info['parents'],
                'threshold': model_info.get('threshold', 0),
                'value': model_info.get('value', 0)
            }
            
            if model_info['type'] == 'regressor':
                save_models[var]['model'] = model_info['model']
                save_models[var]['parent_indices'] = model_info['parent_indices']
        
        joblib.dump(save_models, models_path)
        print(f"Models saved to: {models_path}")
    
    def detect_anomalies(self, test_data, feature_columns):
        """
        Detect anomalies in test data using trained models with improved scoring.
        
        Args:
            test_data (pd.DataFrame): Test data (attack data)
            feature_columns (list): List of feature column names
            
        Returns:
            tuple: (anomaly_scores, anomaly_flags, causal_paths)
        """
        print("Detecting anomalies with improved scoring...")
        
        # Prepare test data
        X_test = test_data[feature_columns].values
        X_test_scaled = self.scaler.transform(X_test)
        
        n_samples, n_features = X_test_scaled.shape
        anomaly_scores = np.zeros((n_samples, n_features))
        anomaly_flags = np.zeros((n_samples, n_features), dtype=bool)
        causal_paths = []
        
        # Predict each variable and calculate anomaly scores
        for i, target_var in enumerate(tqdm(feature_columns, desc="Detecting anomalies")):
            model_info = self.models[target_var]
            
            if model_info['type'] == 'statistical':
                # Statistical model - use mean and std
                y_pred = np.full(n_samples, model_info['mean'])
                y_true = X_test_scaled[:, i]
                errors = np.abs(y_true - y_pred)
                anomaly_scores[:, i] = errors
                
                # Use multiple thresholds for better recall
                threshold = model_info.get('threshold', 0)
                anomaly_flags[:, i] = errors > threshold
                
            elif model_info['type'] == 'ensemble':
                # Ensemble model
                parent_indices = model_info['parent_indices']
                X_parents = X_test_scaled[:, parent_indices]
                
                # Get ensemble prediction
                y_pred_rf = model_info['models'][0].predict(X_parents)
                y_pred_lr = model_info['models'][1].predict(X_parents)
                y_pred = 0.7 * y_pred_rf + 0.3 * y_pred_lr
                
                # Calculate residuals
                y_true = X_test_scaled[:, i]
                residuals = np.abs(y_true - y_pred)
                anomaly_scores[:, i] = residuals
                
                # Use precision-recall optimized threshold
                threshold = model_info.get('threshold', 0)
                
                # Use optimized threshold for better precision-recall balance
                anomaly_flags[:, i] = residuals > threshold
                
            elif model_info['type'] == 'regressor':
                parent_indices = model_info['parent_indices']
                X_parents = X_test_scaled[:, parent_indices]
                y_pred = model_info['model'].predict(X_parents)
                y_true = X_test_scaled[:, i]
                residuals = np.abs(y_true - y_pred)
                anomaly_scores[:, i] = residuals
                threshold = model_info.get('threshold', 0)
                anomaly_flags[:, i] = residuals > threshold
                
            else:
                # Fallback to constant model
                y_pred = np.full(n_samples, model_info.get('value', 0))
                y_true = X_test_scaled[:, i]
                errors = np.abs(y_true - y_pred)
                anomaly_scores[:, i] = errors
                
                threshold = model_info.get('threshold', 0)
                anomaly_flags[:, i] = errors > threshold
        
        # Find causal paths for detected anomalies
        causal_paths = self._find_causal_paths(anomaly_flags, feature_columns)
        
        return anomaly_scores, anomaly_flags, causal_paths
    
    def detect_anomalies_simple(self, test_data, feature_columns):
        """
        Simplified anomaly detection that returns 1D prediction array.
        
        Args:
            test_data (pd.DataFrame): Test data
            feature_columns (list): Feature column names
            
        Returns:
            np.ndarray: 1D array of predictions (0=normal, 1=anomaly)
        """
        anomaly_scores, anomaly_flags, _ = self.detect_anomalies(test_data, feature_columns)
        
        # Aggregate anomaly flags across all features
        # Sample is anomalous if ANY feature is flagged
        predictions = np.any(anomaly_flags, axis=1).astype(int)
        
        return predictions
    
    def _find_causal_paths(self, anomaly_flags, feature_columns):
        """
        Find causal paths for detected anomalies.
        
        Args:
            anomaly_flags (np.ndarray): Boolean array of anomaly flags
            feature_columns (list): List of feature column names
            
        Returns:
            list: List of causal paths for each sample
        """
        causal_paths = []
        
        for sample_idx in range(anomaly_flags.shape[0]):
            sample_anomalies = anomaly_flags[sample_idx]
            anomalous_vars = [feature_columns[i] for i in range(len(feature_columns)) 
                            if sample_anomalies[i]]
            
            if not anomalous_vars:
                causal_paths.append([])
                continue
            
            # Find causal relationships between anomalous variables
            paths = []
            for var in anomalous_vars:
                if self.causal_graph and var in self.causal_graph.nodes:
                    # Find parents of this variable that are also anomalous
                    parents = [edge[0] for edge in self.causal_graph.edges if edge[1] == var]
                    anomalous_parents = [p for p in parents if p in anomalous_vars]
                    
                    if anomalous_parents:
                        paths.append(f"{var} <- {', '.join(anomalous_parents)}")
                    else:
                        paths.append(f"{var} (root cause)")
                else:
                    paths.append(f"{var} (independent)")
            
            causal_paths.append(paths)
        
        return causal_paths
    
    def evaluate_performance(self, test_data, anomaly_flags, feature_columns):
        """
        Evaluate anomaly detection performance.
        
        Args:
            test_data (pd.DataFrame): Test data with true labels
            anomaly_flags (np.ndarray): Predicted anomaly flags
            feature_columns (list): List of feature column names
            
        Returns:
            dict: Performance metrics
        """
        print("Evaluating performance...")
        
        # Get true labels
        true_labels = test_data['Label'].values if 'Label' in test_data.columns else np.zeros(len(test_data))
        
        # Create overall anomaly score (max across all features)
        overall_anomaly_scores = np.max(anomaly_flags, axis=1).astype(int)
        
        # Calculate metrics
        precision = precision_score(true_labels, overall_anomaly_scores, zero_division=0)
        recall = recall_score(true_labels, overall_anomaly_scores, zero_division=0)
        f1 = f1_score(true_labels, overall_anomaly_scores, zero_division=0)
        
        # Calculate per-feature metrics
        feature_metrics = {}
        for i, feature in enumerate(feature_columns):
            feature_flags = anomaly_flags[:, i]
            if len(np.unique(feature_flags)) > 1:  # Only if there are both 0s and 1s
                feature_precision = precision_score(true_labels, feature_flags, zero_division=0)
                feature_recall = recall_score(true_labels, feature_flags, zero_division=0)
                feature_f1 = f1_score(true_labels, feature_flags, zero_division=0)
                feature_metrics[feature] = {
                    'precision': feature_precision,
                    'recall': feature_recall,
                    'f1': feature_f1
                }
        
        metrics = {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'features': feature_metrics
        }
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, 'results', 'performance_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        self._create_evaluation_plots(true_labels, overall_anomaly_scores, feature_columns)
        
        return metrics
    
    def _create_evaluation_plots(self, true_labels, predicted_labels, feature_columns):
        """
        Create evaluation plots (confusion matrix, ROC curve).
        """
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        plt.figure(figsize=(12, 5))
        
        # Confusion Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC Curve (if we have probability scores)
        plt.subplot(1, 2, 2)
        if len(np.unique(predicted_labels)) > 1:
            fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
            auc = roc_auc_score(true_labels, predicted_labels)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No ROC curve available\n(only one class predicted)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('ROC Curve')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'plots', 'evaluation_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Evaluation plots saved to: {plot_path}")
    
    def generate_explanations(self, test_data, anomaly_flags, causal_paths, feature_columns, n_examples=3):
        """
        Generate counterfactual explanations for detected anomalies.
        
        Args:
            test_data (pd.DataFrame): Test data
            anomaly_flags (np.ndarray): Predicted anomaly flags
            causal_paths (list): Causal paths for each sample
            feature_columns (list): List of feature column names
            n_examples (int): Number of examples to explain
        """
        print(f"Generating explanations for {n_examples} anomalies...")
        
        # Find samples with anomalies
        overall_anomalies = np.max(anomaly_flags, axis=1)
        anomaly_indices = np.where(overall_anomalies)[0]
        
        if len(anomaly_indices) == 0:
            print("No anomalies detected to explain.")
            return
        
        # Select random examples
        selected_indices = np.random.choice(anomaly_indices, 
                                          size=min(n_examples, len(anomaly_indices)), 
                                          replace=False)
        
        explanations = []
        
        for idx in selected_indices:
            sample_anomalies = anomaly_flags[idx]
            anomalous_vars = [feature_columns[i] for i in range(len(feature_columns)) 
                            if sample_anomalies[i]]
            
            explanation = {
                'sample_index': idx,
                'anomalous_variables': anomalous_vars,
                'causal_paths': causal_paths[idx],
                'explanation': self._generate_counterfactual_explanation(
                    anomalous_vars, causal_paths[idx], idx, test_data, feature_columns
                )
            }
            explanations.append(explanation)
        
        # Save explanations
        explanations_path = os.path.join(self.output_dir, 'results', 'explanations.json')
        with open(explanations_path, 'w') as f:
            json.dump(explanations, f, indent=2, default=str)
        
        # Save detailed counterfactual examples to text file
        self._save_counterfactual_examples(explanations, test_data, feature_columns)
        
        # Print explanations
        print("\n" + "="*50)
        print("COUNTERFACTUAL EXPLANATIONS")
        print("="*50)
        
        for i, exp in enumerate(explanations, 1):
            print(f"\nExample {i} (Sample {exp['sample_index']}):")
            print(f"Anomalous variables: {', '.join(exp['anomalous_variables'])}")
            print(f"Causal paths: {exp['causal_paths']}")
            print(f"Explanation: {exp['explanation']}")
        
        print(f"\nExplanations saved to: {explanations_path}")
    
    def _generate_counterfactual_explanation(self, anomalous_vars, causal_paths, sample_idx=None, test_data=None, feature_columns=None):
        """
        Generate a detailed counterfactual explanation for anomalous variables.
        
        Args:
            anomalous_vars (list): List of anomalous variable names
            causal_paths (list): Causal paths for the sample
            sample_idx (int): Sample index for detailed analysis
            test_data (pd.DataFrame): Test data for actual values
            feature_columns (list): Feature column names
            
        Returns:
            str: Detailed counterfactual explanation
        """
        if not anomalous_vars:
            return "No anomalies detected."
        
        explanation_parts = []
        
        # Analyze each anomalous variable
        for var in anomalous_vars:
            var_idx = feature_columns.index(var) if feature_columns else None
            
            # Get actual and expected values if available
            if sample_idx is not None and test_data is not None and var_idx is not None:
                try:
                    actual_value = float(test_data.iloc[sample_idx, var_idx])
                except (ValueError, TypeError):
                    # Skip non-numeric columns
                    continue
                
                # Get expected value from model
                model_info = self.models.get(var, {})
                if model_info.get('type') == 'statistical':
                    expected_value = model_info.get('mean', 0)
                    deviation = abs(actual_value - expected_value)
                    explanation_parts.append(
                        f" {var}: actual={actual_value:.3f}, expected={expected_value:.3f}, deviation={deviation:.3f}"
                    )
                elif model_info.get('type') == 'ensemble':
                    # Calculate expected value from ensemble
                    parent_indices = model_info.get('parent_indices', [])
                    if parent_indices and sample_idx is not None:
                        X_test_scaled = self.scaler.transform(test_data[feature_columns].values)
                        X_parents = X_test_scaled[sample_idx:sample_idx+1, parent_indices]
                        y_pred_rf = model_info['models'][0].predict(X_parents)[0]
                        y_pred_lr = model_info['models'][1].predict(X_parents)[0]
                        expected_value = 0.7 * y_pred_rf + 0.3 * y_pred_lr
                        deviation = abs(actual_value - expected_value)
                        explanation_parts.append(
                            f" {var}: actual={actual_value:.3f}, expected={expected_value:.3f}, deviation={deviation:.3f}"
                        )
                    else:
                        explanation_parts.append(f" {var}: deviated from causal prediction")
                else:
                    explanation_parts.append(f" {var}: deviated from expected value")
            else:
                explanation_parts.append(f" {var}: deviated from expected value")
        
        # Analyze causal relationships
        root_causes = [path for path in causal_paths if "(root cause)" in path]
        dependent_vars = [path for path in causal_paths if "(root cause)" not in path and "independent" not in path]
        
        if root_causes:
            root_vars = [path.split()[0] for path in root_causes]
            explanation_parts.append(f"\nRoot causes identified: {', '.join(root_vars)}")
            
            if dependent_vars:
                dep_vars = [path.split()[0] for path in dependent_vars]
                explanation_parts.append(f"Cascade effects: {', '.join(dep_vars)}")
            
            explanation_parts.append(f"\nCounterfactual: If {', '.join(root_vars)} had maintained normal values, the cascade would not have occurred.")
        else:
            explanation_parts.append(f"\nCounterfactual: If {', '.join(anomalous_vars)} had maintained normal values, the system would have remained stable.")
        
        return "\n".join(explanation_parts)
    
    def _save_counterfactual_examples(self, explanations, test_data, feature_columns):
        """
        Save detailed counterfactual examples to a text file.
        """
        examples_file = os.path.join(self.output_dir, 'results', 'counterfactual_examples.txt')
        
        with open(examples_file, 'w') as f:
            f.write("DETAILED COUNTERFACTUAL EXPLANATIONS\n")
            f.write("="*60 + "\n\n")
            f.write("This file contains detailed counterfactual explanations for detected anomalies.\n")
            f.write("Each explanation shows what would have happened if variables had maintained normal values.\n\n")
            
            for i, explanation in enumerate(explanations[:10]):  # Save first 10 examples
                f.write(f"EXAMPLE {i+1}:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Sample Index: {explanation['sample_index']}\n")
                f.write(f"Anomalous Variables: {', '.join(explanation['anomalous_variables'])}\n")
                f.write(f"Causal Paths: {explanation['causal_paths']}\n")
                f.write(f"Explanation:\n{explanation['explanation']}\n\n")
                
                # Add detailed analysis
                sample_idx = explanation['sample_index']
                anomalous_vars = explanation['anomalous_variables']
                
                f.write("DETAILED ANALYSIS:\n")
                for var in anomalous_vars:
                    if var in test_data.columns:
                        try:
                            actual_value = float(test_data.iloc[sample_idx][var])
                        except (ValueError, TypeError):
                            continue
                        var_idx = feature_columns.index(var) if var in feature_columns else -1
                        
                        if var_idx >= 0:
                            model_info = self.models.get(var, {})
                            if model_info.get('type') == 'statistical':
                                expected_value = model_info.get('mean', 0)
                                deviation = abs(actual_value - expected_value)
                                f.write(f" {var}: actual={actual_value:.3f}, expected={expected_value:.3f}, deviation={deviation:.3f}\n")
                            elif model_info.get('type') == 'ensemble':
                                # Calculate expected value from ensemble
                                parent_indices = model_info.get('parent_indices', [])
                                if parent_indices:
                                    X_parents = test_data[feature_columns].iloc[sample_idx:sample_idx+1, parent_indices].values
                                    rf_pred = model_info['models'][0].predict(X_parents)[0]
                                    lr_pred = model_info['models'][1].predict(X_parents)[0]
                                    expected_value = 0.7 * rf_pred + 0.3 * lr_pred
                                    deviation = abs(actual_value - expected_value)
                                    f.write(f" {var}: actual={actual_value:.3f}, expected={expected_value:.3f}, deviation={deviation:.3f}\n")
                
                f.write("\n" + "="*60 + "\n\n")
        
        print(f" Detailed counterfactual examples saved to: {examples_file}")
    
    def _load_previous_state(self):
        """
        Load previous model state for adaptive learning.
        """
        try:
            # Load previous models
            models_path = os.path.join(self.output_dir, 'adaptive', 'previous_models.joblib')
            if os.path.exists(models_path):
                previous_models = joblib.load(models_path)
                print(f" Loaded {len(previous_models)} previous models for adaptive learning")
                self.models = previous_models
            
            # Load performance history
            history_path = os.path.join(self.output_dir, 'adaptive', 'performance_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.performance_history = json.load(f)
                print(f" Loaded performance history: {len(self.performance_history)} previous runs")
            
            # Load causal graph
            graph_path = os.path.join(self.output_dir, 'adaptive', 'causal_graph.joblib')
            if os.path.exists(graph_path):
                self.causal_graph = joblib.load(graph_path)
                print(f" Loaded previous causal graph with {len(self.causal_graph.edges)} edges")
            
            # Load scaler
            scaler_path = os.path.join(self.output_dir, 'adaptive', 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(" Loaded previous scaler")
                
        except Exception as e:
            print(f"Warning: Could not load previous state: {e}")
            print("Starting with fresh models...")
    
    def _save_adaptive_state(self):
        """
        Save current model state for future adaptive learning.
        """
        try:
            # Save models
            models_path = os.path.join(self.output_dir, 'adaptive', 'previous_models.joblib')
            joblib.dump(self.models, models_path)
            
            # Save performance history
            history_path = os.path.join(self.output_dir, 'adaptive', 'performance_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
            
            # Save causal graph
            if self.causal_graph:
                graph_path = os.path.join(self.output_dir, 'adaptive', 'causal_graph.joblib')
                joblib.dump(self.causal_graph, graph_path)
            
            # Save scaler
            scaler_path = os.path.join(self.output_dir, 'adaptive', 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
            print(" Saved adaptive learning state")
            
        except Exception as e:
            print(f"Warning: Could not save adaptive state: {e}")
    
    def _adapt_models_with_new_data(self, normal_data, feature_columns):
        """
        Adapt existing models with new data using online learning.
        """
        if not self.models or not self.adaptive_learning:
            return
        
        print("Adapting models with new data...")
        
        # Prepare new data
        X_new = normal_data[feature_columns].values
        
        # Check if scaler is fitted, if not fit it first
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            X_new_scaled = self.scaler.fit_transform(X_new)
        else:
            X_new_scaled = self.scaler.transform(X_new)
        
        adapted_count = 0
        
        for var, model_info in self.models.items():
            if model_info.get('type') == 'ensemble' and 'models' in model_info:
                try:
                    # Get parent indices
                    parent_indices = model_info.get('parent_indices', [])
                    if not parent_indices:
                        continue
                    
                    X_parents = X_new_scaled[:, parent_indices]
                    y_target = X_new_scaled[:, feature_columns.index(var)]
                    
                    # Adapt Random Forest model (partial fit not available, so we'll use weighted learning)
                    rf_model = model_info['models'][0]
                    lr_model = model_info['models'][1]
                    
                    # For ensemble models, we'll update thresholds based on new data
                    y_pred_rf = rf_model.predict(X_parents)
                    y_pred_lr = lr_model.predict(X_parents)
                    y_pred = 0.7 * y_pred_rf + 0.3 * y_pred_lr
                    
                    # Calculate new residuals
                    residuals = np.abs(y_target - y_pred)
                    
                    # Adapt threshold using exponential moving average
                    old_threshold = model_info.get('threshold', 0)
                    new_threshold = np.percentile(residuals, 95)
                    adapted_threshold = (1 - self.learning_rate) * old_threshold + self.learning_rate * new_threshold
                    
                    # Update model info
                    model_info['threshold'] = adapted_threshold
                    model_info['threshold_95'] = np.percentile(residuals, 95)
                    model_info['threshold_99'] = np.percentile(residuals, 99)
                    
                    adapted_count += 1
                    
                except Exception as e:
                    print(f"Warning: Could not adapt model for {var}: {e}")
            
            elif model_info.get('type') == 'statistical':
                try:
                    # Adapt statistical model
                    y_target = X_new_scaled[:, feature_columns.index(var)]
                    
                    # Update mean and std using exponential moving average
                    old_mean = model_info.get('mean', 0)
                    old_std = model_info.get('std', 1)
                    
                    new_mean = np.mean(y_target)
                    new_std = np.std(y_target)
                    
                    adapted_mean = (1 - self.learning_rate) * old_mean + self.learning_rate * new_mean
                    adapted_std = (1 - self.learning_rate) * old_std + self.learning_rate * new_std
                    
                    model_info['mean'] = adapted_mean
                    model_info['std'] = adapted_std
                    model_info['threshold'] = 4 * adapted_std
                    
                    adapted_count += 1
                    
                except Exception as e:
                    print(f"Warning: Could not adapt statistical model for {var}: {e}")
        
        print(f" Adapted {adapted_count} models with new data")
    
    def _update_performance_history(self, metrics):
        """
        Update performance history for adaptive learning.
        """
        timestamp = datetime.now().isoformat()
        performance_entry = {
            'timestamp': timestamp,
            'metrics': metrics,
            'model_count': len(self.models),
            'graph_edges': len(self.causal_graph.edges) if self.causal_graph else 0
        }
        
        self.performance_history.append(performance_entry)
        
        # Keep only last 50 runs to prevent memory issues
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
    
    def _analyze_performance_trends(self):
        """
        Analyze performance trends to suggest improvements.
        """
        if len(self.performance_history) < 2:
            return "Insufficient history for trend analysis"
        
        recent_runs = self.performance_history[-5:]  # Last 5 runs
        
        # Calculate trend in F1 score
        f1_scores = [run['metrics']['overall']['f1'] for run in recent_runs]
        f1_trend = np.polyfit(range(len(f1_scores)), f1_scores, 1)[0]
        
        # Calculate trend in recall
        recall_scores = [run['metrics']['overall']['recall'] for run in recent_runs]
        recall_trend = np.polyfit(range(len(recall_scores)), recall_scores, 1)[0]
        
        analysis = []
        
        if f1_trend > 0.01:
            analysis.append(" F1-score is improving over time")
        elif f1_trend < -0.01:
            analysis.append(" F1-score is declining - consider retraining")
        else:
            analysis.append(" F1-score is stable")
        
        if recall_trend > 0.01:
            analysis.append(" Recall is improving over time")
        elif recall_trend < -0.01:
            analysis.append(" Recall is declining - consider adjusting thresholds")
        else:
            analysis.append(" Recall is stable")
        
        # Suggest improvements
        latest_f1 = f1_scores[-1]
        latest_recall = recall_scores[-1]
        
        if latest_f1 < 0.6:
            analysis.append(" Consider increasing model complexity or feature engineering")
        
        if latest_recall < 0.7:
            analysis.append(" Consider lowering anomaly thresholds for better recall")
        
        return "\n".join(analysis)
    
    def run_full_pipeline(self, verify=False, run_adversarial=False, adversarial_epsilon=0.15,
                          adversarial_subsample=512):
        """
        Run the complete anomaly detection pipeline with adaptive learning.
        
        Args:
            verify: If True, run Phase 2 formal verification and write certificates.
            run_adversarial: If True (and verify is True), run FGSM/PGD robustness checks.
            adversarial_epsilon: L_inf radius in scaled feature space for attacks.
            adversarial_subsample: Max rows used for adversarial gradient estimation (speed).
        """
        print("Starting SCM-based Anomaly Detection Pipeline with Adaptive Learning")
        print("="*70)
        
        # Load and preprocess data
        normal_data, attack_data, feature_columns = self.load_and_preprocess_data()
        
        # Adaptive learning: adapt existing models with new data
        if self.adaptive_learning and self.models:
            self._adapt_models_with_new_data(normal_data, feature_columns)
        
        # Learn causal graph (or use existing if adaptive learning)
        if not self.causal_graph:
            self.learn_causal_graph(normal_data, feature_columns)
        else:
            print(" Using existing causal graph for adaptive learning")
        
        # Train structural models (or adapt existing ones)
        if not self.models:
            self.train_structural_models(normal_data, feature_columns)
        else:
            print(" Using existing models with adaptive learning")
        
        # Detect anomalies
        anomaly_scores, anomaly_flags, causal_paths = self.detect_anomalies(attack_data, feature_columns)
        
        # Evaluate performance
        metrics = self.evaluate_performance(attack_data, anomaly_flags, feature_columns)
        
        # Update performance history for adaptive learning
        if self.adaptive_learning:
            self._update_performance_history(metrics)
        
        # Generate explanations
        self.generate_explanations(attack_data, anomaly_flags, causal_paths, feature_columns)
        
        # Save adaptive state for future runs
        if self.adaptive_learning:
            self._save_adaptive_state()
        
        # Analyze performance trends
        if self.adaptive_learning and len(self.performance_history) > 1:
            trend_analysis = self._analyze_performance_trends()
            print("\n" + "="*50)
            print("ADAPTIVE LEARNING ANALYSIS")
            print("="*50)
            print(trend_analysis)
        
        # Print summary
        print("\n" + "="*50)
        print("PIPELINE SUMMARY")
        print("="*50)
        print(f"Features processed: {len(feature_columns)}")
        print(f"Causal graph edges: {len(self.causal_graph.edges) if self.causal_graph else 0}")
        print(f"Models trained: {len(self.models)}")
        print(f"Overall Precision: {metrics['overall']['precision']:.3f}")
        print(f"Overall Recall: {metrics['overall']['recall']:.3f}")
        print(f"Overall F1-Score: {metrics['overall']['f1']:.3f}")
        print(f"Adaptive Learning: {'Enabled' if self.adaptive_learning else 'Disabled'}")
        print(f"Performance History: {len(self.performance_history)} runs")
        print(f"Results saved to: {self.output_dir}")
        
        out = {
            'metrics': metrics,
            'anomaly_scores': anomaly_scores,
            'anomaly_flags': anomaly_flags,
            'causal_paths': causal_paths,
            'performance_history': self.performance_history
        }
        
        if verify:
            if self._normal_training_scaled is None:
                self._cache_training_matrix(normal_data, feature_columns)
            from formal_verification import FormalVerifier
            true_labels = attack_data['Label'].values if 'Label' in attack_data.columns else np.zeros(len(attack_data))
            predictions = np.any(anomaly_flags, axis=1).astype(int)
            verifier = FormalVerifier()
            verification_results = verifier.run_verification_suite(
                detector=self,
                test_data=attack_data,
                true_labels=true_labels,
                predictions=predictions,
                feature_cols=feature_columns,
                anomaly_scores=anomaly_scores,
                anomaly_flags=anomaly_flags,
                normal_reference_data=normal_data,
                run_adversarial=run_adversarial,
                adversarial_epsilon=adversarial_epsilon,
                adversarial_subsample=adversarial_subsample,
            )
            out['verification'] = verification_results
            print(
                f"\nVerification: {verification_results['properties_verified']}/"
                f"{verification_results['properties_tested']} properties passed "
                f"(certificate: outputs/verification/)"
            )
        
        return out


def main():
    """
    Main function to run the anomaly detection pipeline.
    """
    # Initialize detector
    detector = SWaTCausalDetector()
    
    # Run full pipeline
    results = detector.run_full_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()
