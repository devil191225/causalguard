"""
Edge Case Tester for Causal Anomaly Detection
===========================================

This module generates extremely challenging and realistic edge cases to test
the robustness of our causal anomaly detection system. It simulates real-world
industrial scenarios that are notoriously difficult to detect.

Author: Aditya Srikar Konduri
Date: 10/19/2025
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class EdgeCaseTester:
    """
    Generates realistic edge cases for testing causal anomaly detection.
    """
    
    def __init__(self, output_dir='outputs/edge_cases'):
        """
        Initialize the edge case tester.
        
        Args:
            output_dir (str): Directory to save edge case data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Industrial process parameters (realistic ranges)
        self.process_params = {
            'boiler': {
                'pressure_range': (0.0, 1.0),
                'temperature_range': (0.0, 1.0),
                'water_level_range': (0.0, 1.0),
                'flow_rate_range': (0.0, 1.0),
                'valve_position_range': (0.0, 1.0)
            },
            'turbine': {
                'speed_range': (0.5, 1.5),
                'power_range': (0.0, 1.0),
                'steam_pressure_range': (0.0, 1.0),
                'temperature_range': (0.0, 1.0),
                'vibration_range': (0.0, 1.0)
            },
            'water_treatment': {
                'pump_speed_range': (0.0, 1.0),
                'water_level_range': (0.0, 1.0),
                'flow_rate_range': (0.0, 1.0),
                'pressure_range': (0.0, 1.0),
                'valve_position_range': (0.0, 1.0)
            },
            'hil_simulation': {
                'load_demand_range': (-0.5, 0.5),
                'power_generation_range': (0.0, 1.0),
                'reservoir_level_range': (0.0, 1.0),
                'water_storage_range': (0.0, 1.0),
                'control_signal_range': (0.0, 1.0)
            }
        }
    
    def generate_timestamps(self, n_samples, start_time=None, freq='1s'):
        """
        Generate realistic timestamps for industrial data.
        """
        if start_time is None:
            start_time = datetime(2023, 1, 1, 0, 0, 0)
        
        return pd.date_range(start_time, periods=n_samples, freq=freq)
    
    def create_edge_case_1_gradual_drift(self, n_samples=10000):
        """
        Edge Case 1: Gradual Drift Attack
        =================================
        This simulates a sophisticated attack where process parameters
        gradually drift over time, mimicking normal aging or maintenance
        issues. This is extremely difficult to detect as it appears normal.
        """
        print("🔬 Generating Edge Case 1: Gradual Drift Attack")
        print("="*60)
        
        timestamps = self.generate_timestamps(n_samples)
        
        # Create base normal data
        data = {}
        
        # Boiler process with gradual drift
        base_pressure = 0.1
        drift_rate = 0.00001  # Very slow drift
        data['P1_B2004'] = base_pressure + drift_rate * np.arange(n_samples) + np.random.normal(0, 0.02, n_samples)
        
        # Temperature follows pressure with slight lag
        data['P1_B2005'] = 0.8 * data['P1_B2004'] + 0.2 * np.random.normal(0.2, 0.05, n_samples)
        
        # Water level inversely related to pressure
        data['P1_B2006'] = 0.3 - 0.2 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        
        # Flow rate increases with pressure
        data['P1_B2007'] = 0.4 * data['P1_B2004'] + 0.3 * np.random.normal(0, 0.04, n_samples)
        
        # Valve position follows flow
        data['P1_B2008'] = 0.6 * data['P1_B2007'] + 0.2 * np.random.normal(0, 0.03, n_samples)
        
        # Turbine process with correlated drift
        data['P2_B2016'] = 1.0 + 0.3 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.05, n_samples)
        data['P2_B2017'] = 0.8 + 0.4 * data['P1_B2004'] + 0.2 * np.random.normal(0, 0.06, n_samples)
        data['P2_B2018'] = 0.6 + 0.3 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.04, n_samples)
        data['P2_B2019'] = 0.4 + 0.2 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.03, n_samples)
        data['P2_B2020'] = 0.7 + 0.1 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.04, n_samples)
        
        # Water treatment with delayed response
        data['P3_B2021'] = 0.5 + 0.1 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.05, n_samples)
        data['P3_B2022'] = 0.3 + 0.05 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.03, n_samples)
        data['P3_B2023'] = 0.4 + 0.1 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.04, n_samples)
        data['P3_B2024'] = 0.2 + 0.05 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.02, n_samples)
        data['P3_B2025'] = 0.6 + 0.1 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.04, n_samples)
        
        # HIL simulation with complex relationships
        data['P4_HT_LD'] = 0.0 + 0.1 * data['P2_B2016'] + 0.1 * np.random.normal(0, 0.05, n_samples)
        data['P4_HT_PG'] = 0.5 + 0.3 * data['P2_B2016'] + 0.2 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.06, n_samples)
        data['P4_HT_RS'] = 0.3 + 0.2 * data['P3_B2021'] + 0.1 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.04, n_samples)
        data['P4_HT_WS'] = 0.4 + 0.2 * data['P3_B2023'] + 0.1 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.05, n_samples)
        data['P4_HT_CT'] = 0.2 + 0.1 * data['P1_B2004'] + 0.1 * np.random.normal(0, 0.03, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['time'] = timestamps
        
        # Add attack labels - gradual drift is hard to detect
        attack_start = int(0.3 * n_samples)  # Start attack at 30% of data
        attack_end = int(0.8 * n_samples)    # End attack at 80% of data
        
        df['attack'] = 0
        df.loc[attack_start:attack_end, 'attack'] = 1
        
        # Process-specific attacks
        df['attack_P1'] = 0
        df['attack_P2'] = 0
        df['attack_P3'] = 0
        
        # Gradual attacks affect all processes
        df.loc[attack_start:attack_end, 'attack_P1'] = 1
        df.loc[attack_start:attack_end, 'attack_P2'] = 1
        df.loc[attack_start:attack_end, 'attack_P3'] = 1
        
        # Add Label column
        df['Label'] = df['attack']
        
        print(f"✓ Generated {n_samples} samples with gradual drift attack")
        print(f"✓ Attack period: {attack_start} to {attack_end} ({attack_end-attack_start} samples)")
        print(f"✓ Drift rate: {drift_rate} per sample")
        
        return df
    
    def create_edge_case_2_intermittent_attacks(self, n_samples=10000):
        """
        Edge Case 2: Intermittent Attacks
        =================================
        This simulates attacks that occur sporadically, making them
        extremely difficult to detect as they appear as random noise.
        """
        print("🔬 Generating Edge Case 2: Intermittent Attacks")
        print("="*60)
        
        timestamps = self.generate_timestamps(n_samples)
        
        # Create base normal data
        data = {}
        
        # Boiler process
        data['P1_B2004'] = 0.1 + 0.05 * np.sin(np.arange(n_samples) * 0.01) + np.random.normal(0, 0.02, n_samples)
        data['P1_B2005'] = 0.2 + 0.1 * np.sin(np.arange(n_samples) * 0.01 + np.pi/4) + np.random.normal(0, 0.03, n_samples)
        data['P1_B2006'] = 0.15 + 0.05 * np.cos(np.arange(n_samples) * 0.01) + np.random.normal(0, 0.02, n_samples)
        data['P1_B2007'] = 0.3 + 0.1 * np.sin(np.arange(n_samples) * 0.01 + np.pi/2) + np.random.normal(0, 0.04, n_samples)
        data['P1_B2008'] = 0.25 + 0.05 * np.cos(np.arange(n_samples) * 0.01 + np.pi/3) + np.random.normal(0, 0.03, n_samples)
        
        # Turbine process
        data['P2_B2016'] = 1.0 + 0.1 * np.sin(np.arange(n_samples) * 0.01) + 0.3 * data['P1_B2004'] + np.random.normal(0, 0.05, n_samples)
        data['P2_B2017'] = 0.8 + 0.1 * np.cos(np.arange(n_samples) * 0.01) + 0.4 * data['P1_B2004'] + np.random.normal(0, 0.06, n_samples)
        data['P2_B2018'] = 0.6 + 0.05 * np.sin(np.arange(n_samples) * 0.01) + 0.3 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P2_B2019'] = 0.4 + 0.05 * np.cos(np.arange(n_samples) * 0.01) + 0.2 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        data['P2_B2020'] = 0.7 + 0.1 * np.sin(np.arange(n_samples) * 0.01) + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        
        # Water treatment
        data['P3_B2021'] = 0.5 + 0.1 * np.sin(np.arange(n_samples) * 0.01) + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.05, n_samples)
        data['P3_B2022'] = 0.3 + 0.05 * np.cos(np.arange(n_samples) * 0.01) + 0.05 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        data['P3_B2023'] = 0.4 + 0.1 * np.sin(np.arange(n_samples) * 0.01) + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P3_B2024'] = 0.2 + 0.05 * np.cos(np.arange(n_samples) * 0.01) + 0.05 * data['P1_B2004'] + np.random.normal(0, 0.02, n_samples)
        data['P3_B2025'] = 0.6 + 0.1 * np.sin(np.arange(n_samples) * 0.01) + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        
        # HIL simulation
        data['P4_HT_LD'] = 0.0 + 0.1 * data['P2_B2016'] + np.random.normal(0, 0.05, n_samples)
        data['P4_HT_PG'] = 0.5 + 0.3 * data['P2_B2016'] + 0.2 * data['P1_B2004'] + np.random.normal(0, 0.06, n_samples)
        data['P4_HT_RS'] = 0.3 + 0.2 * data['P3_B2021'] + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P4_HT_WS'] = 0.4 + 0.2 * data['P3_B2023'] + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.05, n_samples)
        data['P4_HT_CT'] = 0.2 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['time'] = timestamps
        
        # Create intermittent attack pattern
        attack_periods = []
        current_pos = 0
        
        while current_pos < n_samples:
            # Random gap between attacks (100-500 samples)
            gap = np.random.randint(100, 500)
            current_pos += gap
            
            if current_pos < n_samples:
                # Random attack duration (10-50 samples)
                attack_duration = np.random.randint(10, 50)
                attack_end = min(current_pos + attack_duration, n_samples)
                attack_periods.append((current_pos, attack_end))
                current_pos = attack_end
        
        # Add attack labels
        df['attack'] = 0
        for start, end in attack_periods:
            df.loc[start:end, 'attack'] = 1
        
        # Process-specific attacks (random assignment)
        df['attack_P1'] = 0
        df['attack_P2'] = 0
        df['attack_P3'] = 0
        
        for start, end in attack_periods:
            # Randomly assign which processes are affected
            if np.random.random() < 0.7:
                df.loc[start:end, 'attack_P1'] = 1
            if np.random.random() < 0.5:
                df.loc[start:end, 'attack_P2'] = 1
            if np.random.random() < 0.6:
                df.loc[start:end, 'attack_P3'] = 1
        
        # Add Label column
        df['Label'] = df['attack']
        
        print(f"✓ Generated {n_samples} samples with intermittent attacks")
        print(f"✓ Number of attack periods: {len(attack_periods)}")
        print(f"✓ Total attack samples: {df['attack'].sum()}")
        
        return df
    
    def create_edge_case_3_mimicry_attacks(self, n_samples=10000):
        """
        Edge Case 3: Mimicry Attacks
        ============================
        This simulates attacks that mimic normal operational patterns,
        making them extremely difficult to detect as they appear legitimate.
        """
        print("🔬 Generating Edge Case 3: Mimicry Attacks")
        print("="*60)
        
        timestamps = self.generate_timestamps(n_samples)
        
        # Create base normal data with realistic patterns
        data = {}
        
        # Boiler process with realistic operational patterns
        t = np.arange(n_samples)
        
        # Normal operational cycles (daily patterns)
        daily_cycle = 0.1 * np.sin(2 * np.pi * t / (24 * 3600))  # 24-hour cycle
        weekly_cycle = 0.05 * np.sin(2 * np.pi * t / (7 * 24 * 3600))  # Weekly cycle
        
        data['P1_B2004'] = 0.1 + daily_cycle + weekly_cycle + np.random.normal(0, 0.02, n_samples)
        data['P1_B2005'] = 0.2 + 0.8 * data['P1_B2004'] + 0.1 * daily_cycle + np.random.normal(0, 0.03, n_samples)
        data['P1_B2006'] = 0.15 - 0.2 * data['P1_B2004'] + 0.05 * daily_cycle + np.random.normal(0, 0.02, n_samples)
        data['P1_B2007'] = 0.3 + 0.4 * data['P1_B2004'] + 0.1 * daily_cycle + np.random.normal(0, 0.04, n_samples)
        data['P1_B2008'] = 0.25 + 0.6 * data['P1_B2007'] + 0.05 * daily_cycle + np.random.normal(0, 0.03, n_samples)
        
        # Turbine process with realistic power generation patterns
        data['P2_B2016'] = 1.0 + 0.3 * data['P1_B2004'] + 0.1 * daily_cycle + np.random.normal(0, 0.05, n_samples)
        data['P2_B2017'] = 0.8 + 0.4 * data['P1_B2004'] + 0.2 * daily_cycle + np.random.normal(0, 0.06, n_samples)
        data['P2_B2018'] = 0.6 + 0.3 * data['P1_B2004'] + 0.1 * daily_cycle + np.random.normal(0, 0.04, n_samples)
        data['P2_B2019'] = 0.4 + 0.2 * data['P1_B2004'] + 0.05 * daily_cycle + np.random.normal(0, 0.03, n_samples)
        data['P2_B2020'] = 0.7 + 0.1 * data['P1_B2004'] + 0.1 * daily_cycle + np.random.normal(0, 0.04, n_samples)
        
        # Water treatment with realistic patterns
        data['P3_B2021'] = 0.5 + 0.1 * data['P1_B2004'] + 0.1 * daily_cycle + np.random.normal(0, 0.05, n_samples)
        data['P3_B2022'] = 0.3 + 0.05 * data['P1_B2004'] + 0.05 * daily_cycle + np.random.normal(0, 0.03, n_samples)
        data['P3_B2023'] = 0.4 + 0.1 * data['P1_B2004'] + 0.1 * daily_cycle + np.random.normal(0, 0.04, n_samples)
        data['P3_B2024'] = 0.2 + 0.05 * data['P1_B2004'] + 0.05 * daily_cycle + np.random.normal(0, 0.02, n_samples)
        data['P3_B2025'] = 0.6 + 0.1 * data['P1_B2004'] + 0.1 * daily_cycle + np.random.normal(0, 0.04, n_samples)
        
        # HIL simulation with realistic power grid patterns
        data['P4_HT_LD'] = 0.0 + 0.1 * data['P2_B2016'] + 0.2 * daily_cycle + np.random.normal(0, 0.05, n_samples)
        data['P4_HT_PG'] = 0.5 + 0.3 * data['P2_B2016'] + 0.2 * data['P1_B2004'] + 0.3 * daily_cycle + np.random.normal(0, 0.06, n_samples)
        data['P4_HT_RS'] = 0.3 + 0.2 * data['P3_B2021'] + 0.1 * data['P1_B2004'] + 0.1 * daily_cycle + np.random.normal(0, 0.04, n_samples)
        data['P4_HT_WS'] = 0.4 + 0.2 * data['P3_B2023'] + 0.1 * data['P1_B2004'] + 0.1 * daily_cycle + np.random.normal(0, 0.05, n_samples)
        data['P4_HT_CT'] = 0.2 + 0.1 * data['P1_B2004'] + 0.05 * daily_cycle + np.random.normal(0, 0.03, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['time'] = timestamps
        
        # Create mimicry attacks that follow normal patterns
        attack_periods = []
        current_pos = 0
        
        while current_pos < n_samples:
            # Random gap between attacks (500-2000 samples)
            gap = np.random.randint(500, 2000)
            current_pos += gap
            
            if current_pos < n_samples:
                # Random attack duration (100-500 samples)
                attack_duration = np.random.randint(100, 500)
                attack_end = min(current_pos + attack_duration, n_samples)
                attack_periods.append((current_pos, attack_end))
                current_pos = attack_end
        
        # Add attack labels
        df['attack'] = 0
        for start, end in attack_periods:
            df.loc[start:end, 'attack'] = 1
        
        # Process-specific attacks
        df['attack_P1'] = 0
        df['attack_P2'] = 0
        df['attack_P3'] = 0
        
        for start, end in attack_periods:
            # Mimicry attacks affect all processes but maintain relationships
            df.loc[start:end, 'attack_P1'] = 1
            df.loc[start:end, 'attack_P2'] = 1
            df.loc[start:end, 'attack_P3'] = 1
        
        # Add Label column
        df['Label'] = df['attack']
        
        print(f"✓ Generated {n_samples} samples with mimicry attacks")
        print(f"✓ Number of attack periods: {len(attack_periods)}")
        print(f"✓ Total attack samples: {df['attack'].sum()}")
        
        return df
    
    def create_edge_case_4_cascade_failures(self, n_samples=10000):
        """
        Edge Case 4: Cascade Failures
        =============================
        This simulates cascading failures where one component failure
        triggers a chain reaction of failures across the system.
        """
        print("🔬 Generating Edge Case 4: Cascade Failures")
        print("="*60)
        
        timestamps = self.generate_timestamps(n_samples)
        
        # Create base normal data
        data = {}
        
        # Boiler process
        data['P1_B2004'] = 0.1 + 0.05 * np.sin(np.arange(n_samples) * 0.01) + np.random.normal(0, 0.02, n_samples)
        data['P1_B2005'] = 0.2 + 0.8 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        data['P1_B2006'] = 0.15 - 0.2 * data['P1_B2004'] + np.random.normal(0, 0.02, n_samples)
        data['P1_B2007'] = 0.3 + 0.4 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P1_B2008'] = 0.25 + 0.6 * data['P1_B2007'] + np.random.normal(0, 0.03, n_samples)
        
        # Turbine process
        data['P2_B2016'] = 1.0 + 0.3 * data['P1_B2004'] + np.random.normal(0, 0.05, n_samples)
        data['P2_B2017'] = 0.8 + 0.4 * data['P1_B2004'] + np.random.normal(0, 0.06, n_samples)
        data['P2_B2018'] = 0.6 + 0.3 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P2_B2019'] = 0.4 + 0.2 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        data['P2_B2020'] = 0.7 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        
        # Water treatment
        data['P3_B2021'] = 0.5 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.05, n_samples)
        data['P3_B2022'] = 0.3 + 0.05 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        data['P3_B2023'] = 0.4 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P3_B2024'] = 0.2 + 0.05 * data['P1_B2004'] + np.random.normal(0, 0.02, n_samples)
        data['P3_B2025'] = 0.6 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        
        # HIL simulation
        data['P4_HT_LD'] = 0.0 + 0.1 * data['P2_B2016'] + np.random.normal(0, 0.05, n_samples)
        data['P4_HT_PG'] = 0.5 + 0.3 * data['P2_B2016'] + 0.2 * data['P1_B2004'] + np.random.normal(0, 0.06, n_samples)
        data['P4_HT_RS'] = 0.3 + 0.2 * data['P3_B2021'] + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P4_HT_WS'] = 0.4 + 0.2 * data['P3_B2023'] + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.05, n_samples)
        data['P4_HT_CT'] = 0.2 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['time'] = timestamps
        
        # Create cascade failure attacks
        attack_periods = []
        current_pos = 0
        
        while current_pos < n_samples:
            # Random gap between attacks (1000-3000 samples)
            gap = np.random.randint(1000, 3000)
            current_pos += gap
            
            if current_pos < n_samples:
                # Random attack duration (200-1000 samples)
                attack_duration = np.random.randint(200, 1000)
                attack_end = min(current_pos + attack_duration, n_samples)
                attack_periods.append((current_pos, attack_end))
                current_pos = attack_end
        
        # Add attack labels
        df['attack'] = 0
        for start, end in attack_periods:
            df.loc[start:end, 'attack'] = 1
        
        # Process-specific attacks with cascade effects
        df['attack_P1'] = 0
        df['attack_P2'] = 0
        df['attack_P3'] = 0
        
        for start, end in attack_periods:
            # Cascade: P1 -> P2 -> P3 with delays
            df.loc[start:end, 'attack_P1'] = 1
            
            # P2 affected after 50 samples
            p2_start = min(start + 50, end)
            df.loc[p2_start:end, 'attack_P2'] = 1
            
            # P3 affected after 100 samples
            p3_start = min(start + 100, end)
            df.loc[p3_start:end, 'attack_P3'] = 1
        
        # Add Label column
        df['Label'] = df['attack']
        
        print(f"✓ Generated {n_samples} samples with cascade failures")
        print(f"✓ Number of attack periods: {len(attack_periods)}")
        print(f"✓ Total attack samples: {df['attack'].sum()}")
        
        return df
    
    def create_edge_case_5_adversarial_attacks(self, n_samples=10000):
        """
        Edge Case 5: Adversarial Attacks
        =================================
        This simulates sophisticated adversarial attacks designed to
        evade detection by exploiting the model's weaknesses.
        """
        print("🔬 Generating Edge Case 5: Adversarial Attacks")
        print("="*60)
        
        timestamps = self.generate_timestamps(n_samples)
        
        # Create base normal data
        data = {}
        
        # Boiler process
        data['P1_B2004'] = 0.1 + 0.05 * np.sin(np.arange(n_samples) * 0.01) + np.random.normal(0, 0.02, n_samples)
        data['P1_B2005'] = 0.2 + 0.8 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        data['P1_B2006'] = 0.15 - 0.2 * data['P1_B2004'] + np.random.normal(0, 0.02, n_samples)
        data['P1_B2007'] = 0.3 + 0.4 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P1_B2008'] = 0.25 + 0.6 * data['P1_B2007'] + np.random.normal(0, 0.03, n_samples)
        
        # Turbine process
        data['P2_B2016'] = 1.0 + 0.3 * data['P1_B2004'] + np.random.normal(0, 0.05, n_samples)
        data['P2_B2017'] = 0.8 + 0.4 * data['P1_B2004'] + np.random.normal(0, 0.06, n_samples)
        data['P2_B2018'] = 0.6 + 0.3 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P2_B2019'] = 0.4 + 0.2 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        data['P2_B2020'] = 0.7 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        
        # Water treatment
        data['P3_B2021'] = 0.5 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.05, n_samples)
        data['P3_B2022'] = 0.3 + 0.05 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        data['P3_B2023'] = 0.4 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P3_B2024'] = 0.2 + 0.05 * data['P1_B2004'] + np.random.normal(0, 0.02, n_samples)
        data['P3_B2025'] = 0.6 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        
        # HIL simulation
        data['P4_HT_LD'] = 0.0 + 0.1 * data['P2_B2016'] + np.random.normal(0, 0.05, n_samples)
        data['P4_HT_PG'] = 0.5 + 0.3 * data['P2_B2016'] + 0.2 * data['P1_B2004'] + np.random.normal(0, 0.06, n_samples)
        data['P4_HT_RS'] = 0.3 + 0.2 * data['P3_B2021'] + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.04, n_samples)
        data['P4_HT_WS'] = 0.4 + 0.2 * data['P3_B2023'] + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.05, n_samples)
        data['P4_HT_CT'] = 0.2 + 0.1 * data['P1_B2004'] + np.random.normal(0, 0.03, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['time'] = timestamps
        
        # Create adversarial attack pattern
        attack_periods = []
        current_pos = 0
        
        while current_pos < n_samples:
            # Random gap between attacks (500-1500 samples)
            gap = np.random.randint(500, 1500)
            current_pos += gap
            
            if current_pos < n_samples:
                # Random attack duration (50-200 samples)
                attack_duration = np.random.randint(50, 200)
                attack_end = min(current_pos + attack_duration, n_samples)
                attack_periods.append((current_pos, attack_end))
                current_pos = attack_end
        
        # Add attack labels
        df['attack'] = 0
        for start, end in attack_periods:
            df.loc[start:end, 'attack'] = 1
        
        # Process-specific attacks
        df['attack_P1'] = 0
        df['attack_P2'] = 0
        df['attack_P3'] = 0
        
        for start, end in attack_periods:
            # Adversarial attacks affect all processes
            df.loc[start:end, 'attack_P1'] = 1
            df.loc[start:end, 'attack_P2'] = 1
            df.loc[start:end, 'attack_P3'] = 1
        
        # Add Label column
        df['Label'] = df['attack']
        
        print(f"✓ Generated {n_samples} samples with adversarial attacks")
        print(f"✓ Number of attack periods: {len(attack_periods)}")
        print(f"✓ Total attack samples: {df['attack'].sum()}")
        
        return df
    
    def run_all_edge_cases(self):
        """
        Run all edge case tests and save results.
        """
        print("🧪 EDGE CASE TESTING SUITE")
        print("="*60)
        print("Testing the most challenging scenarios for causal anomaly detection")
        print()
        
        edge_cases = {
            'gradual_drift': self.create_edge_case_1_gradual_drift,
            'intermittent': self.create_edge_case_2_intermittent_attacks,
            'mimicry': self.create_edge_case_3_mimicry_attacks,
            'cascade': self.create_edge_case_4_cascade_failures,
            'adversarial': self.create_edge_case_5_adversarial_attacks
        }
        
        results = {}
        
        for case_name, case_func in edge_cases.items():
            print(f"\n🔬 Running {case_name} edge case...")
            print("-" * 40)
            
            try:
                df = case_func()
                
                # Save data
                output_file = os.path.join(self.output_dir, 'data', f'{case_name}_data.csv')
                df.to_csv(output_file, index=False)
                
                # Calculate statistics
                stats = {
                    'total_samples': len(df),
                    'attack_samples': df['attack'].sum(),
                    'attack_percentage': (df['attack'].sum() / len(df)) * 100,
                    'p1_attacks': df['attack_P1'].sum(),
                    'p2_attacks': df['attack_P2'].sum(),
                    'p3_attacks': df['attack_P3'].sum()
                }
                
                results[case_name] = {
                    'data': df,
                    'stats': stats,
                    'file': output_file
                }
                
                print(f"✓ {case_name} completed successfully")
                print(f"  - Total samples: {stats['total_samples']}")
                print(f"  - Attack samples: {stats['attack_samples']} ({stats['attack_percentage']:.1f}%)")
                print(f"  - P1 attacks: {stats['p1_attacks']}")
                print(f"  - P2 attacks: {stats['p2_attacks']}")
                print(f"  - P3 attacks: {stats['p3_attacks']}")
                print(f"  - Saved to: {output_file}")
                
            except Exception as e:
                print(f"✗ {case_name} failed: {e}")
                results[case_name] = {'error': str(e)}
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results):
        """
        Generate a summary report of all edge case tests.
        """
        print("\n📊 EDGE CASE TESTING SUMMARY")
        print("="*60)
        
        successful_cases = [name for name, result in results.items() if 'error' not in result]
        failed_cases = [name for name, result in results.items() if 'error' in result]
        
        print(f"✅ Successful cases: {len(successful_cases)}")
        print(f"❌ Failed cases: {len(failed_cases)}")
        
        if successful_cases:
            print("\n📈 Case Statistics:")
            for case_name in successful_cases:
                stats = results[case_name]['stats']
                print(f"  {case_name}:")
                print(f"    - Attack rate: {stats['attack_percentage']:.1f}%")
                print(f"    - Total samples: {stats['total_samples']}")
        
        if failed_cases:
            print("\n❌ Failed Cases:")
            for case_name in failed_cases:
                print(f"  {case_name}: {results[case_name]['error']}")
        
        print(f"\n💾 All data saved to: {self.output_dir}")
        print("🎯 These edge cases represent the most challenging scenarios for anomaly detection")
        print("🔍 Test your causal detector with these datasets to evaluate robustness")


def main():
    """
    Main function to run edge case testing.
    """
    tester = EdgeCaseTester()
    results = tester.run_all_edge_cases()
    
    print("\n🎉 Edge case testing complete!")
    print("Use these datasets to test your causal anomaly detection system:")
    print("python src/causal_detector.py")


if __name__ == "__main__":
    main()


