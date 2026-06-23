#!/usr/bin/env python3
"""
REALISTIC ICS ATTACK GENERATOR
============================================================
Generates challenging, realistic synthetic attack scenarios based on
real ICS attack patterns documented in literature.

Attacks are designed to be SUBTLE and GRADUAL - not obvious 3-sigma deviations.

Author: Aditya Srikar Konduri
Date: October 19, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path

class RealisticAttackGenerator:
    """Generate realistic ICS attack scenarios."""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.output_dir = Path('data/realistic_attacks')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_attacks(self):
        """Generate all realistic attack scenarios."""
        print("=" * 70)
        print("GENERATING REALISTIC ICS ATTACK SCENARIOS")
        print("=" * 70)
        
        scenarios = [
            ('stealthy_setpoint_change', self.generate_stealthy_setpoint_change),
            ('sensor_spoofing', self.generate_sensor_spoofing_attack),
            ('slow_ramp_attack', self.generate_slow_ramp_attack),
            ('intermittent_pulse', self.generate_intermittent_pulse_attack),
            ('model_informed_attack', self.generate_model_informed_attack),
        ]
        
        results = {}
        for name, generator in scenarios:
            print(f"\nGenerating: {name}")
            print("-" * 70)
            normal_df, attack_df = generator()
            
            # Save
            normal_path = self.output_dir / f'{name}_normal.csv'
            attack_path = self.output_dir / f'{name}_attack.csv'
            
            normal_df.to_csv(normal_path, index=False)
            attack_df.to_csv(attack_path, index=False)
            
            attack_count = attack_df['Label'].sum()
            attack_pct = (attack_count / len(attack_df)) * 100
            
            results[name] = {
                'normal_samples': len(normal_df),
                'attack_samples': len(attack_df),
                'attack_count': int(attack_count),
                'attack_percentage': f'{attack_pct:.1f}%'
            }
            
            print(f"  Normal: {len(normal_df)} samples")
            print(f"  Attack: {len(attack_df)} samples ({attack_count} anomalies, {attack_pct:.1f}%)")
            print(f"  Saved to: {self.output_dir}")
        
        return results
    
    def _generate_baseline_normal_data(self, n_samples=5000):
        """Generate baseline normal operation data with realistic ICS characteristics."""
        # Daily cycle (24 hours)
        t = np.linspace(0, 10 * 2 * np.pi, n_samples)  # 10 days
        
        # Process 1: Pressure sensors with daily cycle
        P1_base = 50 + 5 * np.sin(t) + 0.5 * np.random.randn(n_samples)
        P1_B2004 = P1_base + 0.3 * np.random.randn(n_samples)
        P1_B2005 = 0.95 * P1_base + 0.3 * np.random.randn(n_samples)
        P1_B2006 = P1_base + 2 * np.sin(t * 2) + 0.4 * np.random.randn(n_samples)
        
        # Process 2: Flow sensors (dependent on P1)
        P2_B2016 = 0.8 * P1_base + 3 * np.cos(t) + 0.4 * np.random.randn(n_samples)
        P2_B2017 = 0.9 * P2_B2016 + 0.3 * np.random.randn(n_samples)
        
        # Process 3: Temperature sensors (slower dynamics)
        P3_temp_base = 70 + 3 * np.sin(t / 2)
        P3_B2021 = P3_temp_base + 0.5 * np.random.randn(n_samples)
        P3_B2022 = P3_temp_base + 1 * np.cos(t / 3) + 0.5 * np.random.randn(n_samples)
        
        # Process 4: Control signals
        P4_HT_LD = 30 + 2 * np.sin(t) + 0.3 * np.random.randn(n_samples)
        P4_HT_PG = 0.7 * P4_HT_LD + 0.2 * np.random.randn(n_samples)
        
        df = pd.DataFrame({
            'P1_B2004': P1_B2004,
            'P1_B2005': P1_B2005,
            'P1_B2006': P1_B2006,
            'P2_B2016': P2_B2016,
            'P2_B2017': P2_B2017,
            'P3_B2021': P3_B2021,
            'P3_B2022': P3_B2022,
            'P4_HT_LD': P4_HT_LD,
            'P4_HT_PG': P4_HT_PG,
            'Label': 0
        })
        
        return df
    
    def generate_stealthy_setpoint_change(self):
        """
        Attack: Slowly change setpoint by 10% over 500 samples.
        Detection difficulty: HIGH - stays within 2-sigma of normal.
        """
        normal_df = self._generate_baseline_normal_data(3000)
        attack_df = normal_df.copy()
        
        # Attack window: 1000-1500
        attack_start, attack_end = 1000, 1500
        attack_duration = attack_end - attack_start
        
        # Gradual 10% increase in P1_B2004 (primary pressure)
        original = attack_df.loc[attack_start:attack_end, 'P1_B2004'].values
        gradient = np.linspace(0, 0.1, len(original))
        attack_df.loc[attack_start:attack_end, 'P1_B2004'] = original * (1 + gradient)
        
        # Cascading effects (with delays and attenuation)
        for i in range(attack_start, attack_end):
            delay = min(50, i - attack_start)
            if delay > 0:
                # P2 follows P1 with delay
                attack_df.loc[i, 'P2_B2016'] *= (1 + 0.05 * (delay / 50))
                # P4 control tries to compensate
                attack_df.loc[i, 'P4_HT_LD'] *= (1 - 0.03 * (delay / 50))
        
        attack_df.loc[attack_start:attack_end, 'Label'] = 1
        
        return normal_df, attack_df
    
    def generate_sensor_spoofing_attack(self):
        """
        Attack: Spoof sensor readings to hide actual system state.
        Detection difficulty: VERY HIGH - readings look normal but system is not.
        """
        normal_df = self._generate_baseline_normal_data(3000)
        attack_df = normal_df.copy()
        
        # Attack: 1200-1800
        attack_start, attack_end = 1200, 1800
        
        # Keep P1_B2004 reading constant (spoofed) while actual P2/P3 drift
        spoofed_value = attack_df.loc[attack_start, 'P1_B2004']
        attack_df.loc[attack_start:attack_end, 'P1_B2004'] = spoofed_value + 0.1 * np.random.randn(attack_end - attack_start + 1)
        
        # But P2 and P3 respond to actual (not spoofed) conditions
        # This breaks causal relationships
        for i in range(attack_start, attack_end):
            # P2 drifts up (actual pressure increasing)
            drift = 0.002 * (i - attack_start)
            attack_df.loc[i, 'P2_B2016'] += drift
            attack_df.loc[i, 'P2_B2017'] += drift * 0.9
            
            # P3 temperature rises slowly
            attack_df.loc[i, 'P3_B2021'] += 0.001 * (i - attack_start)
        
        attack_df.loc[attack_start:attack_end, 'Label'] = 1
        
        return normal_df, attack_df
    
    def generate_slow_ramp_attack(self):
        """
        Attack: Linear ramp attack over 1000 samples (5% increase).
        Detection difficulty: MEDIUM-HIGH - very slow, could be mistaken for drift.
        """
        normal_df = self._generate_baseline_normal_data(4000)
        attack_df = normal_df.copy()
        
        # Attack: 1500-2500
        attack_start, attack_end = 1500, 2500
        
        # Linear ramp on multiple variables
        ramp_length = len(attack_df.loc[attack_start:attack_end, 'P1_B2005'])
        ramp = np.linspace(0, 0.05, ramp_length)
        
        attack_df.loc[attack_start:attack_end, 'P1_B2005'] *= (1 + ramp)
        attack_df.loc[attack_start:attack_end, 'P2_B2017'] *= (1 + ramp * 0.7)
        attack_df.loc[attack_start:attack_end, 'P3_B2022'] *= (1 + ramp * 0.5)
        
        attack_df.loc[attack_start:attack_end, 'Label'] = 1
        
        return normal_df, attack_df
    
    def generate_intermittent_pulse_attack(self):
        """
        Attack: Brief pulses every 100 samples (ON for 10, OFF for 90).
        Detection difficulty: HIGH - easy to miss individual pulses.
        """
        normal_df = self._generate_baseline_normal_data(3000)
        attack_df = normal_df.copy()
        
        # Multiple attack pulses
        attack_windows = [(800, 810), (900, 910), (1000, 1010), 
                         (1100, 1110), (1200, 1210), (1300, 1310)]
        
        for start, end in attack_windows:
            # Small spike in P4 control signal
            attack_df.loc[start:end, 'P4_HT_LD'] *= 1.15
            attack_df.loc[start:end, 'P4_HT_PG'] *= 0.90  # Compensatory response
            attack_df.loc[start:end, 'Label'] = 1
        
        return normal_df, attack_df
    
    def generate_model_informed_attack(self):
        """
        Attack: Adversary knows the causal model and manipulates multiple
        variables simultaneously to maintain causal consistency while
        driving system to unsafe state.
        
        Detection difficulty: EXTREME - maintains all causal relationships.
        """
        normal_df = self._generate_baseline_normal_data(3000)
        attack_df = normal_df.copy()
        
        # Attack: 1000-1400
        attack_start, attack_end = 1000, 1400
        
        # Coordinated manipulation maintaining P2 = 0.8 * P1 relationship
        scale_length = len(range(attack_start, attack_end))
        scale = np.linspace(1.0, 1.08, scale_length)
        
        for i, idx in enumerate(range(attack_start, attack_end)):
            # Scale P1 and P2 proportionally (maintains causal relationship)
            attack_df.loc[idx, 'P1_B2004'] *= scale[i]
            attack_df.loc[idx, 'P1_B2005'] *= scale[i]
            attack_df.loc[idx, 'P2_B2016'] *= scale[i]  # Maintains relationship
            attack_df.loc[idx, 'P2_B2017'] *= scale[i]
            
            # Control signals also adjusted to maintain consistency
            attack_df.loc[idx, 'P4_HT_LD'] *= scale[i] ** 0.5
        
        attack_df.loc[attack_start:attack_end, 'Label'] = 1
        
        return normal_df, attack_df

def main():
    generator = RealisticAttackGenerator()
    results = generator.generate_all_attacks()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, stats in results.items():
        print(f"\n{name}:")
        for key, val in stats.items():
            print(f"  {key}: {val}")
    
    print("\n" + "=" * 70)
    print("USAGE:")
    print("  python src/causal_detector.py --data realistic_attacks/stealthy_setpoint_change")
    print("=" * 70)

if __name__ == "__main__":
    main()

