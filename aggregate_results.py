#!/usr/bin/env python3
"""
Aggregates results across multiple seeds and performs proper statistical tests.
"""

import os
import json
import numpy as np
import scipy.stats as stats
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "neurips"
ENVS = ['Hopper-v4', 'Walker2d-v4']
SEEDS = [42, 123, 456, 789, 1024]

def calculate_stats(data_array):
    if len(data_array) == 0:
        return {'mean': 0.0, 'std': 0.0, 'ci95_lower': 0.0, 'ci95_upper': 0.0}
    mean = np.mean(data_array)
    std = np.std(data_array, ddof=1) if len(data_array) > 1 else 0.0
    # 95% CI using t-distribution
    ci = stats.t.ppf(0.975, df=max(1, len(data_array)-1)) * (std / np.sqrt(len(data_array))) if len(data_array) > 1 else 0.0
    return {
        'mean': float(mean),
        'std': float(std),
        'ci95_lower': float(mean - ci),
        'ci95_upper': float(mean + ci)
    }

def perform_welch_ttest(data1, data2):
    if len(data1) < 2 or len(data2) < 2:
        return {'p_value': 1.0, 't_stat': 0.0}
    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
    return {'p_value': float(p_val), 't_stat': float(t_stat)}

def main():
    print("="*70)
    print("AGGREGATED RESULTS (ACROSS SEEDS)")
    print("="*70)
    
    summary = {}
    
    for env_id in ENVS:
        print(f"\nEnvironment: {env_id}")
        
        methods_data = {
            'Oracle': [],
            'Frozen': [],
            'EKF': [],
            'RSSM': [],
            'TOLD': [],
            'PANO': []
        }
        
        for seed in SEEDS:
            # Load SOTA
            sota_file = RESULTS_DIR / f'sota_baselines_{env_id}_seed{seed}.json'
            if sota_file.exists():
                with open(sota_file, 'r') as f:
                    sota_data = json.load(f)
                    methods = sota_data.get('methods', {})
                    if 'oracle' in methods: methods_data['Oracle'].append(methods['oracle']['reward_mean'])
                    if 'frozen' in methods: methods_data['Frozen'].append(methods['frozen']['reward_mean'])
                    if 'rssm' in methods: methods_data['RSSM'].append(methods['rssm']['reward_mean'])
                    if 'told' in methods: methods_data['TOLD'].append(methods['told']['reward_mean'])
            
            # Load PANO
            pano_file = RESULTS_DIR / f'pano_{env_id}_seed{seed}.json'
            if pano_file.exists():
                with open(pano_file, 'r') as f:
                    pano_data = json.load(f)
                    methods = pano_data.get('methods', {})
                    # oracle and frozen are here too, but we might have loaded from sota.
                    # if we didn't get from sota, try getting from pano.
                    if 'oracle' in methods and len(methods_data['Oracle']) < SEEDS.index(seed) + 1:
                        methods_data['Oracle'].append(methods['oracle']['reward_mean'])
                    if 'frozen_baseline' in methods and len(methods_data['Frozen']) < SEEDS.index(seed) + 1:
                        methods_data['Frozen'].append(methods['frozen_baseline']['reward_mean'])
                    
                    if 'ekf' in methods: methods_data['EKF'].append(methods['ekf']['reward_mean'])
                    if 'pano' in methods: methods_data['PANO'].append(methods['pano']['reward_mean'])

        # Calculate stats
        print(f"{'Method':<10} | {'Mean Reward':>12} ± {'95% CI':>6} | {'Seeds':>5}")
        print("-" * 50)
        
        env_summary = {}
        for method, values in methods_data.items():
            st = calculate_stats(values)
            env_summary[method] = {
                'raw_values': values,
                'stats': st
            }
            print(f"{method:<10} | {st['mean']:>12.1f} ± {st['ci95_upper']-st['mean']:>6.1f} | {len(values):>5}")
            
        # Statistical comparisons
        print("\nStatistical Tests (Welch's t-test across seeds):")
        comparisons = [
            ('PANO', 'Oracle'),
            ('PANO', 'EKF'),
            ('PANO', 'Frozen'),
            ('RSSM', 'Frozen'),
            ('TOLD', 'Frozen')
        ]
        
        env_tests = {}
        for m1, m2 in comparisons:
            if len(methods_data[m1]) > 0 and len(methods_data[m2]) > 0:
                test = perform_welch_ttest(methods_data[m1], methods_data[m2])
                env_tests[f"{m1}_vs_{m2}"] = test
                sig = "*" if test['p_value'] < 0.05 else " "
                print(f"  {m1} vs {m2:<8}: p={test['p_value']:.4f} {sig}")
                
        summary[env_id] = {
            'methods': env_summary,
            'tests': env_tests
        }
        
    with open(RESULTS_DIR / 'aggregated_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
