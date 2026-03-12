#!/usr/bin/env python3
"""
NeurIPS Final Figures Generator
Generates all 4 requested figures from JSON results.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import json
import os
import glob
import sys
import argparse

sns.set_palette("colorblind")
colors = sns.color_palette("colorblind")

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 300,
})

def load_all_seeds(pattern):
    files = glob.glob(pattern)
    data = []
    for f in files:
        try:
            with open(f, 'r') as file:
                data.append(json.load(file))
        except Exception as e:
            pass
    return data

def aggregate_metric(data_list, path_to_metric):
    """Extract a metric from a list of result dicts, returning mean and std across seeds."""
    vals = []
    for d in data_list:
        try:
            curr = d
            for key in path_to_metric:
                curr = curr[key]
            vals.append(curr)
        except KeyError:
            continue
    if not vals:
        return 0.0, 0.0
    return np.mean(vals), np.std(vals)

def create_figure1(results_dir, output_dir):
    """Figure 1: Performance comparison across all environments"""
    envs = ['Hopper-v4', 'Walker2d-v4', 'Ant-v4']
    methods = [
        ('oracle', 'methods', 'oracle', 'reward_mean'),
        ('frozen', 'methods', 'frozen_baseline', 'reward_mean'),
        ('rssm', 'methods', 'rssm', 'reward_mean'),
        ('told', 'methods', 'told', 'reward_mean'),
        ('sma', 'sma', 'reward_mean'),
        ('lstm', 'lstm', 'reward_mean'),
        ('pano', 'methods', 'pano', 'reward_mean'),
    ]
    
    labels = ['Oracle', 'Frozen', 'RSSM', 'TOLD', 'SMA', 'LSTM', 'PANO']
    
    # Collect data: method -> env -> (mean, std)
    plot_data = {m: [] for m in labels}
    plot_errs = {m: [] for m in labels}
    
    for env in envs:
        # Load files
        pano_files = load_all_seeds(os.path.join(results_dir, f'pano_{env}_seed*.json'))
        sota_files = load_all_seeds(os.path.join(results_dir, f'sota_baselines_{env}_seed*.json'))
        base_files = load_all_seeds(os.path.join(results_dir, f'new_baselines_{env}_seed*.json'))
        
        # We need a unified dict per seed ideally, but we can just average across available seeds per method
        
        for m in methods:
            m_id = m[0]
            if m_id in ['oracle', 'frozen', 'pano']:
                source = pano_files
                path = list(m[1:])
                # Fix path for frozen
                if m_id == 'frozen':
                    path = ['methods', 'frozen_baseline', 'reward_mean']
            elif m_id in ['rssm', 'told']:
                source = sota_files
                path = list(m[1:])
            elif m_id in ['sma', 'lstm']:
                source = base_files
                path = list(m[1:])
                
            val_mean, val_std = aggregate_metric(source, path)
            # Normalization against oracle
            plot_data[labels[[m[0] for m in methods].index(m_id)]].append(val_mean)
            plot_errs[labels[[m[0] for m in methods].index(m_id)]].append(val_std)
            
    # Plotting
    x = np.arange(len(envs))
    width = 0.12
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, label in enumerate(labels):
        offset = (i - len(labels)/2) * width + width/2
        means = plot_data[label]
        errs = plot_errs[label]
        # Calculate percentage of Oracle mean
        oracle_means = plot_data['Oracle']
        pct_means = [m / max(o, 1) * 100 for m, o in zip(means, oracle_means)]
        pct_errs = [e / max(o, 1) * 100 for e, o in zip(errs, oracle_means)]
        
        ax.bar(x + offset, pct_means, width, label=label, edgecolor='black', alpha=0.9)
        # ax.errorbar(x + offset, pct_means, yerr=pct_errs, fmt='none', ecolor='black', capsize=2)
        
    ax.set_ylabel('Performance (% of Oracle)')
    ax.set_title('Performance Across Environments with Sensor Dropout')
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace('-v4', '') for e in envs])
    ax.axhline(100, color='black', linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure1_multi_env.pdf'))
    plt.close()

def create_figure2(results_dir, output_dir):
    """Figure 2: PANO vs Oracle ablation"""
    files = load_all_seeds(os.path.join(results_dir, 'ablation_oracle_Hopper-v4_seed*.json'))
    
    methods = ['oracle', 'sma_3', 'sma_5', 'ema_0.5', 'ema_0.2']
    labels = ['Standard Oracle', 'Oracle + SMA(3)', 'Oracle + SMA(5)', 'Oracle + EMA(0.5)', 'Oracle + EMA(0.2)']
    
    means = []
    errs = []
    
    for m in methods:
        val_mean, val_std = aggregate_metric(files, [m, 'reward_mean'])
        means.append(val_mean)
        errs.append(val_std)
        
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=errs, capsize=5, color=colors[:len(labels)], edgecolor='black')
    
    ax.set_ylabel('Episodic Reward')
    ax.set_title('Oracle Smoothing Ablation (Why PANO Beats Oracle)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, m in enumerate(means):
        if m > 0:
            ax.annotate(f'{m:.0f}', xy=(i, m + errs[i] + 50), ha='center')
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure2_oracle_ablation.pdf'))
    plt.close()

def create_figure3(results_dir, output_dir):
    """Figure 3: Walker2d failure analysis"""
    files = load_all_seeds(os.path.join(results_dir, 'walker_diagnostics_seed*.json'))
    pano_files = load_all_seeds(os.path.join(results_dir, 'pano_Walker2d-v4_seed*.json'))
    
    # We also need Oracle and Frozen for Walker2d to show context
    oracle_mean, oracle_std = aggregate_metric(pano_files, ['methods', 'oracle', 'reward_mean'])
    frozen_mean, frozen_std = aggregate_metric(pano_files, ['methods', 'frozen_baseline', 'reward_mean'])
    
    methods = ['standard', 'large_net', 'large_data']
    labels = ['Standard PANO', 'Large Network', 'More Data (1k eps)']
    
    means = [frozen_mean]
    errs = [frozen_std]
    all_labels = ['Frozen Baseline']
    
    for m, l in zip(methods, labels):
        val_mean, val_std = aggregate_metric(files, [m, 'reward_mean'])
        means.append(val_mean)
        errs.append(val_std)
        all_labels.append(l)
        
    means.append(oracle_mean)
    errs.append(oracle_std)
    all_labels.append('Oracle')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(all_labels))
    bar_colors = [colors[3]] + [colors[0]]*3 + [colors[2]]
    
    bars = ax.bar(x, means, yerr=errs, capsize=5, color=bar_colors, edgecolor='black')
    
    ax.set_ylabel('Episodic Reward')
    ax.set_title('Walker2d Diagnostic Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure3_walker2d_diagnostics.pdf'))
    plt.close()

def create_figure4(results_dir, output_dir):
    """Figure 4: Latent Drift Visualization (copying from old script basically)"""
    # Assuming bulletproof_results.json exists
    bullet_path = os.path.join(results_dir, 'bulletproof_results.json')
    if not os.path.exists(bullet_path):
        print("No bulletproof_results.json, skipping Figure 4")
        return
        
    with open(bullet_path, 'r') as f:
        data = json.load(f)
        
    profiling = data.get('impact_profiling', [])
    if not profiling: return
    
    steps = [r['step'] for r in profiling]
    overall_errors = [r['overall_error'] for r in profiling]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, overall_errors, 'o-', color=colors[0], linewidth=2, label='Overall Divergence')
    
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Latent Prediction Error (Log Scale)')
    ax.set_title('Latent Drift Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure4_latent_drift.pdf'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results/neurips')
    parser.add_argument('--output-dir', type=str, default='results/neurips/figures')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Generating figures...")
    create_figure1(args.results_dir, args.output_dir)
    create_figure2(args.results_dir, args.output_dir)
    create_figure3(args.results_dir, args.output_dir)
    create_figure4(args.results_dir, args.output_dir)
    print(f"Figures saved to {args.output_dir}")
