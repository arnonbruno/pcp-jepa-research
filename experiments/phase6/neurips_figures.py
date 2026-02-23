#!/usr/bin/env python3
"""
NeurIPS Figures Generator — Data-Driven

Generates publication-quality figures by reading structured JSON
results from the experiment scripts. NO hardcoded numbers.

Usage:
    python neurips_figures.py --results-dir ../../results/phase6

Reads:
    - hopper_pano_results.json  (from hopper_pano.py)
    - bulletproof_results.json  (from bulletproof_negative.py)

Outputs:
    - figure1_latent_drift.pdf
    - figure2_data_scaling.pdf
    - figure3_performance_recovery.pdf
    - figure4_multi_env.pdf
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import json
import os
import sys
import argparse

# Colorblind-friendly palette
sns.set_palette("colorblind")
colors = sns.color_palette("colorblind")

# LaTeX-style formatting for NeurIPS two-column
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (7, 4),
    'figure.dpi': 300,
})


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# =============================================================================
# FIGURE 1: EXPONENTIAL LATENT DRIFT
# =============================================================================

def create_figure1(bulletproof_data, output_dir='.'):
    """Exponential divergence of Standard Latent JEPA from impact profiling data."""
    profiling = bulletproof_data['impact_profiling']

    steps = [r['step'] for r in profiling]
    overall_errors = [r['overall_error'] for r in profiling]
    air_errors = [r['air_error'] for r in profiling]
    impact_errors = [r['impact_error'] for r in profiling]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.semilogy(steps, overall_errors, 'o-', color=colors[0], linewidth=2,
                markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                label='Overall', zorder=3)
    ax.semilogy(steps, air_errors, 's--', color=colors[2], linewidth=1.5,
                markersize=6, alpha=0.8, label='Air phase')
    if any(e > 0 for e in impact_errors):
        ax.semilogy(steps, [max(e, 1e-3) for e in impact_errors], '^--',
                    color=colors[3], linewidth=1.5, markersize=6, alpha=0.8,
                    label='Contact phase')

    # Exponential fit
    if overall_errors[0] > 0 and overall_errors[-1] > 0:
        log_ratio = np.log(overall_errors[-1] / overall_errors[0]) / (len(steps) - 1)
        x_fit = np.linspace(steps[0], steps[-1], 100)
        y_fit = overall_errors[0] * np.exp(log_ratio * (x_fit - steps[0]))
        growth_per_step = np.exp(log_ratio)
        ax.semilogy(x_fit, y_fit, '-', color=colors[1], alpha=0.5, linewidth=1.5,
                    label=f'Exp fit (~{growth_per_step:.1f}× per step)')

    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Latent Prediction Error (Log Scale)')
    ax.set_title('Exponential Divergence of Standard Latent JEPA Rollout')
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(output_dir, f'figure1_latent_drift.{fmt}'),
                    format=fmt, bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 1 saved: figure1_latent_drift.pdf")

# =============================================================================
# FIGURE 2: DATA SCALING ASYMPTOTE
# =============================================================================

def create_figure2(bulletproof_data, output_dir='.'):
    """Data Scaling Asymptote — prediction loss dominates velocity loss."""
    scaling = bulletproof_data['data_scaling']

    labels = [f"{r['n_transitions']//1000}k" for r in scaling]
    vel_losses = [r['velocity_loss'] for r in scaling]
    pred_losses = [r['prediction_loss'] for r in scaling]
    ratios = [r['ratio'] for r in scaling]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, vel_losses, width, label='Velocity Loss (λ=10.0)',
                   color=colors[0], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, pred_losses, width, label='Prediction Loss (λ=0.1)',
                   color=colors[1], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Training Dataset Size')
    ax.set_ylabel('Loss Value (Log Scale)')
    ax.set_title('Data Scaling Does Not Fix Latent–Physics Misalignment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    for i, ratio in enumerate(ratios):
        ax.annotate(f'{ratio:.0f}×', xy=(i + width/2, pred_losses[i] * 1.5),
                    ha='center', fontsize=9, fontweight='bold', color=colors[3])

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(output_dir, f'figure2_data_scaling.{fmt}'),
                    format=fmt, bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 2 saved: figure2_data_scaling.pdf")

# =============================================================================
# FIGURE 3: PERFORMANCE RECOVERY (PANO vs Baselines)
# =============================================================================

def create_figure3(pano_data, output_dir='.'):
    """PANO performance recovery under sensor dropout with statistical tests."""
    methods = pano_data['methods']
    comparisons = pano_data['comparisons']

    method_names = []
    means = []
    stds = []
    ci_lows = []
    ci_highs = []
    method_colors = []

    for key, color in [('oracle', colors[2]), ('frozen_baseline', colors[3]),
                        ('ekf', colors[4]), ('pano', colors[0])]:
        r = methods[key]
        method_names.append(r['method'].replace(' (dropout)', '\n(dropout)').replace(' (no dropout)', '\n(no dropout)'))
        means.append(r['reward_mean'])
        stds.append(r['reward_std'])
        ci_lows.append(r['reward_ci_95_lower'])
        ci_highs.append(r['reward_ci_95_upper'])
        method_colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(method_names))

    # Use 95% CI for error bars
    ci_errors = [[m - lo for m, lo in zip(means, ci_lows)],
                 [hi - m for m, hi in zip(means, ci_highs)]]

    bars = ax.bar(x, means, yerr=ci_errors, capsize=5,
                  color=method_colors, edgecolor='black', linewidth=0.5, alpha=0.9)

    ax.set_xlabel('Method')
    ax.set_ylabel('Episodic Reward')
    ax.set_title('PANO Recovers Performance Under Contact-Triggered Sensor Dropout')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add reward values on bars
    for i, (m, hi) in enumerate(zip(means, ci_highs)):
        ax.annotate(f'{m:.0f}', xy=(i, hi + 5), ha='center', fontsize=10, fontweight='bold')

    # Add significance annotations
    pano_vs_frozen = comparisons.get('pano_vs_frozen', {})
    if pano_vs_frozen:
        p = pano_vs_frozen.get('p_value', 1.0)
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        improv = pano_vs_frozen.get('improvement_pct', 0)

        # Draw bracket between frozen (idx=1) and pano (idx=3)
        y_bracket = max(means) * 1.15
        ax.plot([1, 1, 3, 3], [y_bracket - 5, y_bracket, y_bracket, y_bracket - 5],
                color='black', linewidth=1)
        ax.text(2, y_bracket + 3, f'{stars} +{improv:.1f}% (p={p:.3f})',
                ha='center', fontsize=9, fontweight='bold')

    ax.set_ylim(0, max(means) * 1.35)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(output_dir, f'figure3_performance_recovery.{fmt}'),
                    format=fmt, bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 3 saved: figure3_performance_recovery.pdf")

# =============================================================================
# FIGURE 4: MULTI-ENVIRONMENT ABLATION
# =============================================================================

def create_figure4(bulletproof_data, output_dir='.'):
    """Multi-environment comparison: Standard Latent JEPA vs baselines."""
    ablation = bulletproof_data['multi_env_ablation']

    env_names = [r['env_id'].replace('-v4', '') for r in ablation]
    oracle_rewards = [r['oracle_reward_mean'] for r in ablation]
    baseline_rewards = [r['baseline_reward_mean'] for r in ablation]
    jepa_rewards = [r['jepa_reward_mean'] for r in ablation]

    # Normalize to % of oracle for comparability
    oracle_pct = [100.0] * len(env_names)
    baseline_pct = [b / o * 100 for b, o in zip(baseline_rewards, oracle_rewards)]
    jepa_pct = [j / o * 100 for j, o in zip(jepa_rewards, oracle_rewards)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(env_names))
    width = 0.25

    ax.bar(x - width, oracle_pct, width, label='Oracle (no dropout)',
           color=colors[2], edgecolor='black', linewidth=0.5)
    ax.bar(x, baseline_pct, width, label='Frozen Baseline (dropout)',
           color=colors[3], edgecolor='black', linewidth=0.5)
    ax.bar(x + width, jepa_pct, width, label='Standard Latent JEPA (dropout)',
           color=colors[1], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Environment')
    ax.set_ylabel('Performance (% of Oracle)')
    ax.set_title('Standard Latent JEPA Fails Across Environments')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=15, ha='right')
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add p-values
    for i, r in enumerate(ablation):
        p = r.get('jepa_vs_baseline_p_value', 1.0)
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        if stars:
            ax.text(i + width, jepa_pct[i] + 2, stars, ha='center', fontsize=10,
                    fontweight='bold', color='red')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(output_dir, f'figure4_multi_env.{fmt}'),
                    format=fmt, bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 4 saved: figure4_multi_env.pdf")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate NeurIPS figures from JSON results')
    parser.add_argument('--results-dir', type=str, default='../../results/phase6',
                        help='Directory containing results JSON files')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save figures')
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING NEURIPS-READY FIGURES (DATA-DRIVEN)")
    print(f"Reading results from: {args.results_dir}")
    print("=" * 70)

    # Load results
    pano_path = os.path.join(args.results_dir, 'hopper_pano_results.json')
    bullet_path = os.path.join(args.results_dir, 'bulletproof_results.json')

    if not os.path.exists(pano_path):
        print(f"ERROR: {pano_path} not found. Run hopper_pano.py first.")
        sys.exit(1)
    if not os.path.exists(bullet_path):
        print(f"ERROR: {bullet_path} not found. Run bulletproof_negative.py first.")
        sys.exit(1)

    pano_data = load_json(pano_path)
    bulletproof_data = load_json(bullet_path)

    os.makedirs(args.output_dir, exist_ok=True)

    create_figure1(bulletproof_data, args.output_dir)
    create_figure2(bulletproof_data, args.output_dir)
    create_figure3(pano_data, args.output_dir)
    create_figure4(bulletproof_data, args.output_dir)

    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED FROM EXPERIMENT DATA")
    print("=" * 70)
    print(f"\nFigures saved to: {args.output_dir}/")
    print("  figure1_latent_drift.pdf  — Exponential divergence")
    print("  figure2_data_scaling.pdf  — Data doesn't fix misalignment")
    print("  figure3_performance_recovery.pdf — PANO vs baselines with stats")
    print("  figure4_multi_env.pdf     — Multi-environment ablation")
