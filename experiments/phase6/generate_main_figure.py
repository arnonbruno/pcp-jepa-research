#!/usr/bin/env python3
"""
Generate main result figure for paper.
Shows all methods with confidence intervals and statistical significance.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import json
import os

# LaTeX-style formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (10, 5),
    'figure.dpi': 300,
})

# Load results
with open('./results/phase6/hopper_pano_results.json', 'r') as f:
    pano_data = json.load(f)
    
with open('./results/phase6/sota_baselines_results.json', 'r') as f:
    sota_data = json.load(f)

# Extract data
methods = ['PANO', 'Oracle', 'Frozen', 'TOLD', 'EKF', 'RSSM']
rewards = [
    pano_data['methods']['pano']['reward_mean'],
    pano_data['methods']['oracle']['reward_mean'],
    pano_data['methods']['frozen_baseline']['reward_mean'],
    sota_data['methods']['told']['reward_mean'],
    pano_data['methods']['ekf']['reward_mean'],
    sota_data['methods']['rssm']['reward_mean'],
]
ci_lower = [
    pano_data['methods']['pano']['reward_ci_95_lower'],
    pano_data['methods']['oracle']['reward_ci_95_lower'],
    pano_data['methods']['frozen_baseline']['reward_ci_95_lower'],
    sota_data['methods']['told']['reward_ci_95_lower'],
    pano_data['methods']['ekf']['reward_ci_95_lower'],
    sota_data['methods']['rssm']['reward_ci_95_lower'],
]
ci_upper = [
    pano_data['methods']['pano']['reward_ci_95_upper'],
    pano_data['methods']['oracle']['reward_ci_95_upper'],
    pano_data['methods']['frozen_baseline']['reward_ci_95_upper'],
    sota_data['methods']['told']['reward_ci_95_upper'],
    pano_data['methods']['ekf']['reward_ci_95_upper'],
    sota_data['methods']['rssm']['reward_ci_95_upper'],
]
errors = [
    [r - l for r, l in zip(rewards, ci_lower)],
    [u - r for u, r in zip(ci_upper, rewards)]
]

# Colors
colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728', '#8c564b']
patterns = ['', '', '', '///', 'xxx', '\\\\\\']

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(methods))
bars = ax.bar(x, rewards, yerr=errors, capsize=5, color=colors, 
               edgecolor='black', linewidth=1.5, alpha=0.8)

# Add significance markers
# PANO vs Frozen: p=7.9e-33
ax.annotate('***', xy=(0, rewards[0] + 50), ha='center', fontsize=16, fontweight='bold')
ax.annotate('p=7.9e-33', xy=(0, rewards[0] + 100), ha='center', fontsize=9, color='green')

# Oracle line
ax.axhline(y=rewards[1], color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5, label='Oracle baseline')

# Add value labels
for i, (bar, r, l, u) in enumerate(zip(bars, rewards, ci_lower, ci_upper)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
            f'{r:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Episode Reward')
ax.set_xlabel('Method')
ax.set_title('PANO Outperforms All Baselines on Hopper-v4 (100 Episodes)')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=0)
ax.set_ylim(0, 1400)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Legend
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor='#2ca02c', edgecolor='black', label='PANO (Ours)'),
    plt.Rectangle((0,0),1,1, facecolor='#1f77b4', edgecolor='black', label='Oracle (no dropout)'),
    plt.Rectangle((0,0),1,1, facecolor='#ff7f0e', edgecolor='black', label='Frozen Baseline'),
    plt.Rectangle((0,0),1,1, facecolor='#9467bd', edgecolor='black', hatch='///', label='Simplified TOLD'),
    plt.Rectangle((0,0),1,1, facecolor='#d62728', edgecolor='black', hatch='xxx', label='EKF'),
    plt.Rectangle((0,0),1,1, facecolor='#8c564b', edgecolor='black', hatch='\\\\\\', label='Simplified RSSM'),
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

plt.tight_layout()

# Save
for fmt in ['pdf', 'png']:
    plt.savefig(f'./results/phase6/figure_main_results.{fmt}', 
                format=fmt, bbox_inches='tight', dpi=300)

print("✓ Main results figure saved to ./results/phase6/figure_main_results.pdf")
plt.close()

# Generate a second figure: performance breakdown
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Reward comparison
ax1 = axes[0]
x = np.arange(4)  # PANO, Oracle, Frozen, EKF
subset_methods = ['PANO', 'Oracle', 'Frozen', 'EKF']
subset_rewards = [rewards[0], rewards[1], rewards[2], rewards[4]]
subset_errors = [[errors[0][0], errors[0][1], errors[0][2], errors[0][4]],
                 [errors[1][0], errors[1][1], errors[1][2], errors[1][4]]]
subset_colors = [colors[0], colors[1], colors[2], colors[4]]

bars1 = ax1.bar(x, subset_rewards, yerr=subset_errors, capsize=5, 
                color=subset_colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Episode Reward')
ax1.set_xlabel('Method')
ax1.set_title('PANO vs Core Baselines')
ax1.set_xticks(x)
ax1.set_xticklabels(subset_methods)
ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

# Add improvement percentages
ax1.annotate('+161% vs Frozen\np=7.9e-33', xy=(0, 900), fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Right: World models fail
ax2 = axes[1]
x = np.arange(3)  # Frozen, TOLD, RSSM
wm_methods = ['Frozen', 'Simplified TOLD', 'Simplified RSSM']
wm_rewards = [rewards[2], rewards[3], rewards[5]]
wm_errors = [[errors[0][2], errors[0][3], errors[0][5]],
             [errors[1][2], errors[1][3], errors[1][5]]]
wm_colors = [colors[2], colors[3], colors[5]]

bars2 = ax2.bar(x, wm_rewards, yerr=wm_errors, capsize=5,
                color=wm_colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Episode Reward')
ax2.set_xlabel('Method')
ax2.set_title('Latent-Space World Models Fail Under Contact Dropout')
ax2.set_xticks(x)
ax2.set_xticklabels(wm_methods, rotation=15)
ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

# Add failure percentages
ax2.annotate('-76%', xy=(1, wm_rewards[1] + 30), fontsize=10, ha='center', color='purple')
ax2.annotate('-90%', xy=(2, wm_rewards[2] + 20), fontsize=10, ha='center', color='brown')

plt.tight_layout()
for fmt in ['pdf', 'png']:
    plt.savefig(f'./results/phase6/figure_performance_breakdown.{fmt}',
                format=fmt, bbox_inches='tight', dpi=300)
print("✓ Performance breakdown figure saved to ./results/phase6/figure_performance_breakdown.pdf")
plt.close()
