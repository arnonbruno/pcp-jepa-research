#!/usr/bin/env python3
"""
Generate paper figures for hybrid dynamics observer variance paper.

Figures:
1. Knife-edge sensitivity: success vs injected bias at x0=1.5
2. Seed histogram: baseline vs F3 vs FD
3. Per-init breakdown: success by initial position
"""

import numpy as np
import matplotlib.pyplot as plt
import json

# =============================================================================
# DATA FROM EXPERIMENTS
# =============================================================================

# Baseline results (10 seeds)
baseline_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
baseline_rates = [71.4, 57.1, 57.1, 57.1, 71.4, 57.1, 57.1, 71.4, 57.1, 57.1]

# F3 results (10 seeds)
f3_rates = [57.1, 71.4, 71.4, 71.4, 71.4, 71.4, 71.4, 71.4, 57.1, 57.1]

# FD baseline
fd_rate = 71.4

# =============================================================================
# FIGURE 1: KNIFE-EDGE SENSITIVITY
# =============================================================================

def figure_sensitivity():
    """Success vs injected bias at x0=1.5."""
    
    class Ball:
        def __init__(self):
            self.x, self.v = 0, 0
        def step(self, a):
            self.v += (-9.81 + np.clip(a, -2, 2)) * 0.05
            self.x += self.v * 0.05
            if self.x < 0:
                self.x = -self.x * 0.8
                self.v = -self.v * 0.8
            if self.x > 3:
                self.x = 3 - (self.x - 3) * 0.8
                self.v = -self.v * 0.8
    
    def eval_with_bias(x0, bias, n_trials=100):
        s = 0
        for _ in range(n_trials):
            b = Ball()
            b.x, b.v, xp = x0, 0.0, x0
            for step in range(30):
                v_fd = (b.x - xp) / 0.05
                # Inject bias at startup
                if step == 0:
                    v_est = v_fd + bias
                else:
                    v_est = v_fd
                a = np.clip(1.5 * (2 - b.x) + (-2) * (-v_est), -2, 2)
                xp = b.x
                b.step(a)
                if abs(b.x - 2.0) < 0.3:
                    s += 1
                    break
        return s / n_trials
    
    biases = np.linspace(-0.5, 0.5, 21)
    rates_15 = [eval_with_bias(1.5, b) for b in biases]
    rates_20 = [eval_with_bias(2.0, b) for b in biases]
    rates_25 = [eval_with_bias(2.5, b) for b in biases]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(biases, rates_15, 'b-', linewidth=2, label='x₀=1.5 (boundary)')
    ax.plot(biases, rates_20, 'g--', linewidth=2, label='x₀=2.0')
    ax.plot(biases, rates_25, 'r:', linewidth=2, label='x₀=2.5')
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, label='FD (bias=0)')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between([-0.1, 0.1], 0, 1.1, alpha=0.2, color='green', label='Safe region')
    ax.set_xlabel('Injected Velocity Bias at Startup', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Knife-Edge Sensitivity at Boundary (x₀=1.5)', fontsize=14)
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('figure1_sensitivity.png', dpi=150)
    plt.savefig('figure1_sensitivity.pdf')
    print('Saved: figure1_sensitivity.png')
    plt.close()

# =============================================================================
# FIGURE 2: SEED HISTOGRAM
# =============================================================================

def figure_histogram():
    """Success rate distribution across seeds."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(10)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, f3_rates, width, label='F3 (physics-normalized)', color='darkorange', alpha=0.8)
    
    ax.axhline(fd_rate, color='green', linestyle='--', linewidth=2, label='FD (zero-variance)')
    ax.axhline(np.mean(baseline_rates), color='steelblue', linestyle=':', alpha=0.7)
    ax.axhline(np.mean(f3_rates), color='darkorange', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Training Seed', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Training Instability: Success Rate Distribution Across Seeds', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 80)
    
    # Add mean/std annotations
    ax.annotate(f'Baseline: {np.mean(baseline_rates):.1f}% ± {np.std(baseline_rates):.1f}%',
                xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10, color='steelblue')
    ax.annotate(f'F3: {np.mean(f3_rates):.1f}% ± {np.std(f3_rates):.1f}%',
                xy=(0.02, 0.88), xycoords='axes fraction', fontsize=10, color='darkorange')
    ax.annotate(f'Seeds @ 71.4%: Baseline={sum(1 for r in baseline_rates if r > 65)}/10, F3={sum(1 for r in f3_rates if r > 65)}/10',
                xy=(0.02, 0.81), xycoords='axes fraction', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figure2_histogram.png', dpi=150)
    plt.savefig('figure2_histogram.pdf')
    print('Saved: figure2_histogram.png')
    plt.close()

# =============================================================================
# FIGURE 3: PER-INIT BREAKDOWN
# =============================================================================

def figure_per_init():
    """Success rate by initial position."""
    
    # Per-init results from experiments
    inits = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    # Good seeds (0, 4, 7) vs bad seeds
    good_init = [0, 0, 100, 100, 100, 100, 100]
    bad_init = [0, 0, 0, 100, 100, 100, 100]
    fd_init = [0, 0, 100, 100, 100, 100, 100]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(inits))
    width = 0.25
    
    bars1 = ax.bar(x - width, good_init, width, label='Good seeds (0,4,7)', color='green', alpha=0.8)
    bars2 = ax.bar(x, bad_init, width, label='Bad seeds (1,2,3,5,6,8,9)', color='red', alpha=0.8)
    bars3 = ax.bar(x + width, fd_init, width, label='FD baseline', color='blue', alpha=0.8)
    
    ax.axvline(1.5, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('Boundary\nCase', xy=(2.5, 80), fontsize=10, ha='center')
    
    # Failure region
    ax.fill_between([-0.5, 1.5], 0, 110, alpha=0.1, color='red', label='Failure region')
    
    ax.set_xlabel('Initial Position x₀', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Per-Init Success Rate: All Variance at Boundary (x₀=1.5)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i:.1f}' for i in inits])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig('figure3_per_init.png', dpi=150)
    plt.savefig('figure3_per_init.pdf')
    print('Saved: figure3_per_init.png')
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('Generating paper figures...')
    print()
    
    figure_sensitivity()
    figure_histogram()
    figure_per_init()
    
    print()
    print('All figures saved!')