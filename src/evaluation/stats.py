"""
Statistical testing utilities for experiment evaluation.

Provides:
- Welch's t-test for comparing methods
- Bootstrap confidence intervals
- Effect size (Cohen's d)
- Results formatting for publication
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import os


def _safe_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def welch_ttest(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Welch's t-test (unequal variance) comparing two methods.
    
    Args:
        a: Rewards/metrics from method A
        b: Rewards/metrics from method B
    
    Returns:
        Dictionary with t-statistic, p-value, and effect size
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("Welch t-test requires at least 2 samples per group")

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant_005': bool(p_value < 0.05),
        'significant_001': bool(p_value < 0.01),
    }


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.
    
    Returns:
        (mean, ci_lower, ci_upper)
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("bootstrap_ci requires at least one sample")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")

    rng = np.random.RandomState(seed)
    
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_means, 100 * alpha))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    
    return float(np.mean(data)), ci_lower, ci_upper


def summarize_results(
    method_name: str,
    rewards: np.ndarray,
    velocity_errors: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute summary statistics for a method's results.
    """
    rewards = np.asarray(rewards, dtype=float)
    if rewards.size == 0:
        raise ValueError("summarize_results requires non-empty rewards")
    mean, ci_lo, ci_hi = bootstrap_ci(rewards)
    
    summary = {
        'method': method_name,
        'n_episodes': len(rewards),
        'reward_mean': mean,
        'reward_std': _safe_std(rewards),
        'reward_median': float(np.median(rewards)),
        'reward_ci_95_lower': ci_lo,
        'reward_ci_95_upper': ci_hi,
        'reward_min': float(np.min(rewards)),
        'reward_max': float(np.max(rewards)),
        'rewards_raw': rewards.tolist(),
    }
    
    if velocity_errors is not None:
        velocity_errors = np.asarray(velocity_errors, dtype=float)
        if velocity_errors.size == 0:
            summary['velocity_error_mean'] = 0.0
            summary['velocity_error_std'] = 0.0
            return summary
        summary['velocity_error_mean'] = float(np.mean(velocity_errors))
        summary['velocity_error_std'] = _safe_std(velocity_errors)
    
    return summary


def compare_methods(
    results_a: Dict,
    results_b: Dict,
    label: str = "",
) -> Dict:
    """
    Statistical comparison between two methods.
    
    Args:
        results_a: Summary dict from summarize_results (should be "better" method)
        results_b: Summary dict from summarize_results (baseline)
    
    Returns:
        Comparison dict with test results
    """
    rewards_a = np.array(results_a['rewards_raw'])
    rewards_b = np.array(results_b['rewards_raw'])
    
    test = welch_ttest(rewards_a, rewards_b)
    
    improvement = (results_a['reward_mean'] - results_b['reward_mean'])
    baseline_mean = float(results_b['reward_mean'])
    pct_improvement = (
        improvement / abs(baseline_mean) * 100.0
        if not np.isclose(baseline_mean, 0.0)
        else float('nan')
    )
    
    return {
        'label': label,
        'method_a': results_a['method'],
        'method_b': results_b['method'],
        'mean_a': results_a['reward_mean'],
        'mean_b': results_b['reward_mean'],
        'improvement_absolute': float(improvement),
        'improvement_pct': float(pct_improvement),
        **test,
    }


def save_results(results: Dict, filepath: str):
    """Save results as JSON with pretty printing."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """Load results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)
