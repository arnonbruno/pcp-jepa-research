"""Tests for statistical utilities in src/evaluation/stats.py"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.stats import (
    welch_ttest,
    bootstrap_ci,
    summarize_results,
    compare_methods,
    save_results,
    load_results,
)


class TestWelchTtest:
    """Tests for welch_ttest function."""

    def test_welch_ttest_basic(self):
        """Test basic Welch's t-test returns correct structure."""
        np.random.seed(42)
        a = np.random.normal(100, 15, 50)
        b = np.random.normal(90, 20, 50)
        
        result = welch_ttest(a, b)
        
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'cohens_d' in result
        assert 'significant_005' in result
        assert 'significant_001' in result
        
        assert isinstance(result['t_statistic'], float)
        assert isinstance(result['p_value'], float)
        assert isinstance(result['cohens_d'], float)
        assert isinstance(result['significant_005'], bool)

    def test_welch_ttest_significance(self):
        """Test that clearly different distributions show significance."""
        np.random.seed(42)
        # Clear difference between groups
        a = np.random.normal(100, 5, 100)
        b = np.random.normal(50, 5, 100)
        
        result = welch_ttest(a, b)
        
        assert result['p_value'] < 0.001, "Should be highly significant"
        assert result['significant_005'] is True
        assert result['significant_001'] is True
        assert abs(result['cohens_d']) > 5, "Effect size should be large"

    def test_welch_ttest_same_distribution(self):
        """Test that identical distributions show no significance."""
        np.random.seed(42)
        data = np.random.normal(100, 15, 100)
        
        result = welch_ttest(data, data)
        
        # Same data should have p-value of 1.0 or NaN
        assert result['p_value'] > 0.05 or np.isnan(result['p_value'])
        assert result['cohens_d'] == 0.0

    def test_welch_ttest_unequal_variances(self):
        """Test Welch's t-test handles unequal variances correctly."""
        np.random.seed(42)
        # Same mean, very different variances
        a = np.random.normal(100, 5, 100)
        b = np.random.normal(100, 50, 100)
        
        result = welch_ttest(a, b)
        
        # Should not find significant difference in means
        assert result['p_value'] > 0.01


class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_bootstrap_ci_coverage(self):
        """Test that bootstrap CI contains the true mean."""
        np.random.seed(42)
        # Known distribution
        true_mean = 100.0
        data = np.random.normal(true_mean, 10, 1000)
        
        mean, ci_lower, ci_upper = bootstrap_ci(data, confidence=0.95, n_bootstrap=1000)
        
        # CI should contain the true mean
        assert ci_lower < true_mean < ci_upper
        # Sample mean should be close to true mean
        assert abs(mean - true_mean) < 1.0
        # CI should be reasonably narrow
        assert ci_upper - ci_lower < 3.0

    def test_bootstrap_ci_symmetry(self):
        """Test that CI is roughly symmetric around mean for symmetric data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        
        mean, ci_lower, ci_upper = bootstrap_ci(data, confidence=0.95, n_bootstrap=1000)
        
        # For symmetric distribution, CI should be roughly symmetric
        lower_dist = mean - ci_lower
        upper_dist = ci_upper - mean
        assert abs(lower_dist - upper_dist) < 0.1

    def test_bootstrap_ci_different_confidence(self):
        """Test that higher confidence gives wider interval."""
        np.random.seed(42)
        data = np.random.normal(100, 10, 500)
        
        _, lo_90, hi_90 = bootstrap_ci(data, confidence=0.90, n_bootstrap=1000)
        _, lo_99, hi_99 = bootstrap_ci(data, confidence=0.99, n_bootstrap=1000)
        
        # 99% CI should be wider than 90% CI
        assert (hi_99 - lo_99) > (hi_90 - lo_90)


class TestSummarizeResults:
    """Tests for summarize_results function."""

    def test_summarize_results(self):
        """Test result summarization."""
        np.random.seed(42)
        rewards = np.random.normal(100, 15, 50)
        
        result = summarize_results('TestMethod', rewards)
        
        assert result['method'] == 'TestMethod'
        assert result['n_episodes'] == 50
        assert abs(result['reward_mean'] - 100) < 5  # Close to expected mean
        assert 'reward_std' in result
        assert 'reward_median' in result
        assert 'reward_ci_95_lower' in result
        assert 'reward_ci_95_upper' in result
        assert result['reward_ci_95_lower'] < result['reward_mean'] < result['reward_ci_95_upper']

    def test_summarize_results_with_velocity(self):
        """Test result summarization with velocity errors."""
        np.random.seed(42)
        rewards = np.random.normal(100, 10, 30)
        vel_errors = np.random.normal(0.5, 0.1, 30)
        
        result = summarize_results('MethodWithVel', rewards, vel_errors)
        
        assert 'velocity_error_mean' in result
        assert 'velocity_error_std' in result
        assert abs(result['velocity_error_mean'] - 0.5) < 0.1

    def test_summarize_results_min_max(self):
        """Test that min/max are correctly computed."""
        rewards = np.array([10, 20, 30, 40, 50])
        
        result = summarize_results('Test', rewards)
        
        assert result['reward_min'] == 10
        assert result['reward_max'] == 50


class TestCompareMethods:
    """Tests for compare_methods function."""

    def test_compare_methods(self):
        """Test method comparison."""
        np.random.seed(42)
        rewards_a = np.random.normal(110, 10, 50)
        rewards_b = np.random.normal(100, 10, 50)
        
        result_a = summarize_results('MethodA', rewards_a)
        result_b = summarize_results('MethodB', rewards_b)
        
        comparison = compare_methods(result_a, result_b, "A vs B")
        
        assert comparison['label'] == "A vs B"
        assert comparison['method_a'] == 'MethodA'
        assert comparison['method_b'] == 'MethodB'
        assert comparison['improvement_absolute'] > 0  # A is better
        assert 'p_value' in comparison
        assert 'cohens_d' in comparison

    def test_compare_methods_negative_improvement(self):
        """Test comparison when method B is better."""
        np.random.seed(42)
        rewards_a = np.random.normal(90, 10, 50)
        rewards_b = np.random.normal(100, 10, 50)
        
        result_a = summarize_results('MethodA', rewards_a)
        result_b = summarize_results('MethodB', rewards_b)
        
        comparison = compare_methods(result_a, result_b)
        
        assert comparison['improvement_absolute'] < 0

    def test_compare_methods_pct_improvement(self):
        """Test percentage improvement calculation."""
        rewards_a = np.array([110] * 10)
        rewards_b = np.array([100] * 10)
        
        result_a = summarize_results('A', rewards_a)
        result_b = summarize_results('B', rewards_b)
        
        comparison = compare_methods(result_a, result_b)
        
        # 110 vs 100 = 10% improvement
        assert abs(comparison['improvement_pct'] - 10.0) < 0.1


class TestSaveLoad:
    """Tests for save/load functions."""

    def test_save_and_load_results(self, tmp_path):
        """Test that results can be saved and loaded."""
        np.random.seed(42)
        rewards = np.random.normal(100, 10, 20)
        result = summarize_results('Test', rewards)
        
        filepath = str(tmp_path / "test_results.json")
        save_results(result, filepath)
        
        loaded = load_results(filepath)
        
        assert loaded['method'] == result['method']
        assert loaded['n_episodes'] == result['n_episodes']
        assert abs(loaded['reward_mean'] - result['reward_mean']) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
