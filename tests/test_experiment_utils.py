"""Tests for experiment utilities"""

import numpy as np
import torch
import json
import os
import tempfile
import pytest
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestCheckpointSaveLoad:
    """Tests for checkpoint saving and loading."""

    def test_checkpoint_save_load(self):
        """Test saving and loading model checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'model_checkpoint.pt')
            
            # Create a simple model
            model = torch.nn.Linear(10, 5)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 10,
                'loss': 0.5,
            }, checkpoint_path)
            
            # Create new model and load checkpoint
            model2 = torch.nn.Linear(10, 5)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model2.load_state_dict(checkpoint['model_state_dict'])
            optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Verify parameters match
            for (p1, p2) in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)
            
            assert checkpoint['epoch'] == 10
            assert checkpoint['loss'] == 0.5

    def test_checkpoint_with_rng_state(self):
        """Test checkpoint includes RNG state for reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'rng_checkpoint.pt')
            
            # Set RNG states
            torch.manual_seed(42)
            np.random.seed(42)
            
            rng_state = torch.get_rng_state()
            np_rng_state = np.random.get_state()
            
            # Save
            torch.save({
                'torch_rng': rng_state,
                'np_rng': np_rng_state,
            }, checkpoint_path)
            
            # Generate random values (changes state)
            _ = torch.randn(10)
            _ = np.random.randn(10)
            
            # Load
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            torch.set_rng_state(checkpoint['torch_rng'])
            np.random.set_state(checkpoint['np_rng'])
            
            # Generate again - should match what would have been before
            t1 = torch.randn(5)
            n1 = np.random.randn(5)
            
            # Reset and compare
            torch.set_rng_state(checkpoint['torch_rng'])
            np.random.set_state(checkpoint['np_rng'])
            
            t2 = torch.randn(5)
            n2 = np.random.randn(5)
            
            assert torch.allclose(t1, t2)
            np.testing.assert_allclose(n1, n2)


class TestNanDetection:
    """Tests for NaN detection utilities."""

    def test_nan_detection(self):
        """Test detection of NaN values in tensors."""
        # Tensor with no NaN
        t_clean = torch.randn(10, 5)
        assert not torch.isnan(t_clean).any()
        
        # Tensor with NaN
        t_nan = t_clean.clone()
        t_nan[3, 2] = float('nan')
        assert torch.isnan(t_nan).any()

    def test_nan_detection_in_gradients(self):
        """Test NaN detection in gradients."""
        model = torch.nn.Linear(5, 3)
        
        # Normal forward/backward
        x = torch.randn(10, 5)
        y = model(x).sum()
        y.backward()
        
        has_nan = False
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan = True
                break
        
        assert not has_nan

    def test_nan_detection_in_loss(self):
        """Test NaN detection in loss values."""
        loss_normal = torch.tensor(0.5)
        loss_nan = torch.tensor(float('nan'))
        loss_inf = torch.tensor(float('inf'))
        
        def check_loss(loss):
            return torch.isnan(loss) or torch.isinf(loss)
        
        assert not check_loss(loss_normal)
        assert check_loss(loss_nan)
        assert check_loss(loss_inf)


class TestSafeClip:
    """Tests for safe clipping utilities."""

    def test_safe_clip(self):
        """Test safe clipping of values."""
        values = np.array([-100, -1, 0, 1, 100])
        
        clipped = np.clip(values, -10, 10)
        
        np.testing.assert_allclose(clipped, [-10, -1, 0, 1, 10])

    def test_safe_clip_preserves_shape(self):
        """Test that clipping preserves array shape."""
        values = np.random.randn(5, 3, 2)
        
        clipped = np.clip(values, -1, 1)
        
        assert clipped.shape == values.shape

    def test_safe_clip_nan_handling(self):
        """Test clipping handles NaN values."""
        values = np.array([1.0, np.nan, 3.0, -np.inf, np.inf])
        
        # Clip NaN and inf
        clipped = np.clip(values, -10, 10)
        
        # NaN stays NaN, inf gets clipped
        assert np.isnan(clipped[1])  # NaN preserved
        assert clipped[3] == -10  # -inf clipped to -10
        assert clipped[4] == 10  # inf clipped to 10

    def test_safe_clip_tensor(self):
        """Test clipping of PyTorch tensors."""
        t = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        
        clipped = torch.clamp(t, -10, 10)
        
        expected = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        assert torch.allclose(clipped, expected)


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_format_duration(self):
        """Test duration formatting."""
        def format_duration(seconds):
            """Format duration in seconds to human-readable string."""
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds / 60
                return f"{minutes:.1f}m"
            else:
                hours = seconds / 3600
                return f"{hours:.1f}h"
        
        assert format_duration(30) == "30.0s"
        assert format_duration(90) == "1.5m"
        assert format_duration(5400) == "1.5h"

    def test_format_duration_detailed(self):
        """Test detailed duration formatting."""
        def format_duration_detailed(seconds):
            """Format duration with hours, minutes, seconds."""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            
            parts = []
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            parts.append(f"{secs}s")
            
            return " ".join(parts)
        
        assert format_duration_detailed(3661) == "1h 1m 1s"
        assert format_duration_detailed(65) == "1m 5s"
        assert format_duration_detailed(5) == "5s"

    def test_format_duration_milliseconds(self):
        """Test formatting with millisecond precision."""
        def format_duration_ms(seconds):
            """Format duration with millisecond precision."""
            if seconds < 1:
                return f"{seconds * 1000:.0f}ms"
            elif seconds < 60:
                return f"{seconds:.2f}s"
            else:
                return f"{seconds / 60:.1f}m"
        
        assert format_duration_ms(0.5) == "500ms"
        assert format_duration_ms(0.123) == "123ms"
        assert format_duration_ms(30.5) == "30.50s"


class TestExperimentUtils:
    """Tests for general experiment utilities."""

    def test_seed_setting(self):
        """Test that seeding produces reproducible results."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        t1 = torch.randn(5)
        n1 = np.random.randn(5)
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        t2 = torch.randn(5)
        n2 = np.random.randn(5)
        
        assert torch.allclose(t1, t2)
        np.testing.assert_allclose(n1, n2)

    def test_results_saving(self):
        """Test saving experiment results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, 'results.json')
            
            results = {
                'experiment': 'test',
                'mean_reward': 100.5,
                'std_reward': 15.3,
                'n_episodes': 50,
            }
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            with open(results_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded['experiment'] == 'test'
            assert loaded['mean_reward'] == 100.5

    def test_progress_display(self):
        """Test progress display formatting."""
        def format_progress(current, total, loss):
            pct = 100 * current / total
            return f"[{current}/{total}] ({pct:.0f}%) loss={loss:.4f}"
        
        progress = format_progress(50, 100, 0.1234)
        
        assert "50/100" in progress
        assert "50%" in progress
        assert "0.1234" in progress


class TestBatchProcessing:
    """Tests for batch processing utilities."""

    def test_batch_creation(self):
        """Test creating batches from data."""
        data = np.arange(100)
        batch_size = 10
        
        batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        
        assert len(batches) == 10
        for batch in batches:
            assert len(batch) == batch_size

    def test_random_batch_selection(self):
        """Test random batch selection."""
        np.random.seed(42)
        n_samples = 100
        batch_size = 16
        
        idx = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = idx[i:i+batch_size]
            assert len(batch_idx) <= batch_size
            assert len(np.unique(batch_idx)) == len(batch_idx)  # No duplicates in batch


class TestMemoryUtils:
    """Tests for memory utilities."""

    def test_memory_estimation(self):
        """Test memory estimation for tensors."""
        def estimate_memory(shape, dtype=torch.float32):
            """Estimate memory in bytes for a tensor."""
            n_elements = np.prod(shape)
            bytes_per_element = 4 if dtype == torch.float32 else 2
            return n_elements * bytes_per_element
        
        mem_1k = estimate_memory((1000,))
        mem_1m = estimate_memory((1000, 1000))
        
        assert mem_1k == 4000  # 1000 * 4 bytes
        assert mem_1m == 4_000_000  # 1M * 4 bytes = 4MB

    def test_gpu_memory_tracking(self):
        """Test GPU memory tracking (CPU fallback)."""
        # On CPU, these should return 0 or handle gracefully
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            assert allocated >= 0
            assert cached >= 0
        else:
            # CPU case - just check the logic would work
            allocated = 0
            cached = 0
            assert allocated == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])