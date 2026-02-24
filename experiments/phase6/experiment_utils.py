"""
Utilities for robust experiment execution with:
- Checkpoint/resume capability
- Progress logging
- NaN detection
- Memory monitoring
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional
import numpy as np
import torch

CHECKPOINT_DIR = Path(__file__).parent.parent.parent / ".checkpoints"
LOG_DIR = Path(__file__).parent.parent.parent / "logs"

def ensure_dirs():
    """Create checkpoint and log directories."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

def save_checkpoint(name: str, stage: str, progress: float, data: Dict[str, Any]):
    """Save checkpoint with timestamp and progress."""
    ensure_dirs()
    checkpoint = {
        "name": name,
        "stage": stage,
        "progress": progress,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    checkpoint_file = CHECKPOINT_DIR / f"{name}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)
    
    # Also update the main checkpoint for the runner
    main_checkpoint = CHECKPOINT_DIR / "current_stage.json"
    with open(main_checkpoint, 'w') as f:
        json.dump({"stage": stage, "progress": progress, "timestamp": datetime.now().isoformat()}, f)
    
    print(f"[Checkpoint] {name}: {stage} @ {progress:.1f}%")

def load_checkpoint(name: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint if exists."""
    checkpoint_file = CHECKPOINT_DIR / f"{name}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return None

def clear_checkpoint(name: str):
    """Remove checkpoint file."""
    checkpoint_file = CHECKPOINT_DIR / f"{name}.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()

def log_progress(experiment: str, message: str, level: str = "INFO"):
    """Log progress to file and stdout."""
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] [{experiment}] {message}"
    
    log_file = LOG_DIR / f"{experiment}.log"
    with open(log_file, 'a') as f:
        f.write(log_line + "\n")
    
    print(log_line)

def check_nan(tensor_or_array, name: str = "tensor") -> bool:
    """Check for NaN/Inf and return True if found."""
    if isinstance(tensor_or_array, torch.Tensor):
        has_nan = torch.any(torch.isnan(tensor_or_array)).item()
        has_inf = torch.any(torch.isinf(tensor_or_array)).item()
    else:
        has_nan = np.any(np.isnan(tensor_or_array))
        has_inf = np.any(np.isinf(tensor_or_array))
    
    if has_nan or has_inf:
        print(f"[WARNING] NaN/Inf detected in {name}!")
        return True
    return False

def safe_clip(tensor_or_array, min_val: float = -100.0, max_val: float = 100.0):
    """Safely clip values, replacing NaN/Inf with zeros first."""
    if isinstance(tensor_or_array, torch.Tensor):
        tensor_or_array = torch.where(
            torch.isnan(tensor_or_array) | torch.isinf(tensor_or_array),
            torch.zeros_like(tensor_or_array),
            tensor_or_array
        )
        return torch.clamp(tensor_or_array, min_val, max_val)
    else:
        tensor_or_array = np.where(
            np.isnan(tensor_or_array) | np.isinf(tensor_or_array),
            np.zeros_like(tensor_or_array),
            tensor_or_array
        )
        return np.clip(tensor_or_array, min_val, max_val)

class ExperimentRunner:
    """
    Context manager for running experiments with automatic:
    - Checkpoint saving
    - Progress logging
    - Error handling
    - Resume capability
    """
    
    def __init__(self, name: str, stages: list):
        self.name = name
        self.stages = stages
        self.current_stage = 0
        self.start_time = time.time()
        ensure_dirs()
        
    def __enter__(self):
        # Check for resume
        checkpoint = load_checkpoint(self.name)
        if checkpoint:
            print(f"[Resume] Found checkpoint at stage '{checkpoint['stage']}' ({checkpoint['progress']:.1f}%)")
            # Find stage index
            for i, stage in enumerate(self.stages):
                if stage == checkpoint['stage']:
                    self.current_stage = i
                    break
        
        log_progress(self.name, f"Starting experiment (stage {self.current_stage + 1}/{len(self.stages)})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Save error checkpoint
            error_msg = f"{exc_type.__name__}: {exc_val}"
            save_checkpoint(self.name, self.stages[self.current_stage], -1, {
                "error": error_msg,
                "traceback": traceback.format_exc()
            })
            log_progress(self.name, f"FAILED: {error_msg}", level="ERROR")
            return False  # Re-raise exception
        
        # Save completion checkpoint
        elapsed = time.time() - self.start_time
        save_checkpoint(self.name, "complete", 100.0, {"elapsed_seconds": elapsed})
        log_progress(self.name, f"Completed in {elapsed:.1f}s")
        return True
    
    def advance_stage(self, data: Optional[Dict] = None):
        """Advance to next stage with checkpoint."""
        progress = ((self.current_stage + 1) / len(self.stages)) * 100
        save_checkpoint(self.name, self.stages[self.current_stage], progress, data or {})
        log_progress(self.name, f"Stage complete: {self.stages[self.current_stage]}")
        self.current_stage += 1
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with experiment context."""
        log_progress(self.name, message, level)
    
    def check_progress(self, iteration: int, total: int, checkpoint_data: Optional[Dict] = None):
        """Log progress and save checkpoint periodically."""
        progress = (iteration / total) * 100
        
        # Log every 10%
        if iteration % max(1, total // 10) == 0:
            elapsed = time.time() - self.start_time
            eta = (elapsed / iteration) * (total - iteration) if iteration > 0 else 0
            self.log(f"Progress: {iteration}/{total} ({progress:.1f}%) - ETA: {eta:.0f}s")
        
        # Save checkpoint every 25%
        if iteration % max(1, total // 4) == 0:
            save_checkpoint(self.name, self.stages[self.current_stage], progress, checkpoint_data or {})

def monitor_memory():
    """Return current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return -1

def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
