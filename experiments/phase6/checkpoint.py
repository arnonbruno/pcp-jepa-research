"""
Checkpoint utilities for long-running experiments.

Usage:
    from checkpoint import checkpoint, resume_or_run
    
    @checkpoint("experiment_name")
    def run_experiment():
        # Long computation
        return results
"""

import json
import os
from pathlib import Path
from functools import wraps
from datetime import datetime

CHECKPOINT_DIR = Path(__file__).parent.parent.parent / ".checkpoints"

def save_checkpoint(name: str, data: dict, progress: float = 0.0):
    """Save checkpoint data."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{name}.json"
    checkpoint = {
        "name": name,
        "progress": progress,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"[Checkpoint] {name}: {progress:.1f}% saved")

def load_checkpoint(name: str) -> dict | None:
    """Load checkpoint data if exists."""
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
        print(f"[Checkpoint] {name}: cleared")

def checkpoint(name: str):
    """Decorator to checkpoint function progress."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check for existing checkpoint
            existing = load_checkpoint(name)
            if existing and existing.get("complete", False):
                print(f"[Checkpoint] {name}: resuming from {existing['progress']:.1f}%")
                return existing["data"]
            
            # Run function
            result = func(*args, **kwargs)
            
            # Save final checkpoint
            save_checkpoint(name, result, progress=100.0)
            
            return result
        return wrapper
    return decorator

def resume_or_run(name: str, run_func, resume_from: dict | None = None):
    """
    Resume from checkpoint or run fresh.
    
    Args:
        name: Checkpoint name
        run_func: Function to run if no checkpoint exists
        resume_from: Optional checkpoint data to resume from
    
    Returns:
        Result data
    """
    existing = load_checkpoint(name)
    
    if existing and existing.get("complete", False):
        print(f"[Checkpoint] {name}: returning cached result")
        return existing["data"]
    
    if resume_from:
        print(f"[Checkpoint] {name}: resuming from provided state")
        return run_func(resume_from)
    
    print(f"[Checkpoint] {name}: running fresh")
    result = run_func()
    
    # Save as complete
    save_checkpoint(name, result, progress=100.0)
    
    return result

class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, name: str, total: int):
        self.name = name
        self.total = total
        self.current = 0
        self._load_progress()
    
    def _load_progress(self):
        """Load existing progress."""
        existing = load_checkpoint(self.name)
        if existing:
            self.current = existing.get("data", {}).get("current", 0)
            print(f"[Progress] {self.name}: resuming from {self.current}/{self.total}")
    
    def update(self, amount: int = 1, data: dict = None):
        """Update progress."""
        self.current += amount
        progress = (self.current / self.total) * 100
        
        save_checkpoint(self.name, {
            "current": self.current,
            "total": self.total,
            **(data or {})
        }, progress)
    
    def complete(self, result: dict):
        """Mark as complete with final result."""
        save_checkpoint(self.name, {
            "complete": True,
            **result
        }, progress=100.0)
        
        # Copy to final result file
        final_file = CHECKPOINT_DIR / f"{self.name}_final.json"
        checkpoint_file = CHECKPOINT_DIR / f"{self.name}.json"
        if checkpoint_file.exists():
            import shutil
            shutil.copy(checkpoint_file, final_file)