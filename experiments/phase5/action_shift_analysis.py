#!/usr/bin/bin/python3
"""
Action-Shift Analysis and Closed-Loop Training

Steps:
1. Confirm action-shift gap per-init
2. Measure off-manifold distance
3. Implement adversarial data aggregation (targeted DAgger)
4. Compare all estimators
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

device = torch.device('cuda')
print(f"Using: {device}")

# =============================================================================
# PROTOCOLS
# =============================================================================

DT = 0.05
HORIZON = 30
TAU = 0.3
ACTION_BOUNDS = (-2.0, 2.0)
RESTITUTION = 0.8
GRAVITY = 9.81
SEED = 42
DISCRETE_INITS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

# =============================================================================
# MODELS
# =============================================================================

class Observer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x1, x2):
        return self.net(torch.cat([x1, x2], dim=-1))

# =============================================================================
# ENVIRONMENT
# =============================================================================

class Ball:
    def __init__(self, e=RESTITUTION):
        self.e = e
    
    def reset(self, x0):
        self.x = x0
        self.v = 0.0
    
    def step(self, a):
        a = np.clip(a, ACTION_BOUNDS[0], ACTION_BOUNDS[1])
        self.v += (-GRAVITY + a) * DT
        self.x += self.v * DT
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e

# =============================================================================
# STEP 1: CONFIRM ACTION-SHIFT GAP PER-INIT
# =============================================================================

def evaluate_per_init(obs, test_inits, n_trials=100):
    """Evaluate observer per init, save trajectories"""
    obs.eval()
    results = {}
    
    for x0 in test_inits:
        successes = 0
        trajectories = []
        
        for trial in range(n_trials):
            b = Ball()
            b.reset(x0)
            xp = b.x
            
            traj = {'states': [], 'x_prev': [], 'v_true': [], 'v_pred': [], 'actions': []}
            
            for t in range(HORIZON):
                # Record state
                traj['states'].append((b.x, b.v))
                traj['x_prev'].append(xp)
                traj['v_true'].append(b.v)
                
                # Observer prediction
                with torch.no_grad():
                    v_est = obs(
                        torch.tensor([[float(b.x)]], dtype=torch.float32, device=device),
                        torch.tensor([[float(xp)]], dtype=torch.float32, device=device)
                    ).item()
                traj['v_pred'].append(v_est)
                
                # Action
                a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
                traj['actions'].append(a)
                
                xp = b.x
                b.step(a)
                
                if abs(b.x - 2.0) < TAU:
                    successes += 1
                    break
            
            trajectories.append(traj)
        
        results[x0] = {
            'success_rate': successes / n_trials,
            'trajectories': trajectories[:20]  # Save first 20
        }
    
    return results

def evaluate_fd_per_init(test_inits, n_trials=100):
    """Evaluate FD per init"""
    results = {}
    
    for x0 in test_inits:
        successes = 0
        
        for trial in range(n_trials):
            b = Ball()
            b.reset(x0)
            xp = b.x
            
            for _ in range(HORIZON):
                v_est = (b.x - xp) / DT
                xp = b.x
                a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
                b.step(a)
                if abs(b.x - 2.0) < TAU:
                    successes += 1
                    break
        
        results[x0] = successes / n_trials
    
    return results

# =============================================================================
# STEP 2: MEASURE OFF-MANIFOLD DISTANCE
# =============================================================================

def compute_off_manifold_score(trajectory, training_data):
    """Compute how far trajectory is from training distribution"""
    X_train = training_data['X']  # (N, 2) - [x, x_prev]
    
    scores = []
    for state, xp in zip(trajectory['states'], trajectory['x_prev']):
        x = state[0]
        # Nearest neighbor distance in (x, x_prev) space
        point = np.array([x, xp])
        distances = np.linalg.norm(X_train - point, axis=1)
        min_dist = np.min(distances)
        scores.append(min_dist)
    
    return np.mean(scores)

# =============================================================================
# STEP 3-4: ADVERSARIAL DATA AGGREGATION
# =============================================================================

def generate_expert_data(n_episodes=500):
    """Generate expert data from PD on full state"""
    X, Xp, V = [], [], []
    
    for ep in range(n_episodes):
        x = DISCRETE_INITS[ep % len(DISCRETE_INITS)]
        v, xp = 0.0, x
        
        for _ in range(HORIZON):
            a = np.clip(1.5*(2-x) + (-2)*(-v), *ACTION_BOUNDS)
            v += (-GRAVITY + a) * DT
            x += v * DT
            if x < 0: x, v = -x*RESTITUTION, -v*RESTITUTION
            if x > 3: x, v = 3-(x-3)*RESTITUTION, -v*RESTITUTION
            X.append(x); Xp.append(xp); V.append(v)
            xp = x
    
    return np.array(X, dtype=np.float32), np.array(Xp, dtype=np.float32), np.array(V, dtype=np.float32)

def train_observer(X, Xp, V, epochs=50, lr=0.01):
    """Train observer on data"""
    X = torch.tensor(X, device=device).unsqueeze(-1)
    Xp = torch.tensor(Xp, device=device).unsqueeze(-1)
    V = torch.tensor(V, device=device).unsqueeze(-1)
    
    obs = Observer().to(device)
    opt = torch.optim.Adam(obs.parameters(), lr=lr)
    
    for epoch in range(epochs):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = idx[i:i+64]
            pred = obs(X[b], Xp[b])
            loss = ((pred - V[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    return obs

def adversarial_data_aggregation(n_iterations=5, n_rollouts=100):
    """
    Train observer with adversarial data aggregation:
    1. Roll out current observer
    2. Collect (x, x_prev) pairs visited
    3. Label with true v from simulator
    4. Retrain on aggregated data
    """
    print("\n" + "="*70)
    print("Adversarial Data Aggregation")
    print("="*70)
    
    # Initial expert data
    X, Xp, V = generate_expert_data(500)
    training_data = {'X': np.stack([X, Xp], axis=1)}
    
    # Initial observer
    obs = train_observer(X, Xp, V)
    
    # Track success over iterations
    success_history = []
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")
        
        # Evaluate current observer
        test_inits = [1.5, 2.0, 2.5, 3.0, 3.5]  # Solvable inits
        results = evaluate_per_init(obs, test_inits, n_trials=20)
        avg_success = np.mean([r['success_rate'] for r in results.values()])
        success_history.append(avg_success)
        print(f"  Current success: {avg_success:.1%}")
        
        # Roll out and collect data
        X_new, Xp_new, V_new = [], [], []
        
        for ep in range(n_rollouts):
            x0 = DISCRETE_INITS[ep % len(DISCRETE_INITS)]
            b = Ball()
            b.reset(x0)
            xp = b.x
            
            for t in range(HORIZON):
                # Use current observer
                with torch.no_grad():
                    v_est = obs(
                        torch.tensor([[float(b.x)]], dtype=torch.float32, device=device),
                        torch.tensor([[float(xp)]], dtype=torch.float32, device=device)
                    ).item()
                
                # Record (x, x_prev) and TRUE velocity
                X_new.append(b.x)
                Xp_new.append(xp)
                V_new.append(b.v)  # True velocity from simulator
                
                # Step with observer's action
                a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
                xp = b.x
                b.step(a)
        
        # Aggregate data
        X_new = np.array(X_new, dtype=np.float32)
        Xp_new = np.array(Xp_new, dtype=np.float32)
        V_new = np.array(V_new, dtype=np.float32)
        
        X = np.concatenate([X, X_new])
        Xp = np.concatenate([Xp, Xp_new])
        V = np.concatenate([V, V_new])
        
        training_data['X'] = np.stack([X, Xp], axis=1)
        
        print(f"  Aggregated data: {len(X)} samples")
        
        # Retrain
        obs = train_observer(X, Xp, V, epochs=30)
    
    return obs, success_history

# =============================================================================
# STEP 5: COMPARE ALL ESTIMATORS
# =============================================================================

def compare_all_estimators():
    """Compare FD, supervised, and aggregated observers"""
    print("\n" + "="*70)
    print("Comparing All Estimators")
    print("="*70)
    
    test_inits = DISCRETE_INITS
    
    # 1. FD baseline
    print("\nEvaluating FD...")
    fd_results = evaluate_fd_per_init(test_inits, n_trials=100)
    
    # 2. Supervised observer
    print("Training supervised observer...")
    X, Xp, V = generate_expert_data(500)
    obs_supervised = train_observer(X, Xp, V)
    supervised_results = evaluate_per_init(obs_supervised, test_inits, n_trials=100)
    
    # 3. Aggregated observer
    obs_aggregated, success_history = adversarial_data_aggregation(n_iterations=5, n_rollouts=100)
    aggregated_results = evaluate_per_init(obs_aggregated, test_inits, n_trials=100)
    
    # Print comparison
    print("\n" + "="*70)
    print("Results: Success Rate per Initial Position")
    print("="*70)
    print(f"{'x0':6s} | {'FD':6s} | {'Supervised':10s} | {'Aggregated':10s}")
    print("-"*50)
    
    for x0 in test_inits:
        fd_s = fd_results[x0]
        sup_s = supervised_results[x0]['success_rate']
        agg_s = aggregated_results[x0]['success_rate']
        print(f"{x0:6.1f} | {fd_s:6.0%} | {sup_s:10.0%} | {agg_s:10.0%}")
    
    # Overall
    fd_avg = np.mean(list(fd_results.values()))
    sup_avg = np.mean([r['success_rate'] for r in supervised_results.values()])
    agg_avg = np.mean([r['success_rate'] for r in aggregated_results.values()])
    
    print("-"*50)
    print(f"{'Avg':6s} | {fd_avg:6.1%} | {sup_avg:10.1%} | {agg_avg:10.1%}")
    
    return {
        'fd': fd_results,
        'supervised': {x0: r['success_rate'] for x0, r in supervised_results.items()},
        'aggregated': {x0: r['success_rate'] for x0, r in aggregated_results.items()},
        'agg_history': success_history
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("="*70)
    print("Action-Shift Analysis")
    print("="*70)
    
    # Compare all estimators
    results = compare_all_estimators()
    
    # Save results
    output_path = Path(__file__).parent / 'action_shift_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")