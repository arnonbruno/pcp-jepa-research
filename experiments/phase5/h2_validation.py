#!/usr/bin/env python3
"""
H2 Validation Suite: OOD Generalization Curves

Protocols:
- Continuous init: uniform in [x_min, x_max]
- Discrete init: fixed set of points (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
- Same noise seeds, dt=0.05, horizon=30, tau=0.3, action bounds=[-2, 2]
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

device = torch.device('cuda')
print(f"Using: {device}")

# =============================================================================
# PROTOCOLS (LOCKED)
# =============================================================================

DT = 0.05
HORIZON = 30
TAU = 0.3
ACTION_BOUNDS = (-2.0, 2.0)
RESTITUTION = 0.8
GRAVITY = 9.81
SEED = 42

# Discrete init set (exact points)
DISCRETE_INITS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

# Continuous init range
CONTINUOUS_RANGE = (0.5, 3.5)

# Test grid for success curve
TEST_GRID = np.arange(0.5, 1.55, 0.05)  # Fine grid from 0.5 to 1.5


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
# DATA GENERATION
# =============================================================================

def generate_data_continuous(x_range, n_episodes, seed=SEED):
    """Generate training data from continuous uniform initial positions"""
    np.random.seed(seed)
    X, Xp, V = [], [], []
    
    for ep in range(n_episodes):
        x = np.random.uniform(*x_range)
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


def generate_data_discrete(init_set, n_episodes_per_init, seed=SEED):
    """Generate training data from discrete initial positions"""
    np.random.seed(seed)
    X, Xp, V = [], [], []
    
    for x0 in init_set:
        for ep in range(n_episodes_per_init):
            x, v, xp = x0, 0.0, x0
            
            for _ in range(HORIZON):
                a = np.clip(1.5*(2-x) + (-2)*(-v), *ACTION_BOUNDS)
                v += (-GRAVITY + a) * DT
                x += v * DT
                if x < 0: x, v = -x*RESTITUTION, -v*RESTITUTION
                if x > 3: x, v = 3-(x-3)*RESTITUTION, -v*RESTITUTION
                X.append(x); Xp.append(xp); V.append(v)
                xp = x
    
    return np.array(X, dtype=np.float32), np.array(Xp, dtype=np.float32), np.array(V, dtype=np.float32)


# =============================================================================
# TRAINING
# =============================================================================

def train_observer(X, Xp, V, epochs=50, lr=0.01):
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


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_fd(test_inits, n_trials=100):
    """Evaluate finite-diff controller"""
    successes = {x0: 0 for x0 in test_inits}
    
    for x0 in test_inits:
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
                    successes[x0] += 1
                    break
    
    return {x0: s/n_trials for x0, s in successes.items()}


def evaluate_observer(obs, test_inits, n_trials=100):
    """Evaluate learned observer controller"""
    obs.eval()
    successes = {x0: 0 for x0 in test_inits}
    
    for x0 in test_inits:
        for trial in range(n_trials):
            b = Ball()
            b.reset(x0)
            xp = b.x
            
            for _ in range(HORIZON):
                with torch.no_grad():
                    v_est = obs(
                        torch.tensor([[float(b.x)]], dtype=torch.float32, device=device),
                        torch.tensor([[float(xp)]], dtype=torch.float32, device=device)
                    ).item()
                a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
                xp = b.x
                b.step(a)
                if abs(b.x - 2.0) < TAU:
                    successes[x0] += 1
                    break
    
    return {x0: s/n_trials for x0, s in successes.items()}


# =============================================================================
# V-H2.1: SUCCESS CURVE
# =============================================================================

def run_vh21_success_curve():
    """Build success(x0) curve for FD and observers"""
    print("\n" + "="*70)
    print("V-H2.1: Success(x0) Curve")
    print("="*70)
    
    # Train observers
    print("\nTraining continuous observer...")
    Xc, Xpc, Vc = generate_data_continuous(CONTINUOUS_RANGE, 500)
    obs_cont = train_observer(Xc, Xpc, Vc)
    
    print("Training discrete observer...")
    Xd, Xpd, Vd = generate_data_discrete(DISCRETE_INITS, 71)  # ~500 episodes
    obs_disc = train_observer(Xd, Xpd, Vd)
    
    # Evaluate
    print("\nEvaluating on fine grid...")
    test_grid = list(TEST_GRID) + [1.5, 2.0, 2.5, 3.0, 3.5]
    test_grid = sorted(set(test_grid))
    
    fd_results = evaluate_fd(test_grid, n_trials=100)
    cont_results = evaluate_observer(obs_cont, test_grid, n_trials=100)
    disc_results = evaluate_observer(obs_disc, test_grid, n_trials=100)
    
    # Print results
    print("\nx0     | FD     | Cont-Obs | Disc-Obs")
    print("-"*50)
    for x0 in test_grid:
        print(f"{x0:5.2f}  | {fd_results[x0]:5.0%}  |  {cont_results[x0]:5.0%}   |  {disc_results[x0]:5.0%}")
    
    return fd_results, cont_results, disc_results


# =============================================================================
# V-H2.2: 4-WAY MATRIX
# =============================================================================

def run_vh22_matrix():
    """4-way train/test matrix"""
    print("\n" + "="*70)
    print("V-H2.2: 4-Way Train/Test Matrix")
    print("="*70)
    
    np.random.seed(SEED)
    
    # Define test sets (fixed number of test points)
    test_cont = list(np.random.uniform(*CONTINUOUS_RANGE, 200))
    test_disc = DISCRETE_INITS * 29  # 7 * 29 = 203 trials
    
    # Train observers
    print("\nTraining continuous observer...")
    Xc, Xpc, Vc = generate_data_continuous(CONTINUOUS_RANGE, 500)
    obs_cont = train_observer(Xc, Xpc, Vc)
    
    print("Training discrete observer...")
    Xd, Xpd, Vd = generate_data_discrete(DISCRETE_INITS, 71)
    obs_disc = train_observer(Xd, Xpd, Vd)
    
    # Evaluate all combinations
    results = {}
    
    # Evaluate observer on continuous test
    print("\nEvaluating continuous observer on continuous test...")
    success = 0
    for x0 in test_cont:
        b = Ball(); b.reset(x0); xp = b.x
        for _ in range(HORIZON):
            with torch.no_grad():
                v_est = obs_cont(
                    torch.tensor([[float(b.x)]], dtype=torch.float32, device=device),
                    torch.tensor([[float(xp)]], dtype=torch.float32, device=device)
                ).item()
            a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
            xp = b.x
            b.step(a)
            if abs(b.x - 2.0) < TAU:
                success += 1
                break
    results['cont_train_cont_test'] = success / len(test_cont)
    print(f"Continuous train → continuous test: {results['cont_train_cont_test']:.1%}")
    
    # Evaluate observer on discrete test
    print("Evaluating continuous observer on discrete test...")
    success = 0
    for x0 in test_disc:
        b = Ball(); b.reset(x0); xp = b.x
        for _ in range(HORIZON):
            with torch.no_grad():
                v_est = obs_cont(
                    torch.tensor([[float(b.x)]], dtype=torch.float32, device=device),
                    torch.tensor([[float(xp)]], dtype=torch.float32, device=device)
                ).item()
            a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
            xp = b.x
            b.step(a)
            if abs(b.x - 2.0) < TAU:
                success += 1
                break
    results['cont_train_disc_test'] = success / len(test_disc)
    print(f"Continuous train → discrete test: {results['cont_train_disc_test']:.1%}")
    
    # Evaluate discrete observer on continuous test
    print("Evaluating discrete observer on continuous test...")
    success = 0
    for x0 in test_cont:
        b = Ball(); b.reset(x0); xp = b.x
        for _ in range(HORIZON):
            with torch.no_grad():
                v_est = obs_disc(
                    torch.tensor([[float(b.x)]], dtype=torch.float32, device=device),
                    torch.tensor([[float(xp)]], dtype=torch.float32, device=device)
                ).item()
            a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
            xp = b.x
            b.step(a)
            if abs(b.x - 2.0) < TAU:
                success += 1
                break
    results['disc_train_cont_test'] = success / len(test_cont)
    print(f"Discrete train → continuous test: {results['disc_train_cont_test']:.1%}")
    
    # Evaluate discrete observer on discrete test
    print("Evaluating discrete observer on discrete test...")
    success = 0
    for x0 in test_disc:
        b = Ball(); b.reset(x0); xp = b.x
        for _ in range(HORIZON):
            with torch.no_grad():
                v_est = obs_disc(
                    torch.tensor([[float(b.x)]], dtype=torch.float32, device=device),
                    torch.tensor([[float(xp)]], dtype=torch.float32, device=device)
                ).item()
            a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
            xp = b.x
            b.step(a)
            if abs(b.x - 2.0) < TAU:
                success += 1
                break
    results['disc_train_disc_test'] = success / len(test_disc)
    print(f"Discrete train → discrete test: {results['disc_train_disc_test']:.1%}")
    
    # FD baseline
    print("\nEvaluating FD baselines...")
    success = 0
    for x0 in test_cont:
        b = Ball(); b.reset(x0); xp = b.x
        for _ in range(HORIZON):
            v_est = (b.x - xp) / DT
            xp = b.x
            a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
            b.step(a)
            if abs(b.x - 2.0) < TAU:
                success += 1
                break
    results['fd_continuous'] = success / len(test_cont)
    print(f"FD → continuous test: {results['fd_continuous']:.1%}")
    
    success = 0
    for x0 in test_disc:
        b = Ball(); b.reset(x0); xp = b.x
        for _ in range(HORIZON):
            v_est = (b.x - xp) / DT
            xp = b.x
            a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
            b.step(a)
            if abs(b.x - 2.0) < TAU:
                success += 1
                break
    results['fd_discrete'] = success / len(test_disc)
    print(f"FD → discrete test: {results['fd_discrete']:.1%}")
    
    return results


# =============================================================================
# QUANTIFY HOLES
# =============================================================================

def quantify_holes(fd_results):
    """Quantify failure regions"""
    print("\n" + "="*70)
    print("Failure Region Analysis")
    print("="*70)
    
    # Find failure threshold
    failure_threshold = 0.1
    
    failures = [x0 for x0, rate in fd_results.items() if rate < failure_threshold]
    successes = [x0 for x0, rate in fd_results.items() if rate >= failure_threshold]
    
    print(f"\nFailure region (success < {failure_threshold:.0%}):")
    if failures:
        print(f"  x0 ∈ [{min(failures):.2f}, {max(failures):.2f}]")
        print(f"  Width: {max(failures) - min(failures):.2f}")
    else:
        print("  No failure region found")
    
    print(f"\nSuccess region (success >= {failure_threshold:.0%}):")
    if successes:
        print(f"  x0 ∈ [{min(successes):.2f}, {max(successes):.2f}]")
    
    return failures, successes


# =============================================================================
# H3: DROPOUT ROBUSTNESS
# =============================================================================

def evaluate_fd_with_dropout(test_inits, dropout_rate=0.2, n_trials=100):
    """Evaluate FD with observation dropout"""
    successes = {x0: 0 for x0 in test_inits}
    np.random.seed(SEED)
    
    for x0 in test_inits:
        for trial in range(n_trials):
            b = Ball()
            b.reset(x0)
            xp = b.x
            x_prev_valid = b.x  # Last valid observation
            
            for t in range(HORIZON):
                # Dropout: with probability dropout_rate, observation is missing
                if np.random.rand() < dropout_rate:
                    # Use last valid observation
                    x_obs = x_prev_valid
                else:
                    x_obs = b.x
                    x_prev_valid = b.x
                
                v_est = (x_obs - xp) / DT
                xp = x_obs
                a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
                b.step(a)
                if abs(b.x - 2.0) < TAU:
                    successes[x0] += 1
                    break
    
    return {x0: s/n_trials for x0, s in successes.items()}


def evaluate_observer_with_dropout(obs, test_inits, dropout_rate=0.2, n_trials=100):
    """Evaluate observer with observation dropout"""
    obs.eval()
    successes = {x0: 0 for x0 in test_inits}
    np.random.seed(SEED)
    
    for x0 in test_inits:
        for trial in range(n_trials):
            b = Ball()
            b.reset(x0)
            xp = b.x
            x_prev_valid = b.x
            
            for t in range(HORIZON):
                # Dropout
                if np.random.rand() < dropout_rate:
                    x_obs = x_prev_valid
                else:
                    x_obs = b.x
                    x_prev_valid = b.x
                
                with torch.no_grad():
                    v_est = obs(
                        torch.tensor([[float(x_obs)]], dtype=torch.float32, device=device),
                        torch.tensor([[float(xp)]], dtype=torch.float32, device=device)
                    ).item()
                a = np.clip(1.5*(2-b.x) + (-2)*(-v_est), *ACTION_BOUNDS)
                xp = x_obs
                b.step(a)
                if abs(b.x - 2.0) < TAU:
                    successes[x0] += 1
                    break
    
    return {x0: s/n_trials for x0, s in successes.items()}


def run_h3_dropout():
    """H3: Evaluate robustness under observation dropout"""
    print("\n" + "="*70)
    print("H3: Dropout Robustness")
    print("="*70)
    
    # Test on discrete inits (solvable region)
    test_inits = [1.5, 2.0, 2.5, 3.0, 3.5]
    
    # Train observers
    print("\nTraining observers...")
    Xc, Xpc, Vc = generate_data_continuous(CONTINUOUS_RANGE, 500)
    obs_cont = train_observer(Xc, Xpc, Vc)
    
    Xd, Xpd, Vd = generate_data_discrete(DISCRETE_INITS, 71)
    obs_disc = train_observer(Xd, Xpd, Vd)
    
    # Evaluate at different dropout rates
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("\nDropout | FD    | Cont-Obs | Disc-Obs")
    print("-"*50)
    
    results = {}
    for dr in dropout_rates:
        fd_res = evaluate_fd_with_dropout(test_inits, dropout_rate=dr, n_trials=100)
        cont_res = evaluate_observer_with_dropout(obs_cont, test_inits, dropout_rate=dr, n_trials=100)
        disc_res = evaluate_observer_with_dropout(obs_disc, test_inits, dropout_rate=dr, n_trials=100)
        
        fd_avg = np.mean(list(fd_res.values()))
        cont_avg = np.mean(list(cont_res.values()))
        disc_avg = np.mean(list(disc_res.values()))
        
        print(f"{dr:5.0%}   | {fd_avg:5.0%} |  {cont_avg:5.0%}   |  {disc_avg:5.0%}")
        
        results[f'dropout_{int(dr*100)}'] = {
            'fd': fd_avg,
            'cont_obs': cont_avg,
            'disc_obs': disc_avg
        }
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("H2 Validation Suite")
    print("="*70)
    print(f"\nProtocols:")
    print(f"  dt = {DT}")
    print(f"  horizon = {HORIZON}")
    print(f"  tau (success threshold) = {TAU}")
    print(f"  action bounds = {ACTION_BOUNDS}")
    print(f"  discrete inits = {DISCRETE_INITS}")
    print(f"  continuous range = {CONTINUOUS_RANGE}")
    
    # Run V-H2.1
    fd_results, cont_results, disc_results = run_vh21_success_curve()
    
    # Run V-H2.2
    matrix_results = run_vh22_matrix()
    
    # Quantify holes
    failures, successes = quantify_holes(fd_results)
    
    # Run H3
    h3_results = run_h3_dropout()
    
    # Save results
    results = {
        'fd_curve': fd_results,
        'cont_curve': cont_results,
        'disc_curve': disc_results,
        'matrix': matrix_results,
        'failure_region': failures,
        'success_region': successes,
        'h3_dropout': h3_results,
        'protocols': {
            'dt': DT,
            'horizon': HORIZON,
            'tau': TAU,
            'action_bounds': ACTION_BOUNDS,
            'discrete_inits': DISCRETE_INITS,
            'continuous_range': list(CONTINUOUS_RANGE)
        }
    }
    
    output_path = Path(__file__).parent / 'h2_results.json'
    with open(output_path, 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
