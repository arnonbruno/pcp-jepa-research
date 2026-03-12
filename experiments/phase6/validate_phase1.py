#!/usr/bin/env python3
"""
Quick validation for Phase 1 fundamentals:
1. Verify contact-triggered dropout uses MuJoCo contact forces.
2. Compare the legacy EKF against the tuned/calibrated EKF baseline.

This script is intentionally lightweight: it uses short random-action rollouts so
the validation can run in a few seconds without downloading pretrained policies.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.contact_dropout import ContactDropoutEnv
from src.models.ekf import EKFEstimator


def sample_action(action_space, rng):
    low = np.asarray(action_space.low, dtype=float)
    high = np.asarray(action_space.high, dtype=float)
    low = np.where(np.isfinite(low), low, -1.0)
    high = np.where(np.isfinite(high), high, 1.0)
    return rng.uniform(low=low, high=high).astype(np.float32)


def classification_metrics(pred, target):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = int(np.sum(pred & target))
    fp = int(np.sum(pred & ~target))
    fn = int(np.sum(~pred & target))
    tn = int(np.sum(~pred & ~target))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": float(np.mean(pred)) if len(pred) else 0.0,
    }


def collect_rollouts(
    env_id,
    episodes=6,
    max_steps=250,
    dropout_duration=5,
    velocity_threshold=0.1,
    seed=42,
):
    env = ContactDropoutEnv(
        env_id=env_id,
        dropout_duration=dropout_duration,
        velocity_threshold=velocity_threshold,
    )
    rng = np.random.default_rng(seed)

    trajectories = []
    dropout_masks = []
    old_proxy_flags = []
    physics_flags = []
    ncon_flags = []
    contact_forces = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        prev_true_obs = obs.copy()
        trajectory = [obs.copy()]
        dropout_mask = [False]

        for _ in range(max_steps):
            action = sample_action(env.action_space, rng)
            _, _, term, trunc, info = env.step(action)
            true_obs = info["true_obs"].copy()

            delta_max = float(np.max(np.abs(true_obs - prev_true_obs)))
            old_proxy_flags.append(
                bool(env.step_count > env.warmup_steps and delta_max > velocity_threshold)
            )
            physics_flags.append(bool(info["contact_detected"]))
            ncon_flags.append(bool(info.get("ncon", 0) > 0))
            contact_forces.append(float(info.get("contact_force_max", 0.0)))

            trajectory.append(true_obs)
            dropout_mask.append(bool(info["dropout_active"]))
            prev_true_obs = true_obs

            if term or trunc:
                break

        if len(trajectory) > 2:
            trajectories.append(np.asarray(trajectory))
            dropout_masks.append(np.asarray(dropout_mask, dtype=bool))

    metadata = {
        "env_id": env_id,
        "dt": float(env.env.unwrapped.dt) if hasattr(env.env.unwrapped, "dt") else 0.002,
        "contact_force_threshold": float(env.contact_force_threshold),
        "contact_source": getattr(env, "_contact_source", None),
        "dropout_duration": int(dropout_duration),
        "velocity_threshold_alias": float(velocity_threshold),
    }
    env.close()

    return {
        "trajectories": trajectories,
        "dropout_masks": dropout_masks,
        "old_proxy_flags": np.asarray(old_proxy_flags, dtype=bool),
        "physics_flags": np.asarray(physics_flags, dtype=bool),
        "ncon_flags": np.asarray(ncon_flags, dtype=bool),
        "contact_forces": np.asarray(contact_forces, dtype=float),
        "metadata": metadata,
    }


def evaluate_ekf(trajectories, dropout_masks, dt, calibration=None, env_id=None):
    dropout_rmses = []
    overall_rmses = []

    for traj, dropout_mask in zip(trajectories, dropout_masks):
        if calibration is None:
            ekf = EKFEstimator.legacy_baseline(obs_dim=traj.shape[1], dt=dt)
        else:
            ekf = EKFEstimator.from_calibration(
                calibration,
                obs_dim=traj.shape[1],
                dt=dt,
                env_id=env_id,
            )

        estimates = ekf.filter_sequence(traj, measurement_mask=~dropout_mask)
        eval_mask = dropout_mask if np.any(dropout_mask) else np.ones(len(traj), dtype=bool)

        dropout_error = traj[eval_mask] - estimates[eval_mask]
        overall_error = traj - estimates

        dropout_rmses.append(float(np.sqrt(np.mean(dropout_error ** 2))))
        overall_rmses.append(float(np.sqrt(np.mean(overall_error ** 2))))

    return {
        "dropout_rmse": float(np.mean(dropout_rmses)) if dropout_rmses else 0.0,
        "overall_rmse": float(np.mean(overall_rmses)) if overall_rmses else 0.0,
    }


def run_validation(args):
    results = {
        "phase": "phase1_fundamentals",
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "dropout_duration": args.dropout_duration,
        "velocity_threshold_alias": args.velocity_threshold,
        "environments": {},
    }

    for env_id in args.envs:
        data = collect_rollouts(
            env_id=env_id,
            episodes=args.episodes,
            max_steps=args.max_steps,
            dropout_duration=args.dropout_duration,
            velocity_threshold=args.velocity_threshold,
            seed=args.seed,
        )

        trajectories = data["trajectories"]
        dropout_masks = data["dropout_masks"]
        split_idx = max(1, len(trajectories) // 2)
        calibration = EKFEstimator.auto_calibrate(
            trajectories[:split_idx],
            dt=data["metadata"]["dt"],
            env_id=env_id,
            dropout_masks=dropout_masks[:split_idx],
        )

        legacy_metrics = evaluate_ekf(
            trajectories[split_idx:],
            dropout_masks[split_idx:],
            dt=data["metadata"]["dt"],
            calibration=None,
        )
        tuned_metrics = evaluate_ekf(
            trajectories[split_idx:],
            dropout_masks[split_idx:],
            dt=data["metadata"]["dt"],
            calibration=calibration,
            env_id=env_id,
        )

        old_vs_force = classification_metrics(data["old_proxy_flags"], data["physics_flags"])
        old_vs_ncon = classification_metrics(data["old_proxy_flags"], data["ncon_flags"])
        physics_vs_ncon = classification_metrics(data["physics_flags"], data["ncon_flags"])

        improvement = 0.0
        if legacy_metrics["dropout_rmse"] > 0:
            improvement = (
                (legacy_metrics["dropout_rmse"] - tuned_metrics["dropout_rmse"])
                / legacy_metrics["dropout_rmse"]
            ) * 100.0

        env_results = {
            "metadata": data["metadata"],
            "contact_detection": {
                "old_proxy_vs_force_trigger": old_vs_force,
                "old_proxy_vs_ncon": old_vs_ncon,
                "physics_trigger_vs_ncon": physics_vs_ncon,
                "force_p50": float(np.percentile(data["contact_forces"], 50)),
                "force_p95": float(np.percentile(data["contact_forces"], 95)),
                "ncon_rate": float(np.mean(data["ncon_flags"])),
                "physics_trigger_rate": float(np.mean(data["physics_flags"])),
            },
            "ekf": {
                "legacy": legacy_metrics,
                "tuned": tuned_metrics,
                "dropout_rmse_improvement_pct": float(improvement),
                "calibration": calibration,
            },
        }
        results["environments"][env_id] = env_results

        print("\n" + "=" * 70)
        print(env_id)
        print("=" * 70)
        print(
            f"Contact threshold: {data['metadata']['contact_force_threshold']:.1f}"
            f" via {data['metadata']['contact_source']}"
        )
        print(
            "Old proxy vs force trigger:"
            f" precision={old_vs_force['precision']:.2f},"
            f" recall={old_vs_force['recall']:.2f},"
            f" f1={old_vs_force['f1']:.2f}"
        )
        print(
            "Physics trigger vs ncon:"
            f" precision={physics_vs_ncon['precision']:.2f},"
            f" recall={physics_vs_ncon['recall']:.2f},"
            f" f1={physics_vs_ncon['f1']:.2f}"
        )
        print(
            f"Legacy EKF dropout RMSE: {legacy_metrics['dropout_rmse']:.4f}"
            f" | Tuned EKF dropout RMSE: {tuned_metrics['dropout_rmse']:.4f}"
            f" | Improvement: {improvement:+.1f}%"
        )
        print(
            "Tuned EKF params:"
            f" process_noise={calibration['process_noise']:.6f},"
            f" measurement_noise={calibration['measurement_noise']:.6f},"
            f" search_rmse={calibration['best_rmse']:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved validation results to:", output_path)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Phase 1 physics and EKF fixes")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"],
    )
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--dropout-duration", type=int, default=5)
    parser.add_argument("--velocity-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "phase6" / "phase1_validation.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_validation(parse_args())
