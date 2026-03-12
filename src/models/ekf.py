import itertools
from typing import Iterable, List, Sequence

import numpy as np


class EKFEstimator:
    """
    Extended Kalman Filter for observation-space state estimation under dropout.

    The filter keeps a constant-velocity latent state `[obs, obs_velocity]` but now
    supports:
    - environment-specific noise priors,
    - diagonal noise matrix construction from calibrated feature scales,
    - grid-search tuning for process and measurement noise,
    - automatic calibration from trajectory data.
    """

    MIN_NOISE = 1e-9
    DEFAULT_DT = 0.002
    DEFAULT_PRIOR = {
        "process_noise": 1.0,
        "measurement_noise": 0.01,
        "position_process_scale": 0.01,
        "velocity_process_scale": 1.0,
        "process_noise_candidates": [0.01, 0.03, 0.1, 0.3, 1.0],
        "measurement_noise_candidates": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    }
    ENVIRONMENT_PRIORS = {
        "Hopper-v4": {
            "process_noise": 0.08,
            "measurement_noise": 0.002,
            "position_process_scale": 0.02,
            "velocity_process_scale": 1.5,
            "process_noise_candidates": [0.01, 0.03, 0.08, 0.2, 0.5],
            "measurement_noise_candidates": [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
        },
        "Walker2d-v4": {
            "process_noise": 0.03,
            "measurement_noise": 0.001,
            "position_process_scale": 0.015,
            "velocity_process_scale": 1.0,
            "process_noise_candidates": [0.005, 0.01, 0.03, 0.08, 0.2],
            "measurement_noise_candidates": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        },
        "HalfCheetah-v4": {
            "process_noise": 0.005,
            "measurement_noise": 0.0003,
            "position_process_scale": 0.008,
            "velocity_process_scale": 0.4,
            "process_noise_candidates": [5e-4, 1e-3, 5e-3, 1e-2, 3e-2],
            "measurement_noise_candidates": [1e-5, 5e-5, 1e-4, 3e-4, 1e-3],
        },
    }
    ENVIRONMENT_NOISE_PROFILES = {
        "Hopper-v4": {
            "state_process_profile": np.array(
                [1.6, 1.3, 1.1, 1.0, 0.9, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8],
                dtype=float,
            ),
            "velocity_process_profile": np.array(
                [2.6, 2.2, 1.8, 1.5, 1.3, 1.3, 1.4, 1.7, 2.0, 2.3, 2.6],
                dtype=float,
            ),
            "measurement_noise_profile": np.array(
                [1.8, 1.4, 1.1, 1.0, 0.9, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7],
                dtype=float,
            ),
        },
        "Walker2d-v4": {
            "state_process_profile": np.array(
                [
                    1.3,
                    1.2,
                    1.1,
                    1.0,
                    0.95,
                    0.95,
                    1.0,
                    1.05,
                    1.1,
                    1.15,
                    1.2,
                    1.25,
                    1.3,
                    1.35,
                    1.4,
                    1.45,
                    1.5,
                ],
                dtype=float,
            ),
            "velocity_process_profile": np.array(
                [
                    1.8,
                    1.7,
                    1.6,
                    1.5,
                    1.4,
                    1.35,
                    1.35,
                    1.4,
                    1.45,
                    1.5,
                    1.55,
                    1.6,
                    1.7,
                    1.8,
                    1.9,
                    2.0,
                    2.1,
                ],
                dtype=float,
            ),
            "measurement_noise_profile": np.array(
                [
                    1.4,
                    1.3,
                    1.2,
                    1.1,
                    1.0,
                    0.95,
                    0.95,
                    1.0,
                    1.05,
                    1.1,
                    1.15,
                    1.2,
                    1.25,
                    1.3,
                    1.35,
                    1.4,
                    1.45,
                ],
                dtype=float,
            ),
        },
        "HalfCheetah-v4": {
            "state_process_profile": np.array(
                [
                    0.9,
                    0.85,
                    0.8,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                    1.0,
                    1.05,
                    1.1,
                    1.15,
                    1.2,
                    1.15,
                    1.1,
                    1.05,
                    1.0,
                    0.95,
                ],
                dtype=float,
            ),
            "velocity_process_profile": np.array(
                [
                    1.1,
                    1.0,
                    0.95,
                    0.95,
                    1.0,
                    1.05,
                    1.1,
                    1.15,
                    1.2,
                    1.25,
                    1.3,
                    1.35,
                    1.3,
                    1.25,
                    1.2,
                    1.15,
                    1.1,
                ],
                dtype=float,
            ),
            "measurement_noise_profile": np.array(
                [
                    0.8,
                    0.75,
                    0.7,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                    1.0,
                    1.05,
                    1.1,
                    1.05,
                    1.0,
                    0.95,
                    0.9,
                    0.85,
                ],
                dtype=float,
            ),
        },
    }

    def __init__(
        self,
        obs_dim=11,
        dt=None,
        process_noise=None,
        measurement_noise=None,
        env_id=None,
        state_scale=None,
        velocity_scale=None,
        position_process_scale=None,
        velocity_process_scale=None,
    ):
        self.obs_dim = obs_dim
        self.dt = dt if dt is not None else self.DEFAULT_DT
        self.state_dim = obs_dim * 2
        self.env_id = env_id
        self.env_prior = self._get_env_prior(env_id)

        self.process_noise = (
            self.env_prior["process_noise"] if process_noise is None else process_noise
        )
        self.measurement_noise = (
            self.env_prior["measurement_noise"]
            if measurement_noise is None
            else measurement_noise
        )
        self.position_process_scale = (
            self.env_prior["position_process_scale"]
            if position_process_scale is None
            else position_process_scale
        )
        self.velocity_process_scale = (
            self.env_prior["velocity_process_scale"]
            if velocity_process_scale is None
            else velocity_process_scale
        )

        self.state_scale = self._sanitize_scale(state_scale, obs_dim)
        self.velocity_scale = self._sanitize_scale(velocity_scale, obs_dim)
        (
            self.state_process_profile,
            self.velocity_process_profile,
            self.measurement_noise_profile,
        ) = self._get_noise_profiles(env_id, obs_dim)

        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1.0
        self.H = np.zeros((obs_dim, self.state_dim))
        self.H[:obs_dim, :obs_dim] = np.eye(obs_dim)

        self.Q = np.eye(self.state_dim) * self.MIN_NOISE
        self.R = np.eye(obs_dim) * self.MIN_NOISE
        self.configure_noise(self.process_noise, self.measurement_noise)

    @classmethod
    def _get_env_prior(cls, env_id):
        prior = cls._copy_prior(cls.DEFAULT_PRIOR)
        if env_id in cls.ENVIRONMENT_PRIORS:
            prior.update(cls.ENVIRONMENT_PRIORS[env_id])
        return prior

    @staticmethod
    def _copy_prior(prior):
        copied = {}
        for key, value in prior.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, list):
                copied[key] = list(value)
            else:
                copied[key] = value
        return copied

    @staticmethod
    def _resample_profile(profile, dim):
        arr = np.asarray(profile, dtype=float)
        if arr.shape == (dim,):
            return np.maximum(arr.copy(), EKFEstimator.MIN_NOISE)
        if dim == 1:
            return np.array([max(float(np.mean(arr)), EKFEstimator.MIN_NOISE)], dtype=float)

        x_old = np.linspace(0.0, 1.0, num=len(arr))
        x_new = np.linspace(0.0, 1.0, num=dim)
        return np.maximum(np.interp(x_new, x_old, arr), EKFEstimator.MIN_NOISE)

    @classmethod
    def _get_noise_profiles(cls, env_id, obs_dim):
        defaults = (
            np.ones(obs_dim, dtype=float),
            np.ones(obs_dim, dtype=float),
            np.ones(obs_dim, dtype=float),
        )
        if env_id not in cls.ENVIRONMENT_NOISE_PROFILES:
            return defaults

        profile = cls.ENVIRONMENT_NOISE_PROFILES[env_id]
        return (
            cls._resample_profile(profile["state_process_profile"], obs_dim),
            cls._resample_profile(profile["velocity_process_profile"], obs_dim),
            cls._resample_profile(profile["measurement_noise_profile"], obs_dim),
        )

    @staticmethod
    def _is_calibration_dict(value):
        return isinstance(value, dict) and {
            "process_noise",
            "measurement_noise",
        }.issubset(value)

    @classmethod
    def legacy_baseline(cls, obs_dim=11, dt=None):
        """Reproduce the original scalar-noise EKF baseline."""
        return cls(obs_dim=obs_dim, dt=dt, process_noise=1.0, measurement_noise=0.01)

    @classmethod
    def from_calibration(
        cls,
        calibration,
        obs_dim=None,
        dt=None,
        env_id=None,
        measurement_masks=None,
        dropout_masks=None,
        process_noise_candidates=None,
        measurement_noise_candidates=None,
    ):
        """
        Instantiate a fresh EKF from calibration output or raw trajectory data.

        Passing trajectories here is a convenience wrapper around `auto_calibrate`
        followed by construction of a tuned estimator.
        """
        if not cls._is_calibration_dict(calibration):
            calibration = cls.auto_calibrate(
                calibration,
                dt=dt,
                env_id=env_id,
                measurement_masks=measurement_masks,
                dropout_masks=dropout_masks,
                process_noise_candidates=process_noise_candidates,
                measurement_noise_candidates=measurement_noise_candidates,
            )

        state_scale = calibration.get("state_scale")
        velocity_scale = calibration.get("velocity_scale")
        return cls(
            obs_dim=obs_dim if obs_dim is not None else calibration["obs_dim"],
            dt=dt if dt is not None else calibration["dt"],
            env_id=env_id if env_id is not None else calibration.get("env_id"),
            process_noise=calibration["process_noise"],
            measurement_noise=calibration["measurement_noise"],
            state_scale=(
                np.asarray(state_scale, dtype=float) if state_scale is not None else None
            ),
            velocity_scale=(
                np.asarray(velocity_scale, dtype=float)
                if velocity_scale is not None
                else None
            ),
            position_process_scale=calibration.get("position_process_scale"),
            velocity_process_scale=calibration.get("velocity_process_scale"),
        )

    @staticmethod
    def _sanitize_scale(scale, dim):
        if scale is None:
            return np.ones(dim, dtype=float)
        arr = np.asarray(scale, dtype=float)
        if arr.shape != (dim,):
            raise ValueError(f"Expected scale shape {(dim,)}, got {arr.shape}")
        return np.maximum(arr, 1e-3)

    @staticmethod
    def _to_diagonal_matrix(value, diag_dim, diag_values):
        if np.isscalar(value):
            return np.diag(np.maximum(diag_values * float(value), EKFEstimator.MIN_NOISE))

        arr = np.asarray(value, dtype=float)
        if arr.shape == (diag_dim,):
            return np.diag(np.maximum(arr, EKFEstimator.MIN_NOISE))
        if arr.shape == (diag_dim, diag_dim):
            diag = np.maximum(np.diag(arr), EKFEstimator.MIN_NOISE)
            return np.diag(diag)
        raise ValueError(
            f"Noise must be scalar, diag vector, or square matrix of size {diag_dim}; "
            f"got {arr.shape}"
        )

    @classmethod
    def _as_trajectory_list(cls, trajectories):
        if isinstance(trajectories, np.ndarray):
            if trajectories.ndim == 2:
                return [np.asarray(trajectories, dtype=float)]
            if trajectories.ndim == 3:
                return [np.asarray(traj, dtype=float) for traj in trajectories]
            raise ValueError("Trajectory array must be [T, D] or [N, T, D]")

        traj_list = [np.asarray(traj, dtype=float) for traj in trajectories]
        if not traj_list:
            raise ValueError("At least one trajectory is required for calibration")
        return traj_list

    @classmethod
    def _default_measurement_masks(cls, trajectories, dropout_window=5, stride=25, warmup=10):
        masks = []
        for traj in trajectories:
            mask = np.ones(len(traj), dtype=bool)
            if len(traj) <= warmup:
                masks.append(mask)
                continue
            for start in range(warmup, len(traj), stride):
                stop = min(start + dropout_window, len(traj))
                mask[start:stop] = False
            mask[0] = True
            masks.append(mask)
        return masks

    @classmethod
    def _normalize_measurement_masks(cls, trajectories, measurement_masks=None, dropout_masks=None):
        if measurement_masks is not None and dropout_masks is not None:
            raise ValueError("Pass either measurement_masks or dropout_masks, not both")

        if measurement_masks is not None:
            masks = measurement_masks
        elif dropout_masks is not None:
            masks = [~np.asarray(mask, dtype=bool) for mask in dropout_masks]
        else:
            return cls._default_measurement_masks(trajectories)

        if isinstance(masks, np.ndarray) and masks.ndim == 1:
            masks = [masks]
        elif isinstance(masks, np.ndarray) and masks.ndim == 2:
            masks = [mask for mask in masks]

        normalized = []
        for traj, mask in zip(trajectories, masks):
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != (len(traj),):
                raise ValueError(
                    f"Measurement mask shape {mask_arr.shape} does not match trajectory length {len(traj)}"
                )
            mask_arr = mask_arr.copy()
            mask_arr[0] = True
            normalized.append(mask_arr)
        return normalized

    @classmethod
    def estimate_noise_from_trajectories(cls, trajectories, dt=None, env_id=None):
        """
        Estimate feature scales and heuristic noise levels from trajectory data.

        The heuristics are only used to seed the grid search; the final parameters
        still come from explicit tuning on held-out dropout windows.
        """
        traj_list = cls._as_trajectory_list(trajectories)
        dt = dt if dt is not None else cls.DEFAULT_DT
        prior = cls._get_env_prior(env_id)

        stacked_obs = np.concatenate(traj_list, axis=0)
        obs_dim = stacked_obs.shape[1]
        state_scale = np.maximum(np.std(stacked_obs, axis=0), 1e-3)

        velocity_chunks = []
        acceleration_chunks = []
        residual_chunks = []
        for traj in traj_list:
            if len(traj) < 2:
                continue
            vel = np.diff(traj, axis=0) / dt
            velocity_chunks.append(vel)
            if len(vel) > 1:
                accel = np.diff(vel, axis=0) / dt
                acceleration_chunks.append(accel)
            if len(traj) > 2:
                pred = traj[1:-1] + (traj[1:-1] - traj[:-2])
                residual = traj[2:] - pred
                residual_chunks.append(residual)

        velocity_scale = (
            np.maximum(np.std(np.concatenate(velocity_chunks, axis=0), axis=0), 1e-3)
            if velocity_chunks
            else np.ones(obs_dim, dtype=float)
        )
        process_noise_estimate = prior["process_noise"]
        if acceleration_chunks:
            accel_var = np.var(np.concatenate(acceleration_chunks, axis=0), axis=0)
            process_noise_estimate = max(float(np.median(accel_var) * (dt ** 2)), cls.MIN_NOISE)

        measurement_noise_estimate = prior["measurement_noise"]
        if residual_chunks:
            residual_var = np.var(np.concatenate(residual_chunks, axis=0), axis=0)
            measurement_noise_estimate = max(float(np.median(residual_var)), cls.MIN_NOISE)

        return {
            "obs_dim": obs_dim,
            "dt": dt,
            "env_id": env_id,
            "state_scale": state_scale,
            "velocity_scale": velocity_scale,
            "process_noise_estimate": process_noise_estimate,
            "measurement_noise_estimate": measurement_noise_estimate,
            "position_process_scale": prior["position_process_scale"],
            "velocity_process_scale": prior["velocity_process_scale"],
        }

    @classmethod
    def _build_candidate_grid(cls, env_id, heuristic_value, key):
        prior = cls._get_env_prior(env_id)
        base_values = list(prior[key]) + list(cls.DEFAULT_PRIOR[key])
        lower = min(base_values) * 0.5
        upper = max(base_values) * 2.0
        if heuristic_value is not None and np.isfinite(heuristic_value):
            base_values.extend(
                [
                    np.clip(heuristic_value * 0.5, lower, upper),
                    np.clip(heuristic_value, lower, upper),
                    np.clip(heuristic_value * 2.0, lower, upper),
                ]
            )
        filtered = sorted(
            {
                float(v)
                for v in base_values
                if v is not None and np.isfinite(v) and float(v) > 0.0
            }
        )
        return filtered

    @classmethod
    def auto_calibrate(
        cls,
        trajectories,
        dt=None,
        env_id=None,
        measurement_masks=None,
        dropout_masks=None,
        process_noise_candidates=None,
        measurement_noise_candidates=None,
    ):
        """
        Tune process and measurement noise using dropout RMSE on trajectory data.
        """
        traj_list = cls._as_trajectory_list(trajectories)
        stats = cls.estimate_noise_from_trajectories(traj_list, dt=dt, env_id=env_id)
        dt = stats["dt"]
        masks = cls._normalize_measurement_masks(
            traj_list,
            measurement_masks=measurement_masks,
            dropout_masks=dropout_masks,
        )

        process_grid = (
            list(process_noise_candidates)
            if process_noise_candidates is not None
            else cls._build_candidate_grid(
                env_id,
                stats["process_noise_estimate"],
                "process_noise_candidates",
            )
        )
        measurement_grid = (
            list(measurement_noise_candidates)
            if measurement_noise_candidates is not None
            else cls._build_candidate_grid(
                env_id,
                stats["measurement_noise_estimate"],
                "measurement_noise_candidates",
            )
        )

        search_results = []
        best_result = None

        for process_noise, measurement_noise in itertools.product(process_grid, measurement_grid):
            rmse_values = []
            for traj, measurement_mask in zip(traj_list, masks):
                ekf = cls(
                    obs_dim=traj.shape[1],
                    dt=dt,
                    env_id=env_id,
                    process_noise=process_noise,
                    measurement_noise=measurement_noise,
                    state_scale=stats["state_scale"],
                    velocity_scale=stats["velocity_scale"],
                    position_process_scale=stats["position_process_scale"],
                    velocity_process_scale=stats["velocity_process_scale"],
                )
                estimates = ekf.filter_sequence(traj, measurement_mask=measurement_mask)
                eval_mask = ~measurement_mask
                if not np.any(eval_mask):
                    eval_mask = np.ones(len(traj), dtype=bool)
                error = traj[eval_mask] - estimates[eval_mask]
                rmse_values.append(float(np.sqrt(np.mean(error ** 2))))

            result = {
                "process_noise": float(process_noise),
                "measurement_noise": float(measurement_noise),
                "rmse": float(np.mean(rmse_values)),
            }
            search_results.append(result)
            if best_result is None or result["rmse"] < best_result["rmse"]:
                best_result = result

        return {
            "obs_dim": stats["obs_dim"],
            "dt": dt,
            "env_id": env_id,
            "process_noise": best_result["process_noise"],
            "measurement_noise": best_result["measurement_noise"],
            "state_scale": stats["state_scale"].tolist(),
            "velocity_scale": stats["velocity_scale"].tolist(),
            "position_process_scale": stats["position_process_scale"],
            "velocity_process_scale": stats["velocity_process_scale"],
            "heuristics": {
                "process_noise_estimate": float(stats["process_noise_estimate"]),
                "measurement_noise_estimate": float(stats["measurement_noise_estimate"]),
            },
            "grid_search": search_results,
            "best_rmse": best_result["rmse"],
        }

    def auto_tune(
        self,
        trajectories,
        measurement_masks=None,
        dropout_masks=None,
        process_noise_candidates=None,
        measurement_noise_candidates=None,
    ):
        """
        Tune the current EKF in place from trajectory data.

        Returns the full calibration payload so callers can inspect the search
        results or recreate the estimator later with `from_calibration`.
        """
        calibration = self.auto_calibrate(
            trajectories,
            dt=self.dt,
            env_id=self.env_id,
            measurement_masks=measurement_masks,
            dropout_masks=dropout_masks,
            process_noise_candidates=process_noise_candidates,
            measurement_noise_candidates=measurement_noise_candidates,
        )
        if calibration["obs_dim"] != self.obs_dim:
            raise ValueError(
                f"Calibration obs_dim {calibration['obs_dim']} does not match EKF obs_dim {self.obs_dim}"
            )

        self.env_id = calibration.get("env_id", self.env_id)
        self.env_prior = self._get_env_prior(self.env_id)
        self.dt = calibration["dt"]
        self.position_process_scale = calibration["position_process_scale"]
        self.velocity_process_scale = calibration["velocity_process_scale"]
        self.state_scale = self._sanitize_scale(calibration["state_scale"], self.obs_dim)
        self.velocity_scale = self._sanitize_scale(
            calibration["velocity_scale"], self.obs_dim
        )
        (
            self.state_process_profile,
            self.velocity_process_profile,
            self.measurement_noise_profile,
        ) = self._get_noise_profiles(self.env_id, self.obs_dim)
        self.configure_noise(
            calibration["process_noise"],
            calibration["measurement_noise"],
        )
        return calibration

    def configure_noise(
        self,
        process_noise=None,
        measurement_noise=None,
        state_scale=None,
        velocity_scale=None,
    ):
        if state_scale is not None:
            self.state_scale = self._sanitize_scale(state_scale, self.obs_dim)
        if velocity_scale is not None:
            self.velocity_scale = self._sanitize_scale(velocity_scale, self.obs_dim)

        if process_noise is not None:
            self.process_noise = process_noise
        if measurement_noise is not None:
            self.measurement_noise = measurement_noise

        process_diag = np.concatenate(
            [
                np.square(self.state_scale)
                * self.position_process_scale
                * self.state_process_profile,
                np.square(self.velocity_scale)
                * self.velocity_process_scale
                * self.velocity_process_profile,
            ]
        )
        measurement_diag = self.measurement_noise_profile.copy()

        self.Q = self._to_diagonal_matrix(self.process_noise, self.state_dim, process_diag)
        self.R = self._to_diagonal_matrix(
            self.measurement_noise, self.obs_dim, measurement_diag
        )

    def reset(self, obs, velocity=None):
        obs = np.asarray(obs, dtype=float)
        self.x = np.zeros(self.state_dim)
        self.x[: self.obs_dim] = obs
        if velocity is not None:
            self.x[self.obs_dim :] = np.asarray(velocity, dtype=float)
        self.P = np.eye(self.state_dim) * 1.0

    def predict(self):
        """Prediction step for a constant-velocity motion model."""
        F = np.eye(self.state_dim)
        F[: self.obs_dim, self.obs_dim :] = np.eye(self.obs_dim) * self.dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, obs):
        """Measurement update when an observation is available."""
        obs = np.asarray(obs, dtype=float)
        y = obs - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

    def filter_sequence(self, observations, measurement_mask=None):
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 2:
            raise ValueError("observations must have shape [T, D]")

        if measurement_mask is None:
            measurement_mask = np.ones(len(observations), dtype=bool)
        else:
            measurement_mask = np.asarray(measurement_mask, dtype=bool)
            if measurement_mask.shape != (len(observations),):
                raise ValueError(
                    f"measurement_mask shape {measurement_mask.shape} does not match sequence length {len(observations)}"
                )
        measurement_mask = measurement_mask.copy()
        measurement_mask[0] = True

        self.reset(observations[0])
        estimates = [self.get_obs_estimate().copy()]

        for t in range(1, len(observations)):
            self.predict()
            if measurement_mask[t]:
                self.update(observations[t])
            estimates.append(self.get_obs_estimate().copy())
        return np.asarray(estimates)

    def get_obs_estimate(self):
        return self.x[: self.obs_dim]

    def get_velocity_estimate(self):
        return self.x[self.obs_dim :]
