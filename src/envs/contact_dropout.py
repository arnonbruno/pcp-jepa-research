import numpy as np
import gymnasium as gym


class ContactDropoutEnv:
    """
    Wrapper that triggers sensor dropout from MuJoCo contact forces.

    Existing call sites still pass `velocity_threshold`; this is now interpreted
    as a relative sensitivity multiplier around an environment-specific contact
    force threshold instead of an observation-delta cutoff.
    """

    ENV_CONTACT_FORCE_THRESHOLDS = {
        "Hopper-v4": 25.0,
        "Walker2d-v4": 50.0,
        "HalfCheetah-v4": 30.0,
    }

    def __init__(
        self,
        env_id='Hopper-v4',
        dropout_duration=5,
        velocity_threshold=0.1,
        contact_force_threshold=None,
        warmup_steps=10,
    ):
        self.env = gym.make(env_id)
        self.env_id = env_id
        self.dropout_duration = dropout_duration
        self.velocity_threshold = velocity_threshold
        self.warmup_steps = warmup_steps
        self.obs_prev = None
        self.frozen_obs = None
        self.dropout_countdown = 0
        self.step_count = 0
        self.force_history = []
        self.contact_force_threshold = self._resolve_contact_force_threshold(
            contact_force_threshold
        )
        self._contact_source = self._detect_contact_source()

    def _resolve_contact_force_threshold(self, contact_force_threshold):
        if contact_force_threshold is not None:
            return float(contact_force_threshold)

        base_threshold = self.ENV_CONTACT_FORCE_THRESHOLDS.get(self.env_id)
        if base_threshold is None:
            base_threshold = 25.0

        relative_sensitivity = max(float(self.velocity_threshold), 1e-3) / 0.1
        return base_threshold * relative_sensitivity

    def _detect_contact_source(self):
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "sim") and hasattr(unwrapped.sim, "data"):
            data = unwrapped.sim.data
            if hasattr(data, "cfrc_ext"):
                return "sim.data.cfrc_ext"
        if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "cfrc_ext"):
            return "data.cfrc_ext"
        return None

    def _extract_contact_forces(self):
        unwrapped = self.env.unwrapped
        data = None
        if hasattr(unwrapped, "sim") and hasattr(unwrapped.sim, "data"):
            data = unwrapped.sim.data
        elif hasattr(unwrapped, "data"):
            data = unwrapped.data

        if data is None or not hasattr(data, "cfrc_ext"):
            return {
                "contact_force_total": 0.0,
                "contact_force_max": 0.0,
                "contact_force_mean": 0.0,
                "contact_body_count": 0,
                "physics_contact": False,
                "ncon": 0,
                "mujoco_available": False,
            }

        cfrc_ext = np.asarray(data.cfrc_ext, dtype=float)
        if cfrc_ext.ndim != 2 or cfrc_ext.shape[1] < 6:
            return {
                "contact_force_total": 0.0,
                "contact_force_max": 0.0,
                "contact_force_mean": 0.0,
                "contact_body_count": 0,
                "physics_contact": False,
                "ncon": int(getattr(data, "ncon", 0)),
                "mujoco_available": False,
            }

        # MuJoCo cfrc_ext is [torque_x, torque_y, torque_z, force_x, force_y, force_z]
        linear_force_norms = np.linalg.norm(cfrc_ext[:, 3:], axis=1)
        contact_force_total = float(np.sum(linear_force_norms)) if linear_force_norms.size else 0.0
        contact_force_max = float(np.max(linear_force_norms)) if linear_force_norms.size else 0.0
        contact_force_mean = (
            float(np.mean(linear_force_norms)) if linear_force_norms.size else 0.0
        )
        contact_body_count = int(np.sum(linear_force_norms > self.contact_force_threshold))

        return {
            "contact_force_total": contact_force_total,
            "contact_force_max": contact_force_max,
            "contact_force_mean": contact_force_mean,
            "contact_body_count": contact_body_count,
            "physics_contact": contact_force_total >= self.contact_force_threshold,
            "ncon": int(getattr(data, "ncon", 0)),
            "mujoco_available": True,
        }

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.obs_prev = obs.copy()
        self.frozen_obs = obs.copy()
        self.dropout_countdown = 0
        self.step_count = 0
        self.force_history = []
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.step_count += 1
        contact_info = self._extract_contact_forces()
        self.force_history.append(contact_info["contact_force_total"])

        if contact_info["mujoco_available"]:
            contact_detected = contact_info["physics_contact"]
        else:
            contact_detected = False
            if self.obs_prev is not None:
                delta = np.abs(obs - self.obs_prev).max()
                if delta > self.velocity_threshold:
                    contact_detected = True

        if self.dropout_countdown == 0:
            physics_contact = (
                self.step_count > self.warmup_steps and contact_detected
            )
            if physics_contact:
                self.dropout_countdown = self.dropout_duration
                self.frozen_obs = obs.copy()

        info['true_obs'] = obs.copy()
        info['dropout_active'] = self.dropout_countdown > 0
        info['dropout_step'] = self.dropout_duration - self.dropout_countdown
        info['frozen_obs'] = self.frozen_obs.copy()
        info['contact_detected'] = contact_detected
        info['contact_force_total'] = contact_info["contact_force_total"]
        info['contact_force_max'] = contact_info["contact_force_max"]
        info['contact_force_mean'] = contact_info["contact_force_mean"]
        info['contact_body_count'] = contact_info["contact_body_count"]
        info['contact_force_threshold'] = self.contact_force_threshold
        info['contact_source'] = self._contact_source
        info['ncon'] = contact_info["ncon"]

        if self.dropout_countdown > 0:
            obs_return = self.frozen_obs.copy()
            self.dropout_countdown -= 1
        else:
            obs_return = obs.copy()
            self.obs_prev = obs.copy()

        return obs_return, reward, term, trunc, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()


# Alias for compatibility with older code
CriticalDropoutEnv = ContactDropoutEnv
