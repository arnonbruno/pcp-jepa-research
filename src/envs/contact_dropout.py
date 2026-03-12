import numpy as np
import gymnasium as gym

try:
    import mujoco
except ImportError:  # pragma: no cover - optional dependency at test time
    mujoco = None


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
        "Ant-v4": 40.0,
    }

    def __init__(
        self,
        env_id='Hopper-v4',
        dropout_duration=5,
        velocity_threshold=0.1,
        contact_force_threshold=None,
        warmup_steps=10,
    ):
        if env_id == 'Ant-v4':
            self.env = gym.make(env_id, use_contact_forces=True)
        else:
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
        model, data = self._get_model_and_data()
        if (
            mujoco is not None
            and model is not None
            and data is not None
            and hasattr(data, "contact")
            and hasattr(data, "ncon")
            and hasattr(mujoco, "mj_contactForce")
        ):
            return "mujoco.mj_contactForce"

        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "sim") and hasattr(unwrapped.sim, "data"):
            data = unwrapped.sim.data
            if hasattr(data, "cfrc_ext"):
                return "sim.data.cfrc_ext"
        if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "cfrc_ext"):
            return "data.cfrc_ext"
        return None

    def _get_model_and_data(self):
        unwrapped = self.env.unwrapped

        model = getattr(unwrapped, "model", None)
        data = getattr(unwrapped, "data", None)

        if hasattr(unwrapped, "sim"):
            if model is None:
                model = getattr(unwrapped.sim, "model", None)
            if data is None:
                data = getattr(unwrapped.sim, "data", None)

        return model, data

    def _geom_name(self, model, geom_id):
        if model is None or geom_id is None or int(geom_id) < 0:
            return None

        geom_id = int(geom_id)
        if hasattr(model, "geom_id2name"):
            try:
                return model.geom_id2name(geom_id)
            except Exception:
                return None

        if mujoco is not None and hasattr(mujoco, "mj_id2name") and hasattr(mujoco, "mjtObj"):
            try:
                return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            except Exception:
                return None

        return None

    def _body_id_from_geom(self, model, geom_id):
        if model is None or geom_id is None or int(geom_id) < 0:
            return None

        geom_id = int(geom_id)
        geom_bodyid = getattr(model, "geom_bodyid", None)
        if geom_bodyid is None:
            return None

        try:
            return int(geom_bodyid[geom_id])
        except Exception:
            return None

    def _extract_contact_wrenches(self, model, data):
        ncon = int(getattr(data, "ncon", 0))
        if (
            mujoco is None
            or model is None
            or not hasattr(data, "contact")
            or not hasattr(mujoco, "mj_contactForce")
            or ncon <= 0
        ):
            return None

        contact_pairs = []
        all_force_norms = []
        normal_forces = []
        tangent_forces = []
        contact_distances = []
        body_ids = set()

        for idx in range(ncon):
            contact = data.contact[idx]
            wrench = np.zeros(6, dtype=float)
            try:
                mujoco.mj_contactForce(model, data, idx, wrench)
            except Exception:
                continue

            geom1 = int(getattr(contact, "geom1", -1))
            geom2 = int(getattr(contact, "geom2", -1))
            dist = float(getattr(contact, "dist", 0.0))
            normal_force = float(max(wrench[0], 0.0))
            tangential_force = float(np.linalg.norm(wrench[1:3]))
            force_norm = float(np.linalg.norm(wrench[:3]))

            if force_norm <= 0.0 and dist > 1e-6:
                continue

            body1 = self._body_id_from_geom(model, geom1)
            body2 = self._body_id_from_geom(model, geom2)
            if body1 is not None:
                body_ids.add(body1)
            if body2 is not None:
                body_ids.add(body2)

            all_force_norms.append(force_norm)
            normal_forces.append(normal_force)
            tangent_forces.append(tangential_force)
            contact_distances.append(dist)
            contact_pairs.append(
                {
                    "geom1": geom1,
                    "geom2": geom2,
                    "geom1_name": self._geom_name(model, geom1),
                    "geom2_name": self._geom_name(model, geom2),
                    "dist": dist,
                    "normal_force": normal_force,
                    "tangent_force": tangential_force,
                    "force_norm": force_norm,
                }
            )

        if not contact_pairs:
            return None

        return {
            "contact_pairs": contact_pairs,
            "contact_force_total": float(np.sum(all_force_norms)),
            "contact_force_max": float(np.max(all_force_norms)),
            "contact_force_mean": float(np.mean(all_force_norms)),
            "contact_normal_force_max": float(np.max(normal_forces)),
            "contact_tangent_force_max": float(np.max(tangent_forces)),
            "contact_distance_min": float(np.min(contact_distances)),
            "contact_pair_count": len(contact_pairs),
            "contact_body_count": len(body_ids),
            "physics_contact": True,
            "ncon": ncon,
            "mujoco_available": True,
            "contact_source": "mujoco.mj_contactForce",
        }

    def _extract_contact_forces(self):
        model, data = self._get_model_and_data()

        if data is None:
            return {
                "contact_force_total": 0.0,
                "contact_force_max": 0.0,
                "contact_force_mean": 0.0,
                "contact_normal_force_max": 0.0,
                "contact_tangent_force_max": 0.0,
                "contact_distance_min": np.inf,
                "contact_pair_count": 0,
                "contact_body_count": 0,
                "physics_contact": False,
                "ncon": 0,
                "mujoco_available": False,
                "contact_pairs": [],
                "contact_source": None,
            }

        contact_wrenches = self._extract_contact_wrenches(model, data)
        if contact_wrenches is not None:
            return contact_wrenches

        if not hasattr(data, "cfrc_ext"):
            return {
                "contact_force_total": 0.0,
                "contact_force_max": 0.0,
                "contact_force_mean": 0.0,
                "contact_normal_force_max": 0.0,
                "contact_tangent_force_max": 0.0,
                "contact_distance_min": np.inf,
                "contact_pair_count": 0,
                "contact_body_count": 0,
                "physics_contact": False,
                "ncon": int(getattr(data, "ncon", 0)),
                "mujoco_available": False,
                "contact_pairs": [],
                "contact_source": None,
            }

        cfrc_ext = np.asarray(data.cfrc_ext, dtype=float)
        if cfrc_ext.ndim != 2 or cfrc_ext.shape[1] < 6:
            return {
                "contact_force_total": 0.0,
                "contact_force_max": 0.0,
                "contact_force_mean": 0.0,
                "contact_normal_force_max": 0.0,
                "contact_tangent_force_max": 0.0,
                "contact_distance_min": np.inf,
                "contact_pair_count": 0,
                "contact_body_count": 0,
                "physics_contact": False,
                "ncon": int(getattr(data, "ncon", 0)),
                "mujoco_available": False,
                "contact_pairs": [],
                "contact_source": None,
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
            "contact_normal_force_max": contact_force_max,
            "contact_tangent_force_max": 0.0,
            "contact_distance_min": np.inf,
            "contact_pair_count": int(getattr(data, "ncon", 0)),
            "contact_body_count": contact_body_count,
            "physics_contact": contact_force_total >= self.contact_force_threshold,
            "ncon": int(getattr(data, "ncon", 0)),
            "mujoco_available": True,
            "contact_pairs": [],
            "contact_source": self._contact_source,
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
        info['contact_normal_force_max'] = contact_info["contact_normal_force_max"]
        info['contact_tangent_force_max'] = contact_info["contact_tangent_force_max"]
        info['contact_distance_min'] = contact_info["contact_distance_min"]
        info['contact_pair_count'] = contact_info["contact_pair_count"]
        info['contact_body_count'] = contact_info["contact_body_count"]
        info['contact_force_threshold'] = self.contact_force_threshold
        info['contact_source'] = contact_info.get("contact_source", self._contact_source)
        info['ncon'] = contact_info["ncon"]
        info['contact_pairs'] = contact_info["contact_pairs"]

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
