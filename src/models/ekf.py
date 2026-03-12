import numpy as np

class EKFEstimator:
    """
    Extended Kalman Filter for velocity estimation under dropout.

    State: [obs]
    Model: x_{t+1} = x_t + v_t * dt  (constant velocity assumption)
    Measurement: x_t (when available)
    """
    def __init__(self, obs_dim=11, dt=None, process_noise=1.0, measurement_noise=0.01):
        self.obs_dim = obs_dim
        self.dt = dt if dt is not None else 0.002
        # State = [position, velocity] in obs_dim
        self.state_dim = obs_dim * 2

        # State estimate: [obs, velocity]
        self.x = np.zeros(self.state_dim)
        # Covariance
        self.P = np.eye(self.state_dim) * 1.0
        # Process noise
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[:obs_dim, :obs_dim] *= 0.01  # position evolves slowly
        self.Q[obs_dim:, obs_dim:] *= 1.0   # velocity uncertain
        # Measurement noise
        self.R = np.eye(obs_dim) * measurement_noise
        # Measurement matrix: we observe position only
        self.H = np.zeros((obs_dim, self.state_dim))
        self.H[:obs_dim, :obs_dim] = np.eye(obs_dim)

    def reset(self, obs):
        self.x = np.zeros(self.state_dim)
        self.x[:self.obs_dim] = obs
        self.x[self.obs_dim:] = 0.0  # zero initial velocity
        self.P = np.eye(self.state_dim) * 1.0

    def predict(self):
        """Prediction step (constant velocity model)."""
        # State transition: [x, v] -> [x + v*dt, v]
        F = np.eye(self.state_dim)
        F[:self.obs_dim, self.obs_dim:] = np.eye(self.obs_dim) * self.dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, obs):
        """Measurement update (when observation is available)."""
        y = obs - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

    def get_obs_estimate(self):
        return self.x[:self.obs_dim]

    def get_velocity_estimate(self):
        return self.x[self.obs_dim:]
