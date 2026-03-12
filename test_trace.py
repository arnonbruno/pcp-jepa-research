import numpy as np
from src.models.ekf import EKFEstimator
import faulthandler
faulthandler.enable()

traj = [np.random.randn(50, 111)]

EKFEstimator.auto_calibrate(traj, dt=0.002, env_id='Ant-v4')
