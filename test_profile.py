import numpy as np
from src.models.ekf import EKFEstimator
import time
import cProfile
import pstats

traj = [np.random.randn(50, 111)]

def run():
    EKFEstimator.auto_calibrate(traj, dt=0.002, env_id='Ant-v4')

pr = cProfile.Profile()
pr.enable()
run()
pr.disable()
ps = pstats.Stats(pr).sort_stats('tottime')
ps.print_stats(30)
