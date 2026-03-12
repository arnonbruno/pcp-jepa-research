import numpy as np
from src.models.ekf import EKFEstimator
import traceback
import sys
import threading
import os

def print_trace():
    print("Traceback after 5s:")
    for thread_id, frame in sys._current_frames().items():
        print(f"Thread {thread_id}:")
        traceback.print_stack(frame)
    os._exit(1)

threading.Timer(5.0, print_trace).start()

traj = [np.random.randn(50, 111)]
EKFEstimator.auto_calibrate(traj, dt=0.002, env_id='Ant-v4')
