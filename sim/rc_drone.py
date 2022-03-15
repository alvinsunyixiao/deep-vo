import airsim
import argparse
import datetime
import threading
import time
import os

import numpy as np

from utils.ps4_controller import PS4Controller
from sim.randomize import randomize_time, randomize_weather, clear_weather

CLIENT_LOCK = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def create_trajectory_filepath(base_dir):
    filename = time.strftime("%y-%m-%d_%H-%M-%S.pkl")
    return os.path.join(base_dir, filename)

if __name__ == "__main__":
    args = parse_args()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.simEnableWeather(True)

    controller = PS4Controller()
    controller.register_button_callback(controller.O,
        lambda: randomize_weather(client, client_lock=CLIENT_LOCK))
    controller.register_button_callback(controller.SQUARE,
        lambda: clear_weather(client, client_lock=CLIENT_LOCK))
    controller.register_button_callback(controller.TRIANGLE,
        lambda: randomize_time(client, client_lock=CLIENT_LOCK))

    with controller:
        while not controller.buttons[controller.X]:
            with CLIENT_LOCK:
                kinematics = client.simGetGroundTruthKinematics()
                client.moveByRollPitchYawrateThrottleAsync(
                    roll=controller.axes[controller.RIGHT_LR] * .6,
                    pitch=-controller.axes[controller.RIGHT_UD] * .6,
                    yaw_rate=-controller.axes[controller.LEFT_LR] * 2.,
                    throttle=(-controller.axes[controller.LEFT_UD] + 1) / 2,
                    duration=0.2,
                )
            time.sleep(0.01)

