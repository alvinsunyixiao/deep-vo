import airsim
import argparse
import datetime
import time
import os

import numpy as np

from utils.ps4_controller import PS4Controller
from sim.randomize import randomize_time, randomize_weather, clear_weather

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

    client_env = airsim.MultirotorClient()
    client_env.confirmConnection()
    client_env.simEnableWeather(True)

    with PS4Controller() as controller:
        # register event callback
        controller.register_button_callback(controller.O,
            lambda: randomize_weather(client_env))
        controller.register_button_callback(controller.SQUARE,
            lambda: clear_weather(client_env))
        controller.register_button_callback(controller.TRIANGLE,
            lambda: randomize_time(client_env))

        # main control loop
        while not controller.buttons[controller.X]:
            client.moveByRollPitchYawrateThrottleAsync(
                roll=controller.axes[controller.RIGHT_LR] * 1.,
                pitch=-controller.axes[controller.RIGHT_UD] * 1.,
                yaw_rate=-controller.axes[controller.LEFT_LR] * 2.,
                throttle=(-controller.axes[controller.LEFT_UD] + 1) / 2,
                duration=0.2,
            )
            time.sleep(0.01)

