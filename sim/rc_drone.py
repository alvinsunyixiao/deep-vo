import airsim
import argparse
import datetime
import threading
import time
import os

import numpy as np

from utils.ps4_controller import PS4Controller

CLIENT_LOCK = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="output directory to store output")
    return parser.parse_args()

def create_trajectory_filepath(base_dir):
    filename = time.strftime("%y-%m-%d_%H-%M-%S.pkl")
    return os.path.join(base_dir, filename)

def randomize_time(client: airsim.VehicleClient):
    now = datetime.datetime.now()
    delta_h = np.random.uniform(-12, 12)
    delta_time = datetime.timedelta(hours=delta_h)
    random_time = now + delta_time
    print(f"Randomized time: {random_time}")
    with CLIENT_LOCK:
        client.simSetTimeOfDay(True, random_time.strftime("%Y-%m-%d %H:%M:%S"), update_interval_secs=5)

def randomize_weather(client: airsim.VehicleClient):
    with CLIENT_LOCK:
        client.simSetWeatherParameter(airsim.WeatherParameter.Rain,
                                      min(np.random.exponential(0.1), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.Roadwetness,
                                      min(np.random.exponential(0.1), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.Snow,
                                      min(np.random.exponential(0.1), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.RoadSnow,
                                      min(np.random.exponential(0.1), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf,
                                      min(np.random.exponential(0.1), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.RoadLeaf,
                                      min(np.random.exponential(0.1), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.Dust,
                                      min(np.random.exponential(0.1), 1))
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog,
                                      min(np.random.exponential(0.1), 1))

if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.simEnableWeather(True)

    controller = PS4Controller()
    controller.register_button_callback(controller.O, lambda: randomize_weather(client))
    controller.register_button_callback(controller.TRIANGLE, lambda: randomize_time(client))

    with controller:
        while not controller.buttons[controller.X]:
            with CLIENT_LOCK:
                kinematics = client.simGetGroundTruthKinematics()
                client.moveByRollPitchYawrateThrottleAsync(
                    roll=controller.axes[2] * .6,
                    pitch=-controller.axes[3] * .6,
                    yaw_rate=-controller.axes[0] * 2.,
                    throttle=(-controller.axes[1] + 1) / 2,
                    duration=0.2,
                )
            time.sleep(0.01)

