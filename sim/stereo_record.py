import airsim
import json
import logging
import os
import queue
import shutil
import threading
import time

import numpy as np
import typing as T

from utils.airsim_utils import airsim_to_pose3d
from utils.ps4_controller import PS4Controller

class StereoRecorder:
    def __init__(self):
        # client for control loop
        self.ctrl_cli = airsim.MultirotorClient()
        self.ctrl_cli.confirmConnection()
        self.ctrl_cli.enableApiControl(True)
        self.ctrl_cli.armDisarm(True)
        self.ctrl_thread = threading.Thread(target=self._ctrl_loop)

        # client for image aquisition
        self.img_cli = airsim.MultirotorClient()
        self.img_cli.confirmConnection()

        # exit flag
        self.exit_evt = threading.Event()

    def _exit_callback(self) -> None:
        self.exit_evt.set()

    def _start_stop_callback(self) -> None:
        if self.img_cli.isRecording():
            self.img_cli.stopRecording()
            logging.info("Stop recording")
        else:
            self.img_cli.startRecording()
            logging.info("Start recording")

    def _ctrl_loop(self) -> None:
        with PS4Controller() as controller:
            # callbacks
            controller.register_button_callback(controller.X, self._exit_callback)
            controller.register_button_callback(controller.O, self._start_stop_callback)

            # control loop
            while not self.exit_evt.wait(0.01):
                self.ctrl_cli.moveByRollPitchYawrateThrottleAsync(
                    roll=controller.axes[controller.RIGHT_LR] * 1.,
                    pitch=-controller.axes[controller.RIGHT_UD] * 1.,
                    yaw_rate=-controller.axes[controller.LEFT_LR] * 2.,
                    throttle=(-controller.axes[controller.LEFT_UD] + 1) / 2,
                    duration=0.02,
                )

    def _datum_from_airsim(self, res: airsim.ImageResponse, img_dir: str, prefix: str):
        # get image data
        img_path = os.path.join(img_dir, f"{prefix}_{res.time_stamp}.png")
        with open(img_path, "wb") as f:
            f.write(res.image_data_uint8)

        # get pose data
        world_T_body = airsim.Pose(res.camera_position, res.camera_orientation)

        return {
            "timestamp": res.time_stamp,
            "img_path": img_path,
            "world_T_body": world_T_body,
        }

    def run(self) -> None:
        self.ctrl_thread.start()
        self.ctrl_thread.join()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    StereoRecorder().run()
