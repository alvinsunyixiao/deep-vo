import airsim
import argparse
import time
import numpy as np
import pickle

from tqdm import trange

from utils.pose3d import RandomPose3DGen, Pose3D
from utils.airsim_utils import pose3d_to_airsim, airsim_to_pose3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output path to store output")
    parser.add_argument("--max-rotation", type=float, default=30.0,
                        help="maximum rotational perturbation in [deg]")
    parser.add_argument("--max-translation", type=float, default=3.,
                        help="maximum translational perturbation in [m]")
    parser.add_argument("--num-perturb", type=int, default=16,
                        help="number of perturbation per pose sample")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    pose_gen = RandomPose3DGen(
        min_pos=(-150., -150., -20.0),
        max_pos=(150., 150., -2.),
        min_rp=(-np.pi / 4, -np.pi / 4),
        max_rp=(np.pi / 4, np.pi / 4),
    )

    while True:
        base_pose = pose_gen()
        client.simSetVehiclePose(pose3d_to_airsim(base_pose), True)
        key = input("[q: quit/s: skip]: ")
        if key == "q":
            break
        elif key == "s":
            continue

        data = []
        for i in trange(args.num_perturb):
            perturb = Pose3D.random(np.deg2rad(args.max_rotation), args.max_translation, ())
            pert_pose = base_pose @ perturb
            client.simSetVehiclePose(pose3d_to_airsim(pert_pose), True)
            time.sleep(0.1)
            responses = client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene),
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True),
            ])
            pose = airsim.Pose(responses[0].camera_position,
                               responses[0].camera_orientation)
            pose = airsim_to_pose3d(pose)

            data.append({
                "image_png": responses[0].image_data_uint8,
                "depth": responses[0].image_data_float,
                "pose": pose.to_storage().numpy(),
            })

        with open(args.output, "wb") as f:
            pickle.dump(data, f)
            print(f"Data saved to {args.output}")
