import airsim
import argparse
import time
import os
import numpy as np

from utils.pose3d import RandomPose3DGen, Pose3D
from utils.airsim_utils import pose3d_to_airsim, airsim_to_pose3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="output directory to store output")
    parser.add_argument("--max-rotation", type=float, default=20.0,
                        help="maximum rotational perturbation in [deg]")
    parser.add_argument("--max-translation", type=float, default=1.,
                        help="maximum translational perturbation in [m]")
    return parser.parse_args()

def save_data(client: airsim.VehicleClient, fname: str):
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene),
        airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True),
    ])
    pose = airsim.Pose(responses[0].camera_position,
                       responses[0].camera_orientation)
    pose = airsim_to_pose3d(pose)

    img_file = fname + "_img.png"
    depth_file = fname + "_depth.npy"
    pose_file = fname + "_pose.npy"

    with open(img_file, "wb") as f:
        f.write(responses[0].image_data_uint8)

    np.save(depth_file, responses[1].image_data_float)
    np.save(pose_file, pose.to_storage().numpy())

if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    pose_gen = RandomPose3DGen(
        min_pos=(-150., -150., -20.0),
        max_pos=(150., 150., -2.),
        min_rp=(-np.pi / 4, -np.pi / 4),
        max_rp=(np.pi / 4, np.pi / 4),
    )

    cnt = 0

    while True:
        base_pose = pose_gen()
        client.simSetVehiclePose(pose3d_to_airsim(base_pose), True)
        key = input("[q: quit/s: skip]: ")
        if key == "q":
            break
        elif key == "s":
            continue

        save_data(client, os.path.join(output_dir, f"s{cnt}_1"))

        perturb = Pose3D.random(np.deg2rad(args.max_rotation), args.max_translation, ())
        pert_pose = base_pose @ perturb
        client.simSetVehiclePose(pose3d_to_airsim(pert_pose), True)
        time.sleep(0.1)
        save_data(client, os.path.join(output_dir, f"s{cnt}_2"))

        cnt += 1

