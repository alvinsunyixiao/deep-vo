import airsim
import argparse
import time
import os

import numpy as np
import tensorflow as tf
import typing as T
from collections import defaultdict
from skimage import io
from tqdm import tqdm

from utils.pose3d import RandomPose3DGen, Pose3D
from utils.tf_utils import set_tf_memory_growth, tensor_to_feature, ShardedTFRecordWriter
from sim.randomize import randomize_weather
from utils.airsim_utils import pose3d_to_airsim, airsim_to_pose3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="output directory to store output")
    parser.add_argument("-n", "--num-samples", type=int, default=10000,
                        help="number of random-pose samples to generate")
    parser.add_argument("--num-perturb", type=int, default=16,
                        help="number of perturbation per pose sample")
    parser.add_argument("--min-distance", type=float, default=2.0,
                        help="avoid poses with surrounding obstacles closer than this distance")
    parser.add_argument("--shard-size", type=int, default=400,
                        help="number of samples per TFRecord shard")
    parser.add_argument("--randomize", action="store_true",
                        help="add this flag to perform wheather randomization")
    parser.add_argument("--max-rotation", type=float, default=20.0,
                        help="maximum rotational perturbation in [deg]")
    parser.add_argument("--max-translation", type=float, default=0.5,
                        help="maximum translational perturbation in [m]")
    parser.add_argument("--viz", action="store_true",
                        help="briefly pause between frames to visualize")
    return parser.parse_args()


if __name__ == "__main__":
    set_tf_memory_growth(True)

    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.simEnableWeather(True)

    pose_gen = RandomPose3DGen(
        min_pos=(-150., -150., -20.0),
        max_pos=(150., 150., -2.),
        min_rp=(-np.pi / 4, -np.pi / 4),
        max_rp=(np.pi / 4, np.pi / 4),
    )

    writer = ShardedTFRecordWriter(output_dir, args.shard_size)
    with tqdm(total=args.num_samples) as pbar, writer:
        while pbar.n < args.num_samples:
            base_pose = pose_gen()
            client.simSetVehiclePose(pose3d_to_airsim(base_pose), True)

            # check for surrounding distance
            time.sleep(0.05) # wait for distance sensor update
            dists = [
                client.getDistanceSensorData("DistanceFront"),
                client.getDistanceSensorData("DistanceRight"),
                client.getDistanceSensorData("DistanceBack"),
                client.getDistanceSensorData("DistanceLeft"),
                client.getDistanceSensorData("DistanceUp"),
                client.getDistanceSensorData("DistanceDown"),
            ]
            if any([d.distance < args.min_distance for d in dists]):
                continue

            # generate some pose perturbation
            perturbs = Pose3D.random(
                np.deg2rad(args.max_rotation), args.max_translation, (args.num_perturb,))
            poses = base_pose[None] @ perturbs

            # randomize weather if specified
            if args.randomize:
                randomize_weather(client)

            # collect data
            datum = defaultdict(list)
            for i in range(args.num_perturb):
                client.simSetVehiclePose(pose3d_to_airsim(poses[i]), False)
                responses = client.simGetImages([
                    airsim.ImageRequest("front_center", airsim.ImageType.Scene),
                    airsim.ImageRequest("back_center", airsim.ImageType.Scene),
                    airsim.ImageRequest("bottom_center", airsim.ImageType.Scene),
                ])

                locations = ["front", "back", "bottom"]
                for i, location in enumerate(locations):
                    # save images
                    datum[f"{location}_images"].append(responses[i].image_data_uint8)

                    # save poses
                    camera_pose = airsim.Pose(responses[i].camera_position,
                                              responses[i].camera_orientation)
                    datum[f"{location}_poses"].append(airsim_to_pose3d(camera_pose).to_storage())
                if args.viz:
                    time.sleep(0.5)

            # write to TFRecord
            feature = {}
            for key in datum:
                feature[key] = tensor_to_feature(tf.stack(datum[key]))
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

            pbar.update()

