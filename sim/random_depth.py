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
    parser.add_argument("--min-distance", type=float, default=2.0,
                        help="avoid poses with surrounding obstacles closer than this distance")
    parser.add_argument("--shard-size", type=int, default=2000,
                        help="number of samples per TFRecord shard")
    parser.add_argument("--randomize", action="store_true",
                        help="add this flag to perform wheather randomization")
    parser.add_argument("--viz", action="store_true",
                        help="briefly pause between frames to visualize")
    parser.add_argument("--shard-index-start", type=int, default=0,
                        help="starting index of the shards")
    return parser.parse_args()


if __name__ == "__main__":
    set_tf_memory_growth(True)

    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    client = airsim.VehicleClient()
    client.confirmConnection()
    client.simEnableWeather(True)

    pose_gen = RandomPose3DGen(
        min_pos=(-150., -150., -20.0),
        max_pos=(150., 150., -2.),
        min_rp=(-np.pi / 4, -np.pi / 4),
        max_rp=(np.pi / 4, np.pi / 4),
    )

    writer = ShardedTFRecordWriter(output_dir, args.shard_size, args.shard_index_start)
    with tqdm(total=args.num_samples) as pbar, writer:
        while pbar.n < args.num_samples:
            # randomize weather if specified
            if args.randomize:
                randomize_weather(client)

            # generate random pose
            base_pose = pose_gen()
            client.simSetVehiclePose(pose3d_to_airsim(base_pose), True)

            # check for surrounding distance
            responses = client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene),
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True),
            ])
            depth_data = [np.array(resp.image_data_float) for resp in responses[3:]]
            occlusion_level = [np.sum(d < args.min_distance) / d.shape[0] for d in depth_data]

            # skip if any of the three images have an occlusion level greater than 5%
            if any([o > 0.05 for o in occlusion_level]):
                continue

            # collect data
            datum = {}

            # save image
            datum["image"] = responses[0].image_data_uint8

            # save pose
            camera_pose = airsim.Pose(responses[0].camera_position,
                                      responses[0].camera_orientation)
            datum["pose"] = airsim_to_pose3d(camera_pose).to_storage()

            # save depth
            depth_mm = np.clip(np.asarray(responses[1].image_data_float) * 1e3, 0, 65535)
            depth_mm = np.reshape(depth_mm, (responses[1].height, responses[1].width, 1))
            datum["depth"] = tf.io.encode_png(depth_mm.astype(np.uint16))

            # write to TFRecord
            for key in datum:
                datum[key] = tensor_to_feature(datum[key])

            features = tf.train.Features(feature=datum)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

            pbar.update()

            # pause for visualization
            if args.viz:
                time.sleep(1.)

