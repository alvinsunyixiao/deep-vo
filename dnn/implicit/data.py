import math
import pickle
import random
import typing as T

import numpy as np
import tensorflow as tf

from utils.camera import PinholeCam
from utils.pose3d import Pose3D, Rot3D
from utils.params import ParamDict

T_DATA_DICT = T.Dict[str, T.Union[tf.Tensor, Pose3D, Rot3D, PinholeCam]]

class PointLoader:

    DEFAULT_PARAMS = ParamDict(
        data_path="/data/airsim/implicit/scene1.pkl",
        num_images=1,
        cam=PinholeCam.from_size_and_fov(
            size_xy=np.array((256, 144)),
            fov_xy=np.array((89.903625, 58.633181)),
        ),
        img_size=(256, 144),
        batch_size=40000,
        max_depth=100.,
        min_depth=0.3,
        random_rotation_angle=30.,
        epoch_size=3000,
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS) -> None:
        self.p = params
        cam: PinholeCam = self.p.cam

        with open(self.p.data_path, "rb") as f:
            data_dicts = pickle.load(f)

        # truncate the number of images to use
        if self.p.num_images > 0:
            assert self.p.num_images <= len(data_dicts), \
                "Requiring more samples than available"
            data_dicts = data_dicts[:self.p.num_images]

        # pick the first image to be reference pose
        world_T_ref = Pose3D.from_storage(data_dicts[0]["pose"])

        # parse data
        depth_imgs = []
        color_imgs = []
        ref_T_cams = []

        points_cam = []
        colors = []
        img_indics = []
        for i, data_dict in enumerate(data_dicts):
            # per image data
            img = tf.image.decode_image(data_dict["image_png"], channels=3)
            depth = tf.reshape(data_dict["depth"], (img.shape[0], img.shape[1], 1))

            # convert planar depth to perspective depth
            depth_perspective = tf.linalg.norm(cam.unproject(depth), axis=-1, keepdims=True)

            depth_imgs.append(depth_perspective)
            color_imgs.append(img)
            ref_T_cams.append(world_T_ref.inv() @ Pose3D.from_storage(data_dict["pose"]))

            # use samples that live in valid depth range
            valid_indics_yx_k2 = tf.where((depth_perspective[..., 0] <= self.p.max_depth) & \
                                          (depth_perspective[..., 0] >= self.p.min_depth))

            grid_hw2 = cam.compute_grid(depth_perspective)
            grid_k2 = tf.gather_nd(grid_hw2, valid_indics_yx_k2)
            depth_k1 = tf.gather_nd(depth_perspective, valid_indics_yx_k2)
            points_cam.append(cam.unproject(depth_k1, grid_k2))
            colors.append(tf.gather_nd(img, valid_indics_yx_k2))
            img_indics.append(tf.ones(valid_indics_yx_k2.shape[0], dtype=tf.int32) * i)

        points_cam = tf.concat(points_cam, axis=0)
        self.data_dict = {
            "depth_imgs": tf.stack(depth_imgs),
            "color_imgs": tf.stack(color_imgs),
            "ref_T_cams": tf.stack(ref_T_cams),
            "points_cam": points_cam,
            "colors": tf.concat(colors, axis=0),
            "directions_cam": tf.linalg.normalize(points_cam, axis=-1)[0],
            "img_indics": tf.concat(img_indics, axis=0),
        }
        self.dataset = tf.data.Dataset.from_tensors(self.data_dict)
        self.dataset = self.dataset.repeat(self.p.epoch_size)
        self.dataset = self.dataset.map(self.data_process)
        self.dataset = self.dataset.prefetch(10)

    def data_process(self, data_dict: T_DATA_DICT) -> T_DATA_DICT:
        # generate random sample indices
        rand_idx_b = tf.random.uniform((self.p.batch_size,),
                                       maxval=data_dict["img_indics"].shape[0],
                                       dtype=tf.int32)

        points_cam_b3 = tf.gather(data_dict["points_cam"], rand_idx_b)
        colors_b3 = tf.gather(data_dict["colors"], rand_idx_b)
        directions_cam_b3 = tf.gather(data_dict["directions_cam"], rand_idx_b)
        img_idx_b = tf.gather(data_dict["img_indics"], rand_idx_b)
        ref_T_cam_b = tf.gather(data_dict["ref_T_cams"], img_idx_b)
        rand_img_idx = tf.random.uniform((), maxval=self.p.num_images, dtype=tf.int32)

        return {
            "points_cam_b3": points_cam_b3,
            "colors_b3": colors_b3,
            "directions_cam_b3": directions_cam_b3,
            "img_idx_b": img_idx_b,
            "ref_T_cam_b": ref_T_cam_b,
            "virtual_idx": rand_img_idx,
            "ref_T_virtual": tf.gather(data_dict["ref_T_cams"], rand_img_idx),
            "color_virtual_hw3": tf.gather(data_dict["color_imgs"], rand_img_idx),
            "depth_virtual_hw1": tf.gather(data_dict["depth_imgs"], rand_img_idx),
        }

    def generate_samples(self,
        points_b3: tf.Tensor,
        directions_b3: tf.Tensor
    ) -> T.Tuple[tf.Tensor, tf.Tensor]:
        R_rand_b = Rot3D.random(math.radians(self.p.random_rotation_angle),
                                size=(self.p.batch_size,))
        directions_pert_b3 = R_rand_b @ directions_b3

        depth_rand_b = tf.exp(tf.random.normal(
            shape=(self.p.batch_size,),
            mean=tf.math.log(tf.linalg.norm(points_b3, axis=-1)),
            stddev=0.3,
        ))

        positions_b3 = points_b3 - depth_rand_b[..., tf.newaxis] * directions_pert_b3

        return positions_b3, directions_pert_b3, depth_rand_b

