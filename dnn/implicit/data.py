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
        num_images=12,
        cam=PinholeCam.from_size_and_fov(
            size_xy=np.array((256, 144)),
            fov_xy=np.array((89.903625, 58.633181)),
        ),
        img_size=(256, 144),
        batch_size=65536,
        min_depth=0.2,
        perturb_scale=2.0,
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

        # pick the first image to be reference pose
        world_T_ref = Pose3D.from_storage(data_dicts[0]["pose"])

        # transform from NED to EDN coordinate convention
        ned_T_edn = Pose3D(
            R = Rot3D.from_matrix([
                [0., 0., 1.],
                [1., 0., 0.],
                [0., 1., 0.],
            ]),
            t = tf.zeros(3),
        )

        # parse data
        range_imgs = []
        color_imgs = []
        ref_T_cams = []

        points_cam = []
        ranges = []
        colors = []
        img_indics = []
        for i, data_dict in enumerate(data_dicts):
            # per image data
            img = tf.image.decode_image(data_dict["image_png"], channels=3)
            depth = tf.reshape(data_dict["depth"], (img.shape[0], img.shape[1], 1))

            # convert planar depth to perspective depth
            p_cam = cam.unproject(depth)
            range_img = tf.linalg.norm(p_cam, axis=-1, keepdims=True)

            range_imgs.append(range_img)
            color_imgs.append(img)
            # transform from NED to EDN
            ref_T_cams.append(ned_T_edn.inv() @ \
                              world_T_ref.inv() @ Pose3D.from_storage(data_dict["pose"]) @ \
                              ned_T_edn)

            # use samples that live in valid depth range
            valid_indics_yx_k2 = tf.where(depth[..., 0] >= self.p.min_depth)

            grid_hw2 = cam.compute_grid(range_img)
            grid_k2 = tf.gather_nd(grid_hw2, valid_indics_yx_k2)
            depth_k1 = tf.gather_nd(depth, valid_indics_yx_k2)
            points_cam.append(tf.gather_nd(p_cam, valid_indics_yx_k2))
            ranges.append(tf.gather_nd(range_img, valid_indics_yx_k2))
            colors.append(tf.gather_nd(img, valid_indics_yx_k2))
            img_indics.append(tf.ones(valid_indics_yx_k2.shape[0], dtype=tf.int64) * i)

        points_cam = tf.concat(points_cam, axis=0)
        self.data_dict = {
            "range_imgs": tf.stack(range_imgs),
            "color_imgs": tf.stack(color_imgs),
            "ref_T_cams": tf.stack(ref_T_cams),
            "points_cam": points_cam,
            "ranges": tf.concat(ranges, axis=0),
            "colors": tf.concat(colors, axis=0),
            "directions_cam": tf.linalg.normalize(points_cam, axis=-1)[0],
            "img_indics": tf.concat(img_indics, axis=0),
        }
        self.dataset = tf.data.Dataset.from_tensors(self.data_dict)
        self.dataset = self.dataset.repeat(self.p.epoch_size)
        self.dataset = self.dataset.map(self.data_process,
                                        num_parallel_calls=4, deterministic=False)
        self.dataset = self.dataset.prefetch(10)

    @tf.function(jit_compile=True)
    def data_process(self, data_dict: T_DATA_DICT) -> T_DATA_DICT:
        # generate random sample indices
        rand_idx_b = tf.random.uniform((self.p.batch_size,),
                                       maxval=data_dict["img_indics"].shape[0],
                                       dtype=tf.int64)
        img_idx_b = tf.gather(data_dict["img_indics"], rand_idx_b)
        ref_T_cam_b = tf.gather(data_dict["ref_T_cams"], img_idx_b)
        rand_img_idx = tf.random.uniform((), maxval=self.p.num_images, dtype=tf.int64)

        return {
            "points_cam_b3": tf.gather(data_dict["points_cam"], rand_idx_b),
            "colors_b3": tf.gather(data_dict["colors"], rand_idx_b),
            "directions_cam_b3": tf.gather(data_dict["directions_cam"], rand_idx_b),
            "ranges_b1": tf.gather(data_dict["ranges"], rand_idx_b),
            "img_idx_b": img_idx_b,
            "ref_T_cam_b": ref_T_cam_b,
            "virtual_idx": rand_img_idx,
            "ref_T_virtual": tf.gather(data_dict["ref_T_cams"], rand_img_idx),
            "color_virtual_hw3": tf.gather(data_dict["color_imgs"], rand_img_idx),
            "range_virtual_hw1": tf.gather(data_dict["range_imgs"], rand_img_idx),
        }

    def generate_samples(self, points_cam_b3: tf.Tensor) -> T.Tuple[tf.Tensor, ...]:
        random_points_b3 = tf.random.normal((self.p.batch_size, 3), stddev=self.p.perturb_scale)
        rand_t_point_b3 = points_cam_b3 - random_points_b3
        directions_b3, range_b1 = tf.linalg.normalize(rand_t_point_b3, axis=-1)
        range_b = range_b1[..., 0]

        return random_points_b3, directions_b3, range_b

