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
        batch_size=8,
        min_depth=0.1,
        max_rotation=30.,
        max_translation=3.,
        epoch_size=500,
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
        inv_range_imgs = []
        color_imgs = []
        points_cam = []
        ref_T_cams = []

        for i, data_dict in enumerate(data_dicts):
            # per image data
            img = tf.image.decode_image(data_dict["image_png"], channels=3)
            depth = tf.reshape(data_dict["depth"], (img.shape[0], img.shape[1], 1))

            # convert planar depth to inverse range
            points_3d = cam.unproject(depth)
            range_img = tf.linalg.norm(points_3d, axis=-1, keepdims=True)

            inv_range_imgs.append(1. / range_img)
            color_imgs.append(img)
            points_cam.append(points_3d)
            ref_T_cams.append(world_T_ref.inv() @ Pose3D.from_storage(data_dict["pose"]))

        points_cam = tf.stack(points_cam)
        self.data_dict = {
            "inv_range_imgs": tf.stack(inv_range_imgs),
            "color_imgs": tf.stack(color_imgs),
            "points_cam": points_cam,
            "directions_cam": tf.linalg.normalize(points_cam, axis=-1)[0],
            "ref_T_cams": tf.stack(ref_T_cams),
        }
        self.dataset = tf.data.Dataset.from_tensors(self.data_dict)
        self.dataset = self.dataset.repeat(self.p.epoch_size)
        self.dataset = self.dataset.map(self.data_process)
        self.dataset = self.dataset.prefetch(10)

    def data_process(self, data_dict: T_DATA_DICT) -> T_DATA_DICT:
        # generate random sample indices
        img_indices_b = tf.random.uniform((self.p.batch_size,),
                                          maxval=self.p.num_images, dtype=tf.int32)
        cam_T_virtual_b = Pose3D.random(max_angle=math.radians(self.p.max_rotation),
                                        max_translate=3.,
                                        size=(self.p.batch_size,))
        ref_T_cam_b = tf.gather(data_dict["ref_T_cams"], img_indices_b)

        return {
            "inv_range_imgs_bhw1": tf.gather(data_dict["inv_range_imgs"], img_indices_b),
            "color_imgs_bhw3": tf.gather(data_dict["color_imgs"], img_indices_b),
            "points_cam_bhw3": tf.gather(data_dict["points_cam"], img_indices_b),
            "directions_cam_bhw3": tf.gather(data_dict["directions_cam"], img_indices_b),
            "img_indices_b": img_indices_b,
            "ref_T_cam_b": ref_T_cam_b,
            "cam_T_virtual_b": cam_T_virtual_b,
        }

