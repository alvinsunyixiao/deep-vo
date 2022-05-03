from __future__ import annotations

import math
import numpy as np
import typing as T
import tensorflow as tf
from tensorflow_graphics.rendering import camera

from utils.pose3d import Pose3D

class PinholeCam:
    def __init__(self,
        focal: T.Union[tf.Tensor, np.ndarray],
        center: T.Union[tf.Tensor, np.ndarray],
    ):
        tf.assert_equal(tf.shape(focal)[-1], 2, "focal must be size-4 vector(s)")
        tf.assert_equal(tf.shape(center)[-1], 2, "center must be size-4 vector(s)")
        tf.assert_equal(tf.shape(focal), tf.shape(center),
                "center and focal must be the same size")
        self.focal = tf.cast(focal, tf.float32)
        self.center = tf.cast(center, tf.float32)

    @property
    def shape(self) -> T.Tuple[int, ...]:
        return tf.shape(self.focal)[:-1]

    def to_matrix(self) -> tf.Tensor:
        return camera.perspective.matrix_from_intrinsics(self.focal, self.center)

    def __repr__(self) -> str:
        return f"PinholeCam(focal={self.focal}, center={self.center})"

    def __getitem__(self, key) -> PinholeCam:
        return PinholeCam(self.focal[key], self.center[key])

    def project(self, points_3d: tf.Tensor) -> tf.Tensor:
        return camera.perspective.project(points_3d, self.focal, self.center)

    def unproject(self, depth_bhw1: tf.Tensor) -> tf.Tensor:
        h = tf.shape(depth_bhw1)[-3]
        w = tf.shape(depth_bhw1)[-2]
        x_hw, y_hw = tf.meshgrid(tf.range(w), tf.range(h), indexing="xy")
        points_2d_hw2 = tf.stack([x_hw, y_hw], axis=-1)
        return camera.perspective.unproject(points_2d_hw2, depth_bhw1, self.focal, self.center)

    def reproject(self,
        depth_tgt_bhw1: tf.Tensor,
        src_T_tgt_b: Pose3D,
    ) -> T.Tuple[tf.Tensor, tf.Tensor]:
        p_tgt_bhw3 = self.unproject_depth(depth_tgt_bhw1)
        p_src_bhw3 = src_T_tgt_b[:, None, None] @ p_tgt_bhw3
        pixel_src_bhw2 = self.project(p_src_bhw3)

        return pixel_src_bhw2, p_src_bhw3[..., -1, None]

    @classmethod
    def from_size_and_fov(cls,
        size_xy: T.Union[tf.Tensor, np.ndarray],
        fov_xy: T.Union[tf.Tensor, np.ndarray],
        use_degree: bool = True,
    ) -> PinholeCam:
        if use_degree:
            fov_xy = fov_xy / 180. * math.pi

        center = size_xy / 2.
        focal = center / tf.math.tan(fov_xy / 2)

        return PinholeCam(center, focal)
