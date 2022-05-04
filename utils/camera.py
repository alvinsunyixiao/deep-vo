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

    def project(self, points_3d_bhw3: tf.Tensor) -> tf.Tensor:
        bhw = tf.shape(points_3d_bhw3)[:-1]
        bhw2 = tf.concat([bhw, [2]], axis=0)
        return camera.perspective.project(
            point_3d=points_3d_bhw3,
            focal=tf.broadcast_to(self.focal, bhw2),
            principal_point=tf.broadcast_to(self.center, bhw2),
        )

    def unproject(self, depth_bhw1: tf.Tensor) -> tf.Tensor:
        h = tf.shape(depth_bhw1)[-3]
        w = tf.shape(depth_bhw1)[-2]
        x_hw, y_hw = tf.meshgrid(tf.range(w, dtype=tf.float32),
                                 tf.range(h, dtype=tf.float32),
                                 indexing="xy")
        points_2d_hw2 = tf.stack([x_hw, y_hw], axis=-1)
        bhw = tf.shape(depth_bhw1)[:-1]
        bhw2 = tf.concat([bhw, [2]], axis=0)
        return camera.perspective.unproject(
            point_2d=tf.broadcast_to(points_2d_hw2, bhw2),
            depth=depth_bhw1,
            focal=tf.broadcast_to(self.focal, bhw2),
            principal_point=tf.broadcast_to(self.center, bhw2),
        )

    def reproject(self,
        depth_tgt_bhw1: tf.Tensor,
        src_T_tgt_b: Pose3D,
    ) -> T.Tuple[tf.Tensor, tf.Tensor]:
        bhw = tf.shape(depth_tgt_bhw1)[:-1]
        p_tgt_bhw3 = self.unproject(depth_tgt_bhw1)
        p_src_bhw3 = src_T_tgt_b[:, tf.newaxis, tf.newaxis].broadcast_to(bhw) @ p_tgt_bhw3
        pixel_src_bhw2 = self.project(p_src_bhw3)

        return pixel_src_bhw2, p_src_bhw3[..., -1, tf.newaxis]

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

        return PinholeCam(focal, center)
