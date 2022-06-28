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
        dtype: tf.DType = tf.float32,
    ):
        tf.assert_equal(tf.shape(focal)[-1], 2, "focal must be size-4 vector(s)")
        tf.assert_equal(tf.shape(center)[-1], 2, "center must be size-4 vector(s)")
        tf.assert_equal(tf.shape(focal), tf.shape(center),
                "center and focal must be the same size")
        self.focal = tf.cast(focal, dtype)
        self.center = tf.cast(center, dtype)
        self.dtype = dtype

    @property
    def shape(self) -> tf.TensorShape:
        return self.focal.get_shape()[:-1]

    @property
    def fx(self) -> tf.Tensor:
        return self.focal[..., 0]

    @property
    def fy(self) -> tf.Tensor:
        return self.focal[..., 1]

    @property
    def cx(self) -> tf.Tensor:
        return self.center[..., 0]

    @property
    def cy(self) -> tf.Tensor:
        return self.center[..., 1]

    def to_matrix(self) -> tf.Tensor:
        return camera.perspective.matrix_from_intrinsics(self.focal, self.center)

    def to_storage(self) -> tf.Tensor:
        return tf.concat([self.focal, self.center], axis=-1)

    @classmethod
    def from_storage(cls, data: tf.Tensor) -> PinholeCam:
        tf.assert_equal(tf.shape(data)[-1], 4, "PinholeCam has a storage dimension of 4")
        return cls(
            focal=data[..., :2],
            center=data[..., 2:],
        )

    def __repr__(self) -> str:
        return f"PinholeCam(focal={self.focal}, center={self.center})"

    def __getitem__(self, key) -> PinholeCam:
        return PinholeCam(self.focal[key], self.center[key])

    def project_with_jac(self, points_3d: tf.Tensor) -> T.Tuple[tf.Tensor, tf.Tensor]:
        shp = tf.shape(points_3d)[:-1]
        shp2 = tf.concat([shp, [2]], axis=0)

        # projection
        pixels_2d = camera.perspective.project(
            point_3d=points_3d,
            focal=tf.broadcast_to(self.focal, shp2),
            principal_point=tf.broadcast_to(self.center, shp2),
        )

        # jacobian
        fx = tf.broadcast_to(self.fx, shp)
        fy = tf.broadcast_to(self.fy, shp)
        m00 = fx / points_3d[..., 2]
        m01 = tf.zeros_like(m00)
        m02 = -points_3d[..., 0] * fx / (points_3d[..., 2] ** 2)
        m10 = tf.zeros_like(m00)
        m11 = fy / points_3d[..., 2]
        m12 = -points_3d[..., 1] * fy / (points_3d[..., 2] ** 2)

        m0 = tf.stack([m00, m01, m02], axis=-1)
        m1 = tf.stack([m10, m11, m12], axis=-1)

        pixels_2d_D_points_3d = tf.stack([m0, m1], axis=-2)

        return pixels_2d, pixels_2d_D_points_3d

    def project(self, points_3d: tf.Tensor) -> tf.Tensor:
        return self.project_with_jac(points_3d)[0]

    def unproject_with_jac(self,
        depth: tf.Tensor,
        grid: T.Optional[tf.Tensor] = None
    ) -> T.Tuple[tf.Tensor, tf.Tensor]:
        shp = tf.shape(depth)[:-1]
        shp2 = tf.concat([shp, [2]], axis=0)

        # generate grid assuming shp = (..., h, w)
        if grid is None:
            h = shp[-2]
            w = shp[-1]
            x_hw, y_hw = tf.meshgrid(tf.range(w, dtype=self.dtype),
                                     tf.range(h, dtype=self.dtype),
                                     indexing="xy")
            grid = tf.stack([x_hw, y_hw], axis=-1)

        points_3d_unit_depth = camera.perspective.unproject(
            point_2d=tf.broadcast_to(grid, shp2),
            depth=tf.ones_like(depth),
            focal=tf.broadcast_to(self.focal, shp2),
            principal_point=tf.broadcast_to(self.center, shp2),
        )

        points_3d = points_3d_unit_depth * depth
        points_3d_D_depth = points_3d_unit_depth[..., tf.newaxis]

        return points_3d, points_3d_D_depth

    def unproject(self,
        depth: tf.Tensor,
        grid: T.Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        return self.unproject_with_jac(depth, grid)[0]

    def reproject_with_jac(self,
        depth_tgt: tf.Tensor,
        src_T_tgt: Pose3D,
        pixel_tgt: T.Optional[tf.Tensor] = None,
    ) -> T.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        shp = tf.shape(depth_tgt)[:-1]
        src_T_tgt = src_T_tgt.broadcast_to(shp)

        # forward reproject computation
        p_tgt, p_tgt_D_depth_tgt = self.unproject_with_jac(depth_tgt, pixel_tgt)
        p_src = src_T_tgt @ p_tgt
        pz_src = p_src[..., -1, tf.newaxis]
        pixel_src, pixel_src_D_p_src = self.project_with_jac(p_src)

        # jacobian w.r.t. depth
        p_src_D_p_tgt = src_T_tgt.R.to_matrix()
        pixel_src_D_depth_tgt = pixel_src_D_p_src @ p_src_D_p_tgt @ p_tgt_D_depth_tgt

        # jacobian w.r.t. pose
        shp33 = tf.concat([shp, [3, 3]], axis=0)
        eye33 = tf.eye(3, batch_shape=shp)
        p_tgt_wedge_neg = tf.linalg.cross(tf.broadcast_to(p_tgt[..., tf.newaxis, :], shp33), eye33)
        p_src_D_rotation = src_T_tgt.R.to_matrix() @ p_tgt_wedge_neg
        p_src_D_pose = tf.concat([p_src_D_rotation, eye33], axis=-1)
        pixel_src_D_pose = pixel_src_D_p_src @ p_src_D_pose

        return pixel_src, pz_src, pixel_src_D_depth_tgt, pixel_src_D_pose

    def reproject(self,
        depth_tgt: tf.Tensor,
        src_T_tgt: Pose3D,
        pixel_tgt: T.Optional[tf.Tensor] = None,
    ) -> T.Tuple[tf.Tensor, tf.Tensor]:
        pixel_src, pz_src, _, _ = self.reproject_with_jac(depth_tgt, src_T_tgt, pixel_tgt)
        return pixel_src, pz_src

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
