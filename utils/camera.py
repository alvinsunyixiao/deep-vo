from __future__ import annotations

import math
import numpy as np
import typing as T
import tensorflow as tf
from tensorflow_graphics.rendering import camera

from utils.pose3d import Pose3D
from utils.tf_utils import cast_if_needed

class PinholeCam(tf.experimental.BatchableExtensionType):
    __name__ = "PinholeCam"

    focal: tf.Tensor
    center: tf.Tensor

    def __init__(self,
        focal: T.Union[tf.Tensor, np.ndarray],
        center: T.Union[tf.Tensor, np.ndarray],
        dtype: T.Optional[tf.DType] = None,
    ):
        tf.assert_equal(tf.shape(focal)[-1], 2, "focal must be size-4 vector(s)")
        tf.assert_equal(tf.shape(center)[-1], 2, "center must be size-4 vector(s)")
        tf.assert_equal(tf.shape(focal), tf.shape(center),
                "center and focal must be the same size")

        if isinstance(focal, tf.Tensor):
            dtype = dtype or focal.dtype
        elif isinstance(center, tf.Tensor):
            dtype = dtype or center.dtype
        else:
            dtype = dtype or tf.float32

        self.focal = cast_if_needed(focal, dtype)
        self.center = cast_if_needed(center, dtype)

    @property
    def dtype(self) -> tf.DType:
        return self.focal.dtype

    @property
    def shape(self) -> tf.TensorShape:
        return self.focal.shape[:-1]

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
            center=data[..., 2:4],
        )

    def __repr__(self) -> str:
        focal = self.focal
        center = self.center

        if hasattr(focal, "numpy"):
            focal = focal.numpy()
        if hasattr(center, "numpy"):
            center = center.numpy()

        return f"PinholeCam(focal={focal}, center={center})"

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
        m00 = tf.math.divide_no_nan(fx, points_3d[..., 2])
        m01 = tf.zeros_like(m00)
        m02 = tf.math.divide_no_nan(-points_3d[..., 0] * fx, (points_3d[..., 2] ** 2))
        m10 = tf.zeros_like(m00)
        m11 = tf.math.divide_no_nan(fy, points_3d[..., 2])
        m12 = tf.math.divide_no_nan(-points_3d[..., 1] * fy, (points_3d[..., 2] ** 2))

        m0 = tf.stack([m00, m01, m02], axis=-1)
        m1 = tf.stack([m10, m11, m12], axis=-1)

        pixels_2d_D_points_3d = tf.stack([m0, m1], axis=-2)

        return pixels_2d, pixels_2d_D_points_3d

    def project(self, points_3d: tf.Tensor) -> tf.Tensor:
        return self.project_with_jac(points_3d)[0]

    def unit_depth_ray(self, uv: tf.Tensor) -> tf.Tensor:
        uv_shp = tf.shape(uv)

        return camera.perspective.ray(
            point_2d=uv,
            focal=tf.broadcast_to(self.focal, uv_shp),
            principal_point=tf.broadcast_to(self.center, uv_shp),
        )

    def unit_ray(self, uv: tf.Tensor) -> tf.Tensor:
        unit_depth_ray = self.unit_depth_ray(uv)
        return tf.linalg.normalize(unit_depth_ray, axis=-1)[0]

    def unproject_with_jac(self,
        depth: tf.Tensor,
        grid: T.Optional[tf.Tensor] = None
    ) -> T.Tuple[tf.Tensor, tf.Tensor]:
        # generate grid assuming depth has shape = (..., h, w, 1)
        if grid is None:
            grid = self.compute_grid(depth)

        points_3d_unit_depth = self.unit_depth_ray(grid)

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

    def compute_grid(self, depth: tf.Tensor) -> tf.Tensor:
        shp = tf.shape(depth)[:-1]
        shp2 = tf.concat([shp, [2]], axis=0)

        h = shp[-2]
        w = shp[-1]
        x_hw, y_hw = tf.meshgrid(tf.range(w, dtype=self.dtype),
                                 tf.range(h, dtype=self.dtype),
                                 indexing="xy")
        grid = tf.stack([x_hw, y_hw], axis=-1)
        grid = tf.broadcast_to(grid, shp2)

        return grid

    @classmethod
    def from_size_and_fov(cls,
        size_xy: T.Union[tf.Tensor, np.ndarray],
        fov_xy: T.Union[tf.Tensor, np.ndarray],
        dtype: tf.DType = tf.float32,
        use_degree: bool = True,
    ) -> PinholeCam:
        if use_degree:
            fov_xy = fov_xy / 180. * math.pi

        center = size_xy / 2.
        focal = center / tf.math.tan(fov_xy / 2)

        return PinholeCam(focal, center, dtype=dtype)

