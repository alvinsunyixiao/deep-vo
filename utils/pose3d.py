from __future__ import annotations

import functools

import numpy as np
import typing as T
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import \
    quaternion, axis_angle, rotation_matrix_3d, euler

class Rot3D(tf.experimental.BatchableExtensionType):
    """
    3D rotation represented with a quaternion [x, y, z, w]
    """
    __name__ = "Rot3D"

    quat: tf.Tensor

    def __init__(self,
        quat: T.Union[tf.Tensor, np.ndarray],
        dtype: T.Optional[tf.DType] = None,
        renormalize: bool = False,
    ) -> None:
        tf.assert_equal(tf.shape(quat)[-1], 4, "quaternion must be size-4 vector(s)")

        if not isinstance(quat, tf.Tensor):
            dtype = dtype or tf.float32
            quat = tf.convert_to_tensor(quat, dtype=dtype)
        elif dtype is not None and quat.dtype != dtype:
            quat = tf.cast(quat, dtype=dtype)

        if renormalize:
            quat = tf.math.l2_normalize(quat, axis=-1)

        self.quat = quat

    @property
    def shape(self) -> tf.TensorShape:
        return self.quat.shape[:-1]

    @property
    def dtype(self) -> tf.DType:
        return self.quat.dtype

    def __getitem__(self, key: tf.IndexedSlices) -> Rot3D:
        return Rot3D(self.quat[key])

    def __matmul__(self, other: T.Union[Rot3D, tf.Tensor]) -> T.Union[Rot3D, tf.Tensor]:
        if isinstance(other, Rot3D):
            return Rot3D(quaternion.multiply(self.quat, other.quat), renormalize=True)
        else:
            return quaternion.rotate(other, self.quat)

    def __repr__(self) -> str:
        quat = self.quat
        if hasattr(self.quat, "numpy"):
            quat = quat.numpy()
        return f"Rot3D(quat={quat})"

    def inv(self) -> Rot3D:
        return Rot3D(quaternion.conjugate(self.quat))

    def to_so3(self) -> tf.Tensor:
        a, theta = axis_angle.from_quaternion(self.quat)
        return a * theta

    def to_matrix(self) -> tf.Tensor:
        return rotation_matrix_3d.from_quaternion(self.quat)

    def to_euler(self) -> tf.Tensor:
        return euler.from_quaternion(self.quat)

    def flatten(self) -> Rot3D:
        return Rot3D(tf.reshape(self.quat, (-1, 4)))

    def broadcast_to(self, shape: T.Union[T.Sequence[int], tf.TensorShape, tf.Tensor]) -> Rot3D:
        return Rot3D(tf.broadcast_to(self.quat, tf.concat([shape, [4]], axis=0)))

    @classmethod
    def identity(cls, size: T.Tuple[int, ...] = (), dtype: tf.DType = tf.float32) -> Rot3D:
        return Rot3D(tf.broadcast_to(tf.constant([0.0, 0.0, 0.0, 1.0], dtype=dtype), size + (4,)))

    @classmethod
    def from_matrix(cls, rot_mat: tf.Tensor) -> Rot3D:
        return Rot3D(quaternion.from_rotation_matrix(rot_mat))

    @classmethod
    def from_so3(cls, so3: T.Union[tf.Tensor, np.ndarray]) -> Rot3D:
        theta = tf.linalg.norm(so3, axis=-1, keepdims=True)
        axis = tf.math.divide_no_nan(so3, theta)
        q = quaternion.from_axis_angle(axis, theta)
        q_eye = tf.broadcast_to(tf.constant([0.0, 0.0, 0.0, 1.0], dtype=so3.dtype), tf.shape(q))

        dtype = so3.dtype
        if isinstance(so3, tf.Tensor):
            dtype = dtype.as_numpy_dtype
        eps = np.finfo(dtype).eps

        return Rot3D(tf.where(theta < eps, x=q_eye, y=q))

    @classmethod
    def random(cls,
        max_angle: T.Union[float, tf.Tensor] = np.pi,
        size: T.Tuple[int, ...] = (),
        dtype: tf.DType = tf.float32,
    ) -> Rot3D:
        a = tf.random.normal(size + (3,), dtype=dtype)
        a = a / tf.linalg.norm(a, axis=-1, keepdims=True)
        theta = tf.random.uniform(size + (1,), -max_angle, max_angle, dtype=dtype)
        return Rot3D(quaternion.from_axis_angle(a, theta), dtype=dtype)

class Pose3D(tf.experimental.BatchableExtensionType):
    __name__ = "Pose3D"

    R: Rot3D
    t: tf.Tensor

    def __init__(self,
        R: Rot3D,
        t: T.Union[tf.Tensor, np.ndarray],
        dtype: T.Optional[tf.DType] = None,
    ) -> None:
        tf.assert_equal(tf.shape(t)[-1], 3, "t must be size-3 vector(s)")
        tf.assert_equal(tf.shape(R.quat)[:-1], tf.shape(t)[:-1], "batch dimensions do not match")
        self.R = R
        self.t = t

        if dtype is None:
            dtype = R.dtype
        elif dtype != R.dtype:
            self.R = Rot3D(R.quat, dtype)

        if not isinstance(t, tf.Tensor):
            self.t = tf.convert_to_tensor(t, dtype=dtype)
        elif t.dtype != dtype:
            self.t = tf.cast(t, dtype)

    @property
    def dtype(self) -> tf.DType:
        return self.R.dtype

    @property
    def shape(self) -> tf.TensorShape:
        return self.R.shape

    def __getitem__(self, key: tf.IndexedSlices) -> Pose3D:
        return Pose3D(self.R[key], self.t[key])

    def __matmul__(self, other: T.Union[Pose3D, tf.Tensor]) -> T.Union[Pose3D, tf.Tensor]:
        if isinstance(other, Pose3D):
            return Pose3D(
                t=self.R @ other.t + self.t,
                R=self.R @ other.R,
            )
        else:
            tf.assert_equal(tf.shape(other)[-1], 3, "Pose3D can only be applied to 3D vectors")
            tf.assert_equal(tf.shape(other)[:-1], tf.shape(self.t)[:-1],
                            "batch dimensions do not match")
            return self.R @ other + self.t

    def __repr__(self) -> str:
        t = self.t
        if hasattr(t, "numpy"):
            t = t.numpy()
        return f"Pose3D(R={self.R}, t={t})"

    def inv(self) -> Pose3D:
        R_inv = self.R.inv()
        return Pose3D(
            t=R_inv @ (-self.t),
            R=R_inv,
        )

    def flatten(self) -> Pose3D:
        return Pose3D(
            R=self.R.flatten(),
            t=tf.reshape(self.t, (-1, 3)),
        )

    def to_matrix(self) -> tf.Tensor:
        batch_shp = tf.shape(self.t)[:-1]
        R_b33 = self.R.to_matrix()
        t_b31 = self.t[..., tf.newaxis]
        top_b34 = tf.concat([R_b33, t_b31], axis=-1)

        bottom_4 = tf.constant([0, 0, 0, 1], dtype=self.dtype)
        bottom_b4 = tf.broadcast_to(bottom_4, tf.concat([batch_shp, [4]], axis=0))
        bottom_b14 = bottom_b4[..., tf.newaxis, :]

        return tf.concat([top_b34, bottom_b14], axis=-2)

    def to_se3(self, pseudo: bool = True) -> tf.Tensor:
        w = self.R.to_so3()
        if pseudo:
            return tf.concat([w, self.t], axis=-1)

        eps = np.finfo(self.dtype.as_numpy_dtype).eps
        theta = tf.linalg.norm(w, axis=-1, keepdims=True)
        t = self.t
        wt = tf.linalg.cross(w, t)
        wwt = tf.linalg.cross(w, wt)
        tp = t - .5 * wt + \
            tf.math.divide_no_nan(
                (1 - tf.math.divide_no_nan(theta * tf.cos(theta / 2), 2 * tf.sin(theta / 2))),
                theta**2,
            ) * wwt
        t_safe = tf.where(theta < eps, x=t, y=tp)

        return tf.concat([w, t_safe], axis=-1)

    def to_storage(self) -> tf.Tensor:
        return tf.concat([self.R.quat, self.t], axis=-1)

    def broadcast_to(self, shape: T.Union[T.Sequence[int], tf.TensorShape, tf.Tensor]) -> Pose3D:
        return Pose3D(self.R.broadcast_to(shape),
                      tf.broadcast_to(self.t, tf.concat([shape, [3]], axis=0)))

    @classmethod
    def identity(cls, size: T.Tuple[int, ...] = (), dtype: tf.DType = tf.float32) -> Pose3D:
        return Pose3D(
            R=Rot3D.identity(size, dtype=dtype),
            t=tf.zeros(size + (3,), dtype=dtype),
            dtype=dtype,
        )

    @classmethod
    def from_matrix(cls, pose_mat: tf.Tensor) -> Pose3D:
        return Pose3D(
            R=Rot3D.from_matrix(pose_mat[..., :3, :3]),
            t=pose_mat[..., :3, 3],
        )

    @classmethod
    def from_storage(cls, storage: T.Union[tf.Tensor, np.ndarray]) -> Pose3D:
        tf.assert_equal(tf.shape(storage)[-1], 7), "Pose3D storage are size-7 vectors"
        return Pose3D(
            R=Rot3D(storage[..., :4]),
            t=storage[..., 4:],
        )

    @classmethod
    def from_se3(cls,
                 se3: T.Union[tf.Tensor, np.ndarray],
                 pseudo: bool = True) -> Pose3D:
        tf.assert_equal(tf.shape(se3)[-1], 6, "se3 vectors must be size-6")
        w = se3[..., :3]
        t = se3[..., 3:]
        theta = tf.linalg.norm(w, axis=-1, keepdims=True)

        if pseudo:
            return Pose3D(
                t=t,
                R=Rot3D.from_so3(w)
            )

        dtype = se3.dtype
        if isinstance(se3, tf.Tensor):
            dtype = dtype.as_numpy_dtype
        eps = np.finfo(dtype).eps

        wt = tf.linalg.cross(w, t)
        wwt = tf.linalg.cross(w, wt)
        tp = t + tf.math.divide_no_nan(1 - tf.cos(theta), theta**2) * wt + \
                tf.math.divide_no_nan(theta - tf.sin(theta), theta**3) * wwt
        t_safe = tf.where(theta < eps, x=t, y=tp)

        return Pose3D(
            t=t_safe,
            R=Rot3D.from_so3(w),
        )

    @classmethod
    def random(cls,
        max_angle: float = np.pi,
        max_translate: float = 0.,
        size: T.Tuple[int, ...] = (),
        dtype: tf.DType = tf.float32,
    ) -> Pose3D:
        t = tf.random.uniform(size + (3,), -max_translate, max_translate, dtype=dtype)

        return Pose3D(Rot3D.random(max_angle, size, dtype), t, dtype=dtype)

class RandomPose3DGen:
    def __init__(self,
        min_pos: T.Tuple[float, float, float],
        max_pos: T.Tuple[float, float, float],
        min_rp: T.Tuple[float, float],
        max_rp: T.Tuple[float, float],
    ):
        min_pos = np.array(min_pos, dtype=float)
        max_pos = np.array(max_pos, dtype=float)
        self.pos_gen = functools.partial(np.random.uniform, low=min_pos, high=max_pos)

        min_rpy = np.array(min_rp + (-np.pi,), dtype=float)
        max_rpy = np.array(max_rp + (np.pi,), dtype=float)
        self.rpy_gen = functools.partial(np.random.uniform, low=min_rpy, high=max_rpy)

    def __call__(self) -> Pose3D:
        pos = self.pos_gen()
        rpy = self.rpy_gen()

        return Pose3D(
            t=pos,
            R=Rot3D(quaternion.from_euler(rpy)),
        )

