from __future__ import annotations

import functools

import numpy as np
import typing as T
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import \
    quaternion, axis_angle, rotation_matrix_3d, euler

class Rot3D:
    """
    3D rotation represented with a quaternion [x, y, z, w]
    """
    def __init__(self, quaternion: T.Union[tf.Tensor, np.ndarray]):
        tf.assert_equal(tf.shape(quaternion)[-1], 4, "quaternion must be size-4 vector(s)")
        self.quat = tf.cast(quaternion, tf.float32)

    def inv(self) -> Rot3D:
        return Rot3D(quaternion.inverse(self.quat))

    def __matmul__(self, other: T.Union[Rot3D, tf.Tensor]) -> T.Union[Rot3D, tf.Tensor]:
        if isinstance(other, Rot3D):
            return Rot3D(quaternion.multiply(self.quat, other.quat))
        else:
            return quaternion.rotate(other, self.quat)

    @property
    def shape(self) -> T.Tuple[int, ...]:
        return tf.shape(self.quat)[:-1]

    def to_so3(self, eps: float = 1e-4) -> tf.Tensor:
        a, theta = axis_angle.from_quaternion(self.quat)
        w = a * theta
        return tf.where(theta < eps, x=tf.zeros_like(w), y=w)

    def to_matrix(self) -> tf.Tensor:
        return rotation_matrix_3d.from_quaternion(self.quat)

    def to_euler(self) -> tf.Tensor:
        return euler.from_quaternion(self.quat)

    def flatten(self) -> Rot3D:
        return Rot3D(tf.reshape(self.quat, (-1, 4)))

    def __repr__(self) -> str:
        return f"Rot3D(quaternion={self.quat.numpy()})"

    def __getitem__(self, key) -> Rot3D:
        return Rot3D(self.quat[key])

    @classmethod
    def identity(cls, size: T.Tuple[int, ...] = ()) -> Rot3D:
        return Rot3D(tf.broadcast_to(tf.constant([0.0, 0.0, 0.0, 1.0]), size + (4,)))

    @classmethod
    def from_matrix(cls, rot_mat: tf.Tensor) -> Rot3D:
        return Rot3D(quaternion.from_rotation_matrix(rot_mat))

    @classmethod
    def from_so3(cls, so3: T.Union[tf.Tensor, np.ndarray], eps: float = 1e-4) -> Rot3D:
        theta = tf.linalg.norm(so3, axis=-1, keepdims=True)
        axis = tf.math.divide_no_nan(so3, theta)
        q = quaternion.from_axis_angle(axis, theta)
        q_eye = tf.broadcast_to(tf.constant([0.0, 0.0, 0.0, 1.0]), tf.shape(q))

        return Rot3D(tf.where(theta < eps, x=q_eye, y=q))

    @classmethod
    def random(cls, max_angle: T.Union[float, tf.Tensor], size: T.Tuple[int, ...] = ()) -> Rot3D:
        a = tf.random.normal(size + (3,))
        a = a / tf.linalg.norm(a, axis=-1, keepdims=True)
        theta = tf.random.uniform(size + (1,), -max_angle, max_angle)
        return Rot3D(quaternion.from_axis_angle(a, theta))


class Pose3D:
    def __init__(self, orientation: Rot3D, position: T.Union[tf.Tensor, np.ndarray]):
        tf.assert_equal(tf.shape(position)[-1], 3, "position must be size-3 vector(s)")
        tf.assert_equal(orientation.shape, tf.shape(position)[:-1], "batch dimensions do not match")
        self.R = orientation
        self.t = tf.cast(position, dtype=tf.float32)

    def inv(self) -> Pose3D:
        R_inv = self.R.inv()
        return Pose3D(
            position=R_inv @ (-self.t),
            orientation=R_inv,
        )

    @property
    def shape(self) -> T.Tuple[int, ...]:
        return self.R.shape

    def flatten(self) -> Pose3D:
        return Pose3D(
            orientation=self.R.flatten(),
            position=tf.reshape(self.t, (-1, 3)),
        )

    def __matmul__(self, other: T.Union[Pose3D, tf.Tensor]) -> T.Union[Pose3D, tf.Tensor]:
        if isinstance(other, Pose3D):
            return Pose3D(
                position=self.R @ other.t + self.t,
                orientation=self.R @ other.R,
            )
        else:
            tf.assert_equal(tf.shape(other)[-1], 3, "Pose3D can only be applied to 3D vectors")
            tf.assert_equal(tf.shape(other)[:-1], self.shape(), "batch dimensions do not match")
            return self.R @ other + self.t

    def to_se3(self, eps: float = 1e-4) -> tf.Tensor:
        w = self.R.to_so3(eps)
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

        return tf.concat([t_safe, w], axis=-1)

    def to_pseudo_se3(self) -> tf.Tensor:
        w = self.R.to_so3()
        return tf.concat([self.t, w], axis=-1)

    def to_storage(self) -> tf.Tensor:
        return tf.concat([self.R.quat, self.t], axis=-1)

    def __repr__(self) -> str:
        return f"Pose3D(position={self.t.numpy()}, orientation={self.R})"

    def __getitem__(self, key) -> Pose3D:
        return Pose3D(self.R[key], self.t[key])

    @classmethod
    def identity(cls, size: Tuple[int, ...] = ()) -> Pose3D:
        return Pose3D(
            orientation=Rot3D.identity(size),
            position=tf.zeros(size + (3,)),
        )

    @classmethod
    def from_storage(cls, storage: T.Union[tf.Tensor, np.ndarray]) -> Pose3D:
        tf.assert_equal(tf.shape(storage)[-1], 7), "Pose3D storage are size-7 vectors"
        return Pose3D(
            orientation=Rot3D(storage[..., :4]),
            position=storage[..., 4:],
        )

    @classmethod
    def from_se3(cls, se3: T.Union[tf.Tensor, np.ndarray], eps: float = 1e-4) -> Pose3D:
        tf.assert_equal(tf.shape(se3)[-1], 6, "se3 vectors must be size-6")
        t = se3[..., :3]
        w = se3[..., 3:]
        theta = tf.linalg.norm(w, axis=-1, keepdims=True)

        wt = tf.linalg.cross(w, t)
        wwt = tf.linalg.cross(w, wt)
        tp = t + tf.math.divide_no_nan(1 - tf.cos(theta), theta**2) * wt + \
                tf.math.divide_no_nan(theta - tf.sin(theta), theta**3) * wwt
        t_safe = tf.where(theta < eps, x=t, y=tp)

        return Pose3D(
            position=t_safe,
            orientation=Rot3D.from_so3(w, eps),
        )

    @classmethod
    def random(cls, max_angle: float, max_translate: float, size: T.Tuple[int, ...] = ()) -> Pose3D:
        t = tf.random.uniform(size + (3,), -max_translate, max_translate)

        return Pose3D(Rot3D.random(max_angle, size), t)

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
            position=pos,
            orientation=Rot3D(quaternion.from_euler(rpy)),
        )
