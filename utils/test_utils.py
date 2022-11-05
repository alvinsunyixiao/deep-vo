import unittest
import numpy as np
import tensorflow as tf
import typing as T

from utils.tf_utils import convert_to_numpy
from tensorflow_graphics.geometry.transformation import quaternion, axis_angle

class TestCaseBase(unittest.TestCase):
    def assertArrayClose(self,
        a: T.Union[tf.Tensor, np.ndarray],
        b: T.Union[tf.Tensor, np.ndarray],
        eps: float = 1e-6,
        msg: T.Optional[str] = None
    ) -> None:
        msg = "" if msg is None else ": " + msg
        a = convert_to_numpy(a)
        b = convert_to_numpy(b)
        diff = np.abs(a - b)
        assert np.all(diff <= eps), \
            f"a is not close to b (max diff {np.max(diff)})" + msg

    def assertQuaternionClose(self,
        q1: T.Union[tf.Tensor, np.ndarray],
        q2: T.Union[tf.Tensor, np.ndarray],
        angle_eps: float = 1e-6,
        msg: T.Optional[str] = None
    ) -> None:
        msg = "" if msg is None else ": " + msg
        q_dot = tf.clip_by_value(tf.reduce_sum(q1 * q2, axis=-1), -1., 1.)
        rel_ang = 2 * tf.acos(q_dot)
        rel_ang = (rel_ang + np.pi) % (2 * np.pi) - np.pi
        assert np.all(rel_ang.numpy() <= angle_eps), \
            f"q1 is not close to q2 (max angle diff {rel_ang.numpy().max()})"
