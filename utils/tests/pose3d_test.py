import unittest
import numpy as np
import tensorflow as tf

from utils.pose3d import Rot3D, Pose3D
from utils.test_utils import TestCaseBase
from utils.tf_utils import set_tf_memory_growth

class Rot3DTestCase(TestCaseBase):
    def setUp(self):
        set_tf_memory_growth()
        tf.random.set_seed(42)
        self.default_size = (5, 5)
        self.default_dtype = tf.float64
        self.default_args = {"size": self.default_size, "dtype": self.default_dtype}
        self.default_R = Rot3D.random(**self.default_args)

    def test_inv(self):
        R = self.default_R
        R_inv = R.inv()
        self.assertEqual(R_inv.shape, self.default_size)

        R_I_true = Rot3D.identity(**self.default_args)

        R_I = R @ R_inv
        self.assertQuaternionClose(R_I.quat, R_I_true.quat)

        R_I = R_inv @ R
        self.assertQuaternionClose(R_I.quat, R_I_true.quat)

    def test_so3(self):
        R = self.default_R
        R_tanget = R.to_so3()
        R_recon = Rot3D.from_so3(R_tanget)

        self.assertEqual(R_recon.shape, self.default_size)
        self.assertQuaternionClose(R.quat, R_recon.quat)

    def test_matrix(self):
        R = self.default_R
        R_mat = R.to_matrix()
        R_recon = Rot3D.from_matrix(R_mat)

        self.assertEqual(R_recon.shape, self.default_size)
        self.assertQuaternionClose(R.quat, R_recon.quat)

    def test_flatten(self):
        R = self.default_R
        R_flat = R.flatten()

        self.assertEqual(R_flat.shape, (np.prod(self.default_size),))
        self.assertQuaternionClose(R_flat.quat, tf.reshape(R.quat, (-1, 4)))

    def test_broadcast(self):
        R = self.default_R
        R_broadcast = tf.broadcast_to(R[None], (3,) + self.default_size)

        self.assertEqual(R_broadcast.shape, (3,) + self.default_size)

    def test_cast(self):
        R = self.default_R
        R_cast = tf.cast(R, tf.float32)

        self.assertEqual(R_cast.dtype, tf.float32)
        self.assertQuaternionClose(tf.cast(R.quat, tf.float32), R_cast.quat, angle_eps=1e-3)

    def test_stack(self):
        R = self.default_R
        NUM_DUP = 3

        R_stack = tf.stack([R] * NUM_DUP)
        self.assertEqual(R_stack.shape, (NUM_DUP,) + self.default_size)
        self.assertEqual(R_stack.quat.shape, (NUM_DUP,) + self.default_size + (4,))

        R_stack = tf.stack([R] * NUM_DUP, axis=-1)
        self.assertEqual(R_stack.shape, self.default_size + (NUM_DUP,))
        self.assertEqual(R_stack.quat.shape, self.default_size + (NUM_DUP, 4))

    def test_concat(self):
        R = self.default_R
        NUM_DUP = 3

        R_stack = tf.concat([R] * NUM_DUP, axis=0)
        self.assertEqual(R_stack.shape, (self.default_size[0] * NUM_DUP, self.default_size[1]))
        self.assertEqual(R_stack.quat.shape, (self.default_size[0] * NUM_DUP, self.default_size[1], 4))

        R_stack = tf.concat([R] * NUM_DUP, axis=-1)
        self.assertEqual(R_stack.shape, (self.default_size[0], self.default_size[1] * NUM_DUP))
        self.assertEqual(R_stack.quat.shape, (self.default_size[0], self.default_size[1] * NUM_DUP, 4))

if __name__ == "__main__":
    unittest.main()
