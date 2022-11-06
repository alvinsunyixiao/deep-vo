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

    def test_matrix_multiply(self):
        R1 = self.default_R
        R2 = Rot3D.random(**self.default_args)

        R3 = R1 @ R2
        R3_recon = Rot3D.from_matrix(R1.to_matrix() @ R2.to_matrix())

        self.assertEqual(R3_recon.shape, self.default_size)
        self.assertQuaternionClose(R3.quat, R3_recon.quat)

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


class Pose3DTestCast(TestCaseBase):
    def setUp(self):
        set_tf_memory_growth()
        tf.random.set_seed(42)
        self.default_size = (5, 5)
        self.default_dtype = tf.float64
        self.default_args = {"size": self.default_size, "dtype": self.default_dtype}
        self.default_T = Pose3D.random(max_translate=1., **self.default_args)

    def test_inv(self):
        T = self.default_T
        T_inv = T.inv()
        self.assertEqual(T_inv.shape, self.default_size)

        T_I_true = Pose3D.identity(**self.default_args)

        T_I = T @ T_inv
        self.assertQuaternionClose(T_I.R.quat, T_I_true.R.quat)
        self.assertArrayClose(T_I.t, T_I_true.t)

        T_I = T_inv @ T
        self.assertQuaternionClose(T_I.R.quat, T_I_true.R.quat)
        self.assertArrayClose(T_I.t, T_I_true.t)

    def test_se3(self):
        T = self.default_T
        T_tanget = T.to_se3(pseudo=False)
        T_recon = Pose3D.from_se3(T_tanget, pseudo=False)

        self.assertEqual(T_recon.shape, self.default_size)
        self.assertQuaternionClose(T.R.quat, T_recon.R.quat)
        self.assertArrayClose(T.t, T_recon.t)

    def test_pseudo_se3(self):
        T = self.default_T
        T_tanget = T.to_se3()
        T_recon = Pose3D.from_se3(T_tanget)

        self.assertEqual(T_recon.shape, self.default_size)
        self.assertQuaternionClose(T.R.quat, T_recon.R.quat)
        self.assertArrayClose(T.t, T_recon.t)

    def test_matrix(self):
        T = self.default_T
        T_mat = T.to_matrix()
        T_recon = Pose3D.from_matrix(T_mat)

        self.assertEqual(T_recon.shape, self.default_size)
        self.assertQuaternionClose(T.R.quat, T_recon.R.quat)
        self.assertArrayClose(T.t, T_recon.t)

    def test_matrix_multiply(self):
        T1 = self.default_T
        T2 = Pose3D.random(max_translate=1., **self.default_args)

        T3 = T1 @ T2
        T3_recon = Pose3D.from_matrix(T1.to_matrix() @ T2.to_matrix())

        self.assertEqual(T3_recon.shape, self.default_size)
        self.assertQuaternionClose(T3.R.quat, T3_recon.R.quat)
        self.assertArrayClose(T3.t, T3_recon.t)

    def test_flatten(self):
        T = self.default_T
        T_flat = T.flatten()

        self.assertEqual(T_flat.shape, (np.prod(self.default_size),))
        self.assertQuaternionClose(T_flat.R.quat, tf.reshape(T.R.quat, (-1, 4)))
        self.assertArrayClose(T_flat.t, tf.reshape(T.t, (-1, 3)))

    def test_broadcast(self):
        T = self.default_T
        T_broadcast = tf.broadcast_to(T[None], (3,) + self.default_size)

        self.assertEqual(T_broadcast.shape, (3,) + self.default_size)

    def test_cast(self):
        T = self.default_T
        T_cast = tf.cast(T, tf.float32)

        self.assertEqual(T_cast.dtype, tf.float32)
        self.assertQuaternionClose(tf.cast(T.R.quat, tf.float32), T_cast.R.quat, angle_eps=1e-3)
        self.assertArrayClose(tf.cast(T.t, tf.float32), T_cast.t)

    def test_stack(self):
        T = self.default_T
        NUM_DUP = 3

        T_stack = tf.stack([T] * NUM_DUP)
        self.assertEqual(T_stack.shape, (NUM_DUP,) + self.default_size)
        self.assertEqual(T_stack.R.quat.shape, (NUM_DUP,) + self.default_size + (4,))
        self.assertEqual(T_stack.t.shape, (NUM_DUP,) + self.default_size + (3,))

        T_stack = tf.stack([T] * NUM_DUP, axis=-1)
        self.assertEqual(T_stack.shape, self.default_size + (NUM_DUP,))
        self.assertEqual(T_stack.R.quat.shape, self.default_size + (NUM_DUP, 4))
        self.assertEqual(T_stack.t.shape, self.default_size + (NUM_DUP, 3))

    def test_concat(self):
        T = self.default_T
        NUM_DUP = 3

        T_stack = tf.concat([T] * NUM_DUP, axis=0)
        self.assertEqual(T_stack.shape, (self.default_size[0] * NUM_DUP, self.default_size[1]))
        self.assertEqual(T_stack.R.quat.shape, (self.default_size[0] * NUM_DUP, self.default_size[1], 4))
        self.assertEqual(T_stack.t.shape, (self.default_size[0] * NUM_DUP, self.default_size[1], 3))

        T_stack = tf.concat([T] * NUM_DUP, axis=-1)
        self.assertEqual(T_stack.shape, (self.default_size[0], self.default_size[1] * NUM_DUP))
        self.assertEqual(T_stack.R.quat.shape, (self.default_size[0], self.default_size[1] * NUM_DUP, 4))
        self.assertEqual(T_stack.t.shape, (self.default_size[0], self.default_size[1] * NUM_DUP, 3))

if __name__ == "__main__":
    unittest.main()
