import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
if T.TYPE_CHECKING:
    from keras.api._v2 import keras as tfk

from utils.pose3d import Pose3D

class PoseTransform(tfk.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.world_T_cam_se3 = self.add_weight(
            name="pose_tangent",
            shape=(6,),
            dtype=tf.float32,
            initializer=tfk.initializers.zeros(),
            trainable=self.trainable,
        )
        self.world_T_cam = Pose3D.from_se3(self.world_T_cam_se3)

    def call(self, points3d_cam: tf.Tensor) -> tf.Tensor:
        shp = tf.shape(points_3d)[:-1]
        return self.world_T_cam.broadcast_to(shp) @ points3d_cam

class NeRD(tfk.Model):
    def __init__(self, **kwargs):
        super(NeRD, self).__init__(**kwargs)


