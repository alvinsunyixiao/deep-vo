import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from utils.params import ParamDict
from utils.pose3d import Pose3D

class SameConvRelu(tfk.layers.Layer):
    def __init__(self, filters, kernel, stride, **kwargs):
        super().__init__(**kwargs)
        reg = tfk.regularizers.l2(1e-5)
        self.conv = tfk.layers.Conv2D(filters, kernel, stride, "same", activation="relu",
                                      kernel_regularizer=reg, bias_regularizer=reg)

    def call(self, x):
        x = self.conv(x)

        return x

class BaseModel(tfk.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = SameConvRelu(16, 11, 2)
        self.conv2 = SameConvRelu(32, 11, 2)
        self.conv3 = SameConvRelu(64, 7, 2)
        self.conv4 = SameConvRelu(128, 7, 2)
        self.conv5 = SameConvRelu(256, 5, 2)
        self.conv6 = SameConvRelu(512, 5, 2)
        self.conv7 = SameConvRelu(1024, 3, 2)

        self.conv_fc = tfk.layers.Conv2D(6, 1, 1)
        self.pooling = tfk.layers.GlobalAvgPool2D()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.conv_fc(x)
        x = self.pooling(x)

        return x


class DeepPose(tfk.Model):

    DEFAULT_PARAMS=ParamDict()

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        super().__init__()
        self.p = params
        self.concat = tfk.layers.Concatenate()

        self.base_model = BaseModel()

    def call(self, inputs):
        image1 = inputs["image1"]
        image2 = inputs["image2"]
        images = self.concat([image1, image2])

        x = self.base_model(images)

        # geometric loss
        w_T_c1 = Pose3D.from_storage(inputs["pose1"])
        w_T_c2 = Pose3D.from_storage(inputs["pose2"])
        c1_T_q = Pose3D.from_se3(x)
        w_T_q = w_T_c1 @ c1_T_q
        c2_T_q = w_T_c2.inv() @ w_T_q
        se3 = c2_T_q.to_se3()

        # geometric loss
        geo_loss = tf.reduce_mean(tf.linalg.norm(se3, ord=1, axis=-1))
        self.add_loss(geo_loss)
        self.add_metric(geo_loss, name="Geometric Loss")

        return x

if __name__ == "__main__":
    model = DeepPose()
    dummy_img = tf.zeros((32, 144, 256, 3))
    dummy_pose = tf.zeros((32, 7))
    model({"image1": dummy_img, "image2": dummy_img, "pose1": dummy_pose, "pose2": dummy_pose})
    model.summary()
