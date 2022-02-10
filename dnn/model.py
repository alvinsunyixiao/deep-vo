import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from utils.params import ParamDict
from utils.pose3d import Pose3D

class SameConvBNReLU(tfk.layers.Layer):
    def __init__(self, filters, kernel, stride, **kwargs):
        super().__init__(**kwargs)
        reg = tfk.regularizers.l2(1e-5)
        self.conv = tfk.layers.Conv2D(filters, kernel, stride, "same",
                                      kernel_regularizer=reg, bias_regularizer=reg)
        self.bn = tfk.layers.BatchNormalization()
        self.relu = tfk.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class BaseModel(tfk.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = SameConvBNReLU(16, 11, 2)
        self.conv2 = SameConvBNReLU(32, 11, 2)
        self.conv3 = SameConvBNReLU(64, 7, 2)
        self.conv4 = SameConvBNReLU(128, 7, 2)
        self.conv5 = SameConvBNReLU(256, 5, 2)
        self.conv6 = SameConvBNReLU(512, 5, 2)
        self.conv7 = SameConvBNReLU(1024, 3, 2)


    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x


class DeepPose(tfk.Model):

    DEFAULT_PARAMS=ParamDict()

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        super().__init__()
        self.p = params
        self.concat = tfk.layers.Concatenate()

        self.base_model = BaseModel()

        self.conv_fc_mu = tfk.layers.Conv2D(6, 1, 1)
        self.conv_fc_L = tfk.layers.Conv2D(21, 1, 1,
            kernel_initializer=tfk.initializers.random_normal(stddev=1e-3))
        self.pooling = tfk.layers.GlobalAvgPool2D()

        self.bijector = tfb.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=None)

    def call(self, inputs):
        image1 = inputs["image1"]
        image2 = inputs["image2"]
        images = self.concat([image1, image2])

        x = self.base_model(images)
        mus = self.conv_fc_mu(x)
        Ls = self.conv_fc_L(x)
        mu = self.pooling(mus)
        L = self.pooling(Ls)
        L = self.bijector.forward(x=L)

        # geometric loss
        w_T_c1 = Pose3D.from_storage(inputs["pose1"])
        w_T_c2 = Pose3D.from_storage(inputs["pose2"])
        c1_T_c2 = w_T_c1.inv() @ w_T_c2
        gt = c1_T_c2.to_se3()

        dist = tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)
        geo_loss = -dist.log_prob(gt)
        geo_loss = tf.reduce_mean(geo_loss)
        self.add_loss(geo_loss)
        self.add_metric(geo_loss, name="Geometric Loss")

        return {"mu": mu, "scale_tril": L}

if __name__ == "__main__":
    model = DeepPose()
    dummy_img = tf.zeros((32, 144, 256, 3))
    dummy_pose = tf.zeros((32, 7))
    model({"image1": dummy_img, "image2": dummy_img, "pose1": dummy_pose, "pose2": dummy_pose})
    model.summary()
