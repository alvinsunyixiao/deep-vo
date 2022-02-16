import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from utils.params import ParamDict
from utils.pose3d import Pose3D

class SameConvReLU(tfk.layers.Layer):
    def __init__(self, filters, kernel, stride, **kwargs):
        super().__init__(**kwargs)
        reg = tfk.regularizers.l2(1e-4)
        self.conv = tfk.layers.Conv2D(filters, kernel, stride, "same",
                                      kernel_regularizer=reg, bias_regularizer=reg)
        self.relu = tfk.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x

class BaseModel(tfk.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = SameConvReLU(64, 7, 2)
        vgg = tfk.applications.VGG16(
            include_top=False,
            weights="imagenet",
        )
        self.vgg = tfk.Model(vgg.layers[2].input, vgg.output)


    def call(self, x):
        x = self.conv(x)
        x = self.vgg(x)

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

        self.bijector = tfb.FillScaleTriL()

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
        c1_T_q = Pose3D.from_se3(mu)
        w_T_c1 = Pose3D.from_storage(inputs["pose1"])
        w_T_c2 = Pose3D.from_storage(inputs["pose2"])
        w_T_q = w_T_c1 @ c1_T_q
        c2_T_q = w_T_c2.inv() @ w_T_q

        dist = tfd.MultivariateNormalTriL(loc=tf.zeros_like(mu), scale_tril=L)
        geo_loss = -dist.log_prob(c2_T_q.to_se3())
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
