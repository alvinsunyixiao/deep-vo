import functools
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import typing as T

from utils.params import ParamDict
from utils.pose3d import Pose3D
from utils.tf_utils import set_tf_memory_growth

from tensorflow_probability import distributions as tfd

class DeepPose:

    DEFAULT_PARAMS=ParamDict()

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        self.p = params
        self.concat = tfk.layers.Concatenate()

        Conv2D = functools.partial(tfk.layers.Conv2D,
            padding="same",
            activation="elu",
            kernel_initializer="glorot_normal",
            kernel_regularizer=tfk.regularizers.l2(1e-4),
            bias_regularizer=tfk.regularizers.l2(1e-4),
        )
        self.conv1 = Conv2D(16, 7, 2, name="conv1")
        self.conv2 = Conv2D(32, 7, 2, name="conv2")
        self.conv3 = Conv2D(64, 5, 2, name="conv3")
        self.conv4 = Conv2D(128, 5, 2, name="conv4")
        self.conv5 = Conv2D(256, 3, 2, name="conv5")
        self.conv6 = Conv2D(512, 3, 2, name="conv6")
        self.conv7 = Conv2D(1024, 3, 2, name="conv7")

        self.flatten = tfk.layers.Flatten()

        self.fc_mu = tfk.layers.Dense(6, kernel_initializer=tfk.initializers.random_normal(stddev=1e-3), name="fc_mu")
        self.fc_sigma = tfk.layers.Dense(6, kernel_initializer=tfk.initializers.random_normal(stddev=1e-3), activation="exponential", name="fc_sigma")

    def build_model(self) -> tfk.Model:
        image1 = tfk.layers.Input((144, 256, 3), name="image1")
        image2 = tfk.layers.Input((144, 256, 3), name="image2")
        images = self.concat([image1, image2])

        x = self.conv1(images)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.flatten(x)

        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        inputs = {"image1": image1, "image2": image2}
        outputs = {"mu": mu, "sigma": sigma}

        return tfk.Model(inputs=inputs, outputs=outputs)

class GeodesicLoss(tfk.layers.Layer):
    def call(self, inputs: T.Tuple[tf.Tensor, ...]) -> tf.Tensor:
        mu, sigma, pose1, pose2 = inputs
        w_T_c1 = Pose3D.from_storage(pose1)
        w_T_c2 = Pose3D.from_storage(pose2)

        c1_T_q = Pose3D.from_se3(mu)
        w_T_q = w_T_c1 @ c1_T_q
        c2_T_q = w_T_c2.inv() @ w_T_q

        dist = tfd.MultivariateNormalDiag(loc=tf.zeros_like(mu), scale_diag=sigma)
        loss = tf.reduce_mean(-dist.log_prob(c2_T_q.to_se3()))

        self.add_loss(loss)
        self.add_metric(loss, name="Geodesic Loss")

        return loss

class DeepPoseTrain(DeepPose):

    def build_model(self) -> tfk.Model:
        deep_pose = super().build_model()

        inputs = {
            "image1": deep_pose.input["image1"],
            "image2": deep_pose.input["image2"],
            "pose1": tfk.layers.Input((7,), name="pose1"),
            "pose2": tfk.layers.Input((7,), name="pose2"),
        }

        loss = GeodesicLoss()((
            deep_pose.output["mu"],
            deep_pose.output["sigma"],
            inputs["pose1"],
            inputs["pose2"],
        ))

        return tfk.Model(inputs=inputs, outputs=loss)

if __name__ == "__main__":
    set_tf_memory_growth()
    model = DeepPose().build_model()
    model.summary()
