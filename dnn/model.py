import functools
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import typing as T

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from utils.params import ParamDict
from utils.pose3d import Pose3D
from utils.tf_utils import set_tf_memory_growth

class SameConvBnRelu(tfk.layers.Layer):
    def __init__(self,
        filters: int,
        kernel_size: int,
        strides: int,
        has_bn: bool = True,
        has_relu: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.has_bn = has_bn
        self.has_relu = has_relu

        self.conv = tfk.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            name="conv",
            use_bias=not has_bn,
        )

        self.bn = tfk.layers.BatchNormalization(
            axis=1 if tfk.backend.image_data_format() == "channels_first" else -1,
            momentum=0.9,
            epsilon=1e-5,
            name="bn",
        ) if has_bn else None

        self.relu = tfk.layers.ReLU(name="relu") if has_relu else None

    def get_config(self) -> T.Dict[str, T.Any]:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "has_bn": self.has_bn,
            "has_relu": self.has_relu,
        })
        return config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class ResidualBlock(tfk.layers.Layer):
    def __init__(self,
        filters: int,
        strided: bool = False,
        projection: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        strides = 2 if strided else 1

        self.filters = filters
        self.strided = strided,
        self.projection = projection

        self.branch1 = SameConvBnRelu(
            filters=filters,
            kernel_size=1,
            strides=strides,
            has_relu=False,
            name="branch1"
        ) if projection else None

        self.branch2a = SameConvBnRelu(
            filters=filters,
            kernel_size=3,
            strides=strides,
            name="branch2a"
        )

        self.branch2b = SameConvBnRelu(
            filters=filters,
            kernel_size=3,
            strides=1,
            has_relu=False,
            name="branch2b"
        )

        self.final_relu = tfk.layers.ReLU()

    def get_config(self) -> T.Dict[str, T.Any]:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "strided": self.strided,
            "projection": self.projection,
        })
        return config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        branch1 = self.branch1(x) if self.branch1 else x
        branch2a = self.branch2a(x)
        branch2b = self.branch2b(branch2a)
        return self.final_relu(branch1 + branch2b)

class ResidualGroup(tfk.layers.Layer):
    def __init__(self,
        num_blocks: int,
        filters: int,
        strided: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.filters = filters
        self.strided = strided

        self.blocks = []
        for block_idx in range(num_blocks):
            self.blocks.append(ResidualBlock(
                filters=filters,
                # only do 1x1 projection on the first block of each group
                projection=(block_idx == 0),
                # only do strided convolution on the first block of each group
                strided=(strided and block_idx == 0),
                name="block_" + str(block_idx),
            ))

    def get_config(self) -> T.Dict[str, T.Any]:
        config = super().get_config()
        config.update({
            "num_blocks": self.num_blocks,
            "filters": self.filters,
            "strided": self.strided,
        })
        return config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

class DeepPose:

    DEFAULT_PARAMS=ParamDict()

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        self.p = params
        self.concat = tfk.layers.Concatenate()

        self.conv1 = SameConvBnRelu(32, 7, 2, name="conv1")
        self.pool1 = tfk.layers.MaxPool2D(3, 2, "same", name="pool1")

        self.group1 = ResidualGroup(2, 32, False, name="group1")
        self.group2 = ResidualGroup(2, 64, name="group2")
        self.group3 = ResidualGroup(2, 128, name="group3")
        self.group4 = ResidualGroup(2, 192, name="group4")
        self.group5 = ResidualGroup(2, 256, name="group5")
        self.group6 = ResidualGroup(2, 512, name="group6")

        self.flatten = tfk.layers.Flatten()

        self.fc_mu = tfk.layers.Dense(6,
            kernel_initializer=tfk.initializers.random_normal(stddev=1e-3), name="fc_mu")
        self.fc_sigma = tfk.layers.Dense(6,
            kernel_initializer=tfk.initializers.random_normal(stddev=1e-3), activation="exponential", name="fc_sigma")

    def build_model(self) -> tfk.Model:
        image1 = tfk.layers.Input((144, 256, 3), name="image1")
        image2 = tfk.layers.Input((144, 256, 3), name="image2")
        images = self.concat([image1, image2])

        x = self.conv1(images)
        x = self.pool1(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = self.group6(x)

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
