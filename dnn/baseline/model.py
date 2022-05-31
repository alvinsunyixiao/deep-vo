import functools
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import typing as T

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
            #kernel_regularizer=tfk.regularizers.l2(1e-4),
            #bias_regularizer=tfk.regularizers.l2(1e-4),
            name="conv",
            use_bias=not has_bn,
        )

        self.bn = tfk.layers.BatchNormalization(
            axis=1 if tfk.backend.image_data_format() == "channels_first" else -1,
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

class MinVarEstimator(tfk.layers.Layer):
    def call(self, inputs):
        mus, log_sigmas = inputs
        log_inv_vars = -2 * log_sigmas
        log_inv_vars_sum = tf.reduce_logsumexp(log_inv_vars, axis=(1, 2), keepdims=True)
        log_weights = log_inv_vars - log_inv_vars_sum

        mu = tf.reduce_sum(mus * tf.exp(log_weights), axis=(1, 2))
        log_sigma = -tf.squeeze(log_inv_vars_sum, axis=(1, 2)) / 2.
        log_sigma = tf.maximum(log_sigma, tf.math.log(1e-4))

        return mu, log_sigma

class DeepPose:

    DEFAULT_PARAMS=ParamDict(
        sigma_scale = np.array([1e3, 1e3, 1e3, 1e4, 1e4, 1e4]),
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        self.p = params
        self.concat = tfk.layers.Concatenate()

        self.conv1 = SameConvBnRelu(64, 7, 2, name="conv1")
        self.pool1 = tfk.layers.MaxPool2D(3, 2, "same", name="pool1")

        self.group1 = ResidualGroup(2, 64, False, name="group1")
        self.group2 = ResidualGroup(3, 128, name="group2")
        self.group3 = ResidualGroup(5, 256, name="group3")
        self.group4 = ResidualGroup(2, 512, name="group4")

        self.conv_mu = tfk.layers.Conv2D(6, 1, 1,
            kernel_initializer=tfk.initializers.random_normal(stddev=1e-4),
            kernel_regularizer=tfk.regularizers.l2(1e-4),
            bias_regularizer=tfk.regularizers.l2(1e-4),
            name="conv_mu",
        )
        self.conv_log_sigma = tfk.layers.Conv2D(6, 1, 1,
            kernel_initializer=tfk.initializers.random_normal(stddev=1e-4),
            kernel_regularizer=tfk.regularizers.l2(1e-4),
            bias_regularizer=tfk.regularizers.l2(1e-4),
            name="conv_log_sigma",
        )

        self.estimator = MinVarEstimator(name="min_var_estimator")

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

        mus = self.conv_mu(x)
        log_sigmas = self.conv_log_sigma(x)
        sigmas = tfk.layers.Activation("exponential")(log_sigmas)

        mu, log_sigma = self.estimator((mus, log_sigmas))
        sigma = tfk.layers.Activation("exponential")(log_sigma)

        inputs = {"image1": image1, "image2": image2}
        outputs = {
            "mu": mu,
            "sigma": sigma,
            "log_sigma": log_sigma,
            "spatial_mus": mus,
            "spatial_sigmas": sigmas
        }

        return tfk.Model(inputs=inputs, outputs=outputs)

class GeodesicLoss(tfk.layers.Layer):
    def call(self, inputs: T.Tuple[tf.Tensor, ...]) -> tf.Tensor:
        mu, log_sigma, pose1, pose2 = inputs
        w_T_c1 = Pose3D.from_storage(pose1)
        w_T_c2 = Pose3D.from_storage(pose2)

        c1_T_q = Pose3D.from_se3(mu)
        w_T_q = w_T_c1 @ c1_T_q
        c2_T_q = w_T_c2.inv() @ w_T_q

        residual = c2_T_q.to_se3()
        nll = tf.reduce_sum(log_sigma + .5 * residual**2 * tf.exp(-2 * log_sigma), axis=-1)
        nll += 3 * tf.math.log(2 * np.pi)

        min_loss = tf.reduce_min(nll)
        max_loss = tf.reduce_max(nll)
        self.add_metric(min_loss, name="Min Loss")
        self.add_metric(max_loss, name="Max Loss")

        loss = tf.reduce_mean(nll)
        self.add_loss(loss)
        self.add_metric(loss, name="Geodesic Loss")

        return nll

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
            deep_pose.output["log_sigma"],
            inputs["pose1"],
            inputs["pose2"],
        ))

        return tfk.Model(inputs=inputs, outputs=loss)

if __name__ == "__main__":
    set_tf_memory_growth()
    model = DeepPose().build_model()
    model.summary()
