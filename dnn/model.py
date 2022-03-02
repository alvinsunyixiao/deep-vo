import functools
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import typing as T

from utils.params import ParamDict
from utils.pose3d import Pose3D
from utils.tf_utils import set_tf_memory_growth

from tensorflow_probability import distributions as tfd

class ConvBlock(tfk.layers.Layer):
    def __init__(self,
        filters: int,
        kernel_size: int,
        num_layers: int,
        dilation_rates: T.Union[T.Sequence[int], int],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dilation_rates = dilation_rates
        if isinstance(dilation_rates, int):
            self.dilation_rates = [dilation_rates] * num_layers

        self.convs = []
        for i in range(num_layers):
            self.convs.append(tfk.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                dilation_rate=self.dilation_rates[i],
                activation="relu",
                kernel_initializer="glorot_normal",
                kernel_regularizer=tfk.regularizers.l2(1e-4),
                bias_regularizer=tfk.regularizers.l2(1e-4),
                name=f"conv_{i}"
            ))
        self.pool = tfk.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="pool",
        )

    def call(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "num_layers": self.num_layers,
            "dilation_rates": self.dilation_rates,
        })

        return config

class MinVarEstimator(tfk.layers.Layer):
    def call(self, inputs):
        spatial_axes = (1, 2)
        mus, sigmas = inputs
        variances = tf.square(sigmas)
        inv_variances = 1. / variances
        inv_variances_sum = tf.reduce_sum(inv_variances, axis=spatial_axes, keepdims=True)
        weights = inv_variances / inv_variances_sum

        mu = tf.reduce_sum(mus * weights, axis=spatial_axes)
        variance = 1. / tf.squeeze(inv_variances_sum, axis=(1, 2))

        return mu, tf.sqrt(variance)

class DeepPose:

    DEFAULT_PARAMS=ParamDict()

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        tfk.backend.set_image_data_format("channels_first")

        self.p = params
        self.concat = tfk.layers.Concatenate(
            axis=-1 if tfk.backend.image_data_format() == "channels_last" else 1
        )

        self.block1 = ConvBlock(64, 3, 2, 1, name="block1")
        self.block2 = ConvBlock(128, 3, 2, 1, name="block2")
        self.block3 = ConvBlock(256, 3, 3, 2, name="block3")
        self.block4 = ConvBlock(512, 3, 3, 2, name="block4")
        self.block5 = ConvBlock(512, 3, 3, [4, 4, 8], name="block5")

        small_init = tfk.initializers.random_normal(stddev=1e-3)
        self.conv_mu = tfk.layers.Conv2D(6, 1, 1, kernel_initializer=small_init, name="conv_mu")
        self.conv_sigma = tfk.layers.Conv2D(6, 1, 1, kernel_initializer=small_init, name="conv_sigma")
        self.nnelu = tfk.layers.Lambda(lambda x: tf.nn.elu(x) + 1., name="nnelu")

        self.estimator = MinVarEstimator(name="min_var_estimator")

    def build_model(self) -> tfk.Model:
        image1 = tfk.layers.Input((3, 144, 256), name="image1")
        image2 = tfk.layers.Input((3, 144, 256), name="image2")
        images = self.concat([image1, image2])

        x = self.block1(images)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        mus = self.conv_mu(x)
        sigmas = self.nnelu(self.conv_sigma(x))
        mus = tfk.layers.Permute((2, 3, 1))(mus)
        sigmas = tfk.layers.Permute((2, 3, 1))(sigmas)

        mu, sigma = self.estimator((mus, sigmas))

        inputs = {"image1": image1, "image2": image2}
        outputs = {"mu": mu, "sigma": sigma, "spatial_mus": mus, "spatial_sigmas": sigmas}

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
        loss = -dist.log_prob(c2_T_q.to_se3())

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

        loss_est = GeodesicLoss()((
            deep_pose.output["mu"],
            deep_pose.output["sigma"],
            inputs["pose1"],
            inputs["pose2"],
        ))

        loss_spatial = GeodesicLoss()((
            deep_pose.output["spatial_mus"],
            deep_pose.output["spatial_sigmas"],
            inputs["pose1"][:, tf.newaxis, tf.newaxis],
            inputs["pose2"][:, tf.newaxis, tf.newaxis],
        ))

        model = tfk.Model(inputs=inputs, outputs={"estimator": loss_est, "spatial": loss_spatial})

        loss_est = tf.reduce_mean(loss_est)
        loss_spatial = tf.reduce_mean(tf.reduce_sum(loss_spatial, axis=(1, 2)))
        model.add_metric(loss_est, name="Estimator Loss")
        model.add_metric(loss_spatial, name="Spatial Loss")
        model.add_loss(loss_est)
        #model.add_loss(loss_spatial)

        return model

if __name__ == "__main__":
    set_tf_memory_growth()
    model = DeepPose().build_model()
    model.summary()
