import typing as T

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow_probability import distributions as tfd

if T.TYPE_CHECKING:
    from keras.api._v2 import keras as tfk
    import tensorflow_probability.python.distributions as tfd

from utils.pose3d import Pose3D
from utils.camera import PinholeCam
from utils.params import ParamDict

class PoseTransform(tfk.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.world_T_cam_se3 = self.add_weight(
            name="pose_tangent",
            shape=(6,),
            dtype=tf.float32,
            initializer=tfk.initializers.zeros(),
            trainable=True,
        )
        self.world_T_cam = Pose3D.from_se3(self.world_T_cam_se3)

    def call(self, points3d_cam: tf.Tensor) -> tf.Tensor:
        shp = tf.shape(points_3d)[:-1]
        return self.world_T_cam.broadcast_to(shp) @ points3d_cam

class MLP(tfk.Model):
    def __init__(self,
        units: int,
        num_layers: int,
        num_hidden: int,
        activation: str = "elu",
        kernel_initializer: tfk.initializers.Initializer = tfk.initializers.GlorotNormal(),
        output_kernel_initializer: tfk.initializers.Initializer = tfk.initializers.GlorotNormal(),
        output_bias_initializer: tfk.initializers.Initializer = tfk.initializers.Zeros(),
        output_activation: T.Optional[str] = None,
        weight_decay: T.Optional[float] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        assert num_layers >= 1, "num_layers must be at least 1, otherwise, use tfk.layers.Dense"

        self.units = units
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.output_kernel_initializer = output_kernel_initializer
        self.output_bias_initializer = output_bias_initializer
        self.output_activation = output_activation
        self.weight_decay = weight_decay

        reg = tfk.regularizers.l2(weight_decay) if weight_decay is not None else None
        self.fcs = [
            tfk.layers.Dense(num_hidden,
                             activation=activation,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=reg,
                             name=f"fc_{i}")
            for i in range(num_layers)
        ]
        self.fc_output = tfk.layers.Dense(units=units,
                                          activation=output_activation,
                                          kernel_initializer=output_kernel_initializer,
                                          bias_initializer=output_bias_initializer,
                                          name="fc_output")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for fc in self.fcs:
            x = fc(x)
        return self.fc_output(x)


class NeRD:

    DEFAULT_PARAMS = ParamDict(
        mlp_layers = 7,
        mlp_width = 256,
        mlp_weight_decay = None,
        num_dir_freq = 10,
        max_dir_freq = None,
        num_pos_freq = 10,
        max_pos_freq = None,
        output_bias_init = 0.,
        num_components = 4,
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS) -> None:
        self.p = params
        self.mlp = MLP(
            units=2*self.p.num_components,
            num_layers=self.p.mlp_layers,
            num_hidden=self.p.mlp_width,
            output_bias_initializer=tfk.initializers.Constant(self.p.output_bias_init),
            weight_decay=self.p.mlp_weight_decay,
        )

        self.save = self.mlp.save
        self.save_weights = self.mlp.save_weights
        self.load_weights = self.mlp.load_weights

    def logits_to_dist(self, logits) -> tfd.MixtureSameFamily:
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits[..., :self.p.num_components]),
            components_distribution=tfd.Normal(
                loc=logits[..., self.p.num_components:],
                scale=tf.ones_like(logits[..., self.p.num_components:]),
            ),
        )

    def render_inv_range_raw(self,
        img_size: T.Tuple[int, int],
        camera: PinholeCam,
        world_T_cam_k: Pose3D = Pose3D.identity(),
    ) -> tf.Tensor:
        world_T_cam_k11 = tf.expand_dims(tf.expand_dims(world_T_cam_k, -1), -1)
        x_hw, y_hw = tf.meshgrid(tf.range(img_size[0], dtype=world_T_cam_k.dtype),
                                 tf.range(img_size[1], dtype=world_T_cam_k.dtype),
                                 indexing="xy")
        grid_hw2 = tf.stack([x_hw, y_hw], axis=-1)
        unit_ray_khw3 = world_T_cam_k11.R @ camera.unit_ray(grid_hw2)
        position_khw3 = tf.broadcast_to(world_T_cam_k11.t, tf.shape(unit_ray_khw3))

        mlp_input = self.input_encoding(position_khw3, unit_ray_khw3)
        return self.mlp(mlp_input)

    def render_inv_range_dist(self,
        img_size: T.Tuple[int, int],
        camera: PinholeCam,
        world_T_cam_k: Pose3D = Pose3D.identity(),
    ) -> tfd.MixtureSameFamily:
        mlp_output = self.render_inv_range_raw(img_size, camera, world_T_cam_k)
        return self.logits_to_dist(mlp_output)

    def render_inv_range(self,
        img_size: T.Tuple[int, int],
        camera: PinholeCam,
        world_T_cam_k: Pose3D = Pose3D.identity(),
        min_prob: T.Optional[float] = None,
    ) -> tf.Tensor:
        mlp_output = self.render_inv_range_raw(img_size, camera, world_T_cam_k)

        categorical_khwn = tf.nn.softmax(mlp_output[..., :self.p.num_components])
        inv_ranges_khwn = tf.maximum(mlp_output[..., self.p.num_components:], 1e-3)

        if min_prob is None:
            min_prob = 1. / self.p.num_components
        max_inv_range_khw = tf.zeros_like(inv_ranges_khwn[..., 0])
        for i in range(self.p.num_components):
            max_inv_range_khw = tf.where(categorical_khwn[..., i] > min_prob,
                                         x=tf.maximum(max_inv_range_khw, inv_ranges_khwn[..., i]),
                                         y=max_inv_range_khw)

        return max_inv_range_khw[..., tf.newaxis]

    def frequency_encoding(self,
        data_bn: tf.Tensor,
        num_freqs: int,
        max_freqs: T.Optional[float] = None
    ) -> tf.Tensor:
        num_freqs = float(num_freqs)
        if max_freqs is None:
            max_freqs = num_freqs

        freq_l = 2. ** (tf.range(num_freqs, dtype=data_bn.dtype) / num_freqs * max_freqs) * np.pi
        spectrum_bnl = data_bn[..., tf.newaxis] * freq_l

        batch_shp = tf.shape(data_bn)[:-1]
        spectrum_bm = tf.reshape(spectrum_bnl, tf.concat([batch_shp, [-1]], axis=0))

        return tf.concat([data_bn, tf.sin(spectrum_bm), tf.cos(spectrum_bm)], axis=-1)

    def directional_encoding(self, unit_ray_k3: tf.Tensor) -> tf.Tensor:
        return self.frequency_encoding(unit_ray_k3, self.p.num_dir_freq, self.p.max_dir_freq)

    def positional_encoding(self, position_k3: tf.Tensor) -> tf.Tensor:
        return self.frequency_encoding(position_k3, self.p.num_pos_freq, self.p.max_pos_freq)

    def input_encoding(self, position_k3: tf.Tensor, unit_ray_k3: tf.Tensor) -> tf.Tensor:
        return tf.concat([
            self.directional_encoding(unit_ray_k3),
            self.positional_encoding(position_k3),
        ], axis=-1)
