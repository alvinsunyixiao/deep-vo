import typing as T

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
if T.TYPE_CHECKING:
    from keras.api._v2 import keras as tfk

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
        activation: str = "relu",
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
        output_bias_init = -3.,
        mlp_activation = "relu",
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS) -> None:
        self.p = params
        self.mlp = MLP(
            units=1,
            num_layers=self.p.mlp_layers,
            num_hidden=self.p.mlp_width,
            activation=self.p.mlp_activation,
            output_activation="softplus",
            output_bias_initializer=tfk.initializers.Constant(self.p.output_bias_init),
            weight_decay=self.p.mlp_weight_decay,
        )

        self.save = self.mlp.save
        self.save_weights = self.mlp.save_weights
        self.load_weights = self.mlp.load_weights

        self.freq_alpha = tf.Variable(1., trainable=False, dtype=tf.float32)

    def set_freq_alpha(self, alpha: float) -> None:
        assert alpha >= 0. and alpha <= 1., "alpha must be in range [0, 1]"
        self.freq_alpha.assign(alpha)

    def render_inv_range(self,
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

    def frequency_encoding(self,
        data_bn: tf.Tensor,
        num_freqs: int,
        max_freqs: T.Optional[float] = None
    ) -> tf.Tensor:
        num_freqs = float(num_freqs)
        if max_freqs is None:
            max_freqs = num_freqs

        freq_idx_l = tf.range(num_freqs, dtype=data_bn.dtype)
        freq_l = 2. ** (freq_idx_l / num_freqs * max_freqs) * np.pi
        spectrum_bnl = data_bn[..., tf.newaxis] * freq_l

        freq_mask_l = (freq_idx_l < self.freq_alpha * num_freqs)
        sin_spec_bnl = tf.where(freq_mask_l, x=tf.sin(spectrum_bnl), y=tf.zeros_like(spectrum_bnl))
        cos_spec_bnl = tf.where(freq_mask_l, x=tf.cos(spectrum_bnl), y=tf.zeros_like(spectrum_bnl))

        batch_shp = tf.shape(data_bn)[:-1]
        spec_shp = tf.concat([batch_shp, [-1]], axis=0)
        sin_spec_bm = tf.reshape(sin_spec_bnl, spec_shp)
        cos_spec_bm = tf.reshape(cos_spec_bnl, spec_shp)

        return tf.concat([data_bn, sin_spec_bm, cos_spec_bm], axis=-1)

    def directional_encoding(self, unit_ray_k3: tf.Tensor) -> tf.Tensor:
        return self.frequency_encoding(unit_ray_k3, self.p.num_dir_freq, self.p.max_dir_freq)

    def positional_encoding(self, position_k3: tf.Tensor) -> tf.Tensor:
        return self.frequency_encoding(position_k3, self.p.num_pos_freq, self.p.max_pos_freq)

    def input_encoding(self, position_k3: tf.Tensor, unit_ray_k3: tf.Tensor) -> tf.Tensor:
        # TODO: make this a parameter
        # position_k3 /= 10.
        return tf.concat([
            self.directional_encoding(unit_ray_k3),
            self.positional_encoding(position_k3),
        ], axis=-1)
