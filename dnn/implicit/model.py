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
        kernel_initializer: str = "glorot_normal",
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
                                          kernel_initializer=kernel_initializer,
                                          name="fc_output")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for fc in self.fcs:
            x = fc(x)
        return self.fc_output(x)


class NeRD:

    DEFAULT_PARAMS = ParamDict(
        mlp_layers = 5,
        mlp_width = 64,
        mlp_weight_decay = 1e-4,
        num_dir_freq = 10,
        num_pos_freq = 10,
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS) -> None:
        self.p = params
        self.mlp = MLP(
            units=1,
            num_layers=self.p.mlp_layers,
            num_hidden=self.p.mlp_width,
            output_activation="exponential",
            weight_decay=self.p.mlp_weight_decay,
        )

    def render_depth(self,
        img_size: T.Tuple[int, int],
        camera: PinholeCam,
        world_T_cam: Pose3D
    ) -> tf.Tensor:
        x_hw, y_hw = tf.meshgrid(tf.range(img_size[0], dtype=world_T_cam.dtype),
                                 tf.range(img_size[1], dtype=world_T_cam.dtype),
                                 indexing="xy")
        grid_hw2 = tf.stack([x_hw, y_hw], axis=-1)
        unit_ray_hw3 = camera.unit_ray(grid_hw2)
        position_hw3 = tf.broadcast_to(world_T_cam.t, tf.shape(unit_ray_hw3))

        mlp_input = self.input_encoding(position_hw3, unit_ray_hw3)

        return self.mlp(mlp_input)

    def frequency_encoding(self, data_bn: tf.Tensor, num_freqs: int) -> tf.Tensor:
        freq_l = 2. ** tf.range(num_freqs, dtype=data_bn.dtype) * np.pi
        spectrum_bnl = data_bn[..., tf.newaxis] * freq_l

        batch_shp = tf.shape(data_bn)[:-1]
        spectrum_bm = tf.reshape(spectrum_bnl, tf.concat([batch_shp, [-1]], axis=0))

        return tf.concat([data_bn, tf.sin(spectrum_bm), tf.cos(spectrum_bm)], axis=-1)

    def directional_encoding(self, unit_ray_k3: tf.Tensor) -> tf.Tensor:
        return self.frequency_encoding(unit_ray_k3, self.p.num_dir_freq)

    def positional_encoding(self, position_k3: tf.Tensor) -> tf.Tensor:
        return self.frequency_encoding(position_k3, self.p.num_pos_freq)

    def input_encoding(self, position_k3: tf.Tensor, unit_ray_k3: tf.Tensor) -> tf.Tensor:
        return tf.concat([
            self.directional_encoding(unit_ray_k3),
            self.positional_encoding(position_k3),
        ], axis=-1)
