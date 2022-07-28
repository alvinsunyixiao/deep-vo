import os
import tensorflow as tf
import typing as T
import tensorflow.keras as tfk
if T.TYPE_CHECKING:
    from keras.api._v2 import keras as tfk

from dnn.baseline.model import ResidualGroup, SameConvBnRelu
from utils.params import ParamDict
from utils.pose3d import Pose3D

class Resnet18Encoder(tfk.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.conv1 = SameConvBnRelu(64, 7, 2, name="conv1")
        self.pool1 = tfk.layers.MaxPool2D(3, 2, "same", name="pool1")

        self.group1 = ResidualGroup(2, 64, False, name="group1")
        self.group2 = ResidualGroup(2, 128, name="group2")
        self.group3 = ResidualGroup(2, 256, name="group3")
        self.group4 = ResidualGroup(2, 512, name="group4")

    def call(self, x: tf.Tensor) -> T.List[tf.Tensor]:
        features = []
        x = self.conv1(x)
        features.append(x)
        features.append(self.group1(self.pool1(features[-1])))
        features.append(self.group2(features[-1]))
        features.append(self.group3(features[-1]))
        features.append(self.group4(features[-1]))

        return features

class Conv2DReflect(tfk.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, activation: str = "elu", **kwargs) -> None:
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

        self.conv = tfk.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            activation=activation,
        )

    def get_config(self) -> T.Dict[str, T.Any]:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
        })
        return config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        p = self.kernel_size // 2
        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode="REFLECT")
        x = self.conv(x)
        return x

class Conv3x3Upsample(tfk.layers.Layer):
    def __init__(self, filters: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.filters = filters
        self.conv = Conv2DReflect(filters, 3, name="conv3x3")
        self.upsample = tfk.layers.UpSampling2D(2, interpolation="nearest", name="upsample")

    def get_config(self) -> T.Dict[str, T.Any]:
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    def call(self, x):
        x = self.conv(x)
        x = self.upsample(x)

        return x

class DepthDecoder(tfk.layers.Layer):
    def __init__(self,
        num_channels: T.Sequence[int] = [16, 32, 64, 128, 256],
        num_scales: int = 4,
        min_depth: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.num_scales = num_scales
        self.min_depth = min_depth

        self.upsamples = []
        self.proj_convs = []
        self.depth_convs = []

        for i, ch in enumerate(num_channels):
            self.upsamples.append(Conv3x3Upsample(ch, name=f"upsample_{i}"))
            self.proj_convs.append(Conv2DReflect(ch, 3, name=f"proj_conv_{i}"))

        for i in range(self.num_scales):
            self.depth_convs.append(tfk.layers.Conv2D(1, 3,
                padding="same", activation="softplus", name=f"depth_conv_{i}"))

    def get_config(self) -> T.Dict[str, T.Any]:
        config = super().get_config()
        config.update({
            "num_channels": self.num_channels,
            "num_scales": self.num_scales,
            "min_depth": self.min_depth,
        })
        return config

    def call(self, features):
        x = features[-1]

        depths = []

        for i in range(len(self.num_channels) - 1, -1, -1):
            x = self.upsamples[i](x)
            if i > 0:
                x = tf.concat([x, features[i - 1]], axis=-1)
            x = self.proj_convs[i](x)

            if i < self.num_scales:
                depth = self.depth_convs[i](x)
                depth = tf.maximum(depth, self.min_depth)
                depths.insert(0, depth)

        return depths

class PoseDecoder(tfk.layers.Layer):
    def __init__(self,
                 filters: int = 256,
                 num_layers: int = 3,
                 output_scale: int = 1e-2,
                 **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.num_layers = num_layers
        self.output_scale = output_scale

        self.convs = []
        for i in range(num_layers):
            self.convs.append(tfk.layers.Conv2D(
                filters, 3, 1, "same",
                activation="relu",
                name=f"conv_{i}",
            ))
        self.pose_conv = tfk.layers.Conv2D(
            filters=6,
            kernel_size=1,
            name="pose_conv",
        )
        self.pool = tfk.layers.GlobalAvgPool2D(name="global_avg_pool")

    def get_config(self) -> T.Dict[str, T.Any]:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "num_layers": self.num_layers,
            "output_scale": self.output_scale
        })
        return config

    def call(self, features):
        x = features[-1]
        for conv in self.convs:
            x = conv(x)
        x = self.pose_conv(x)
        x = self.pool(x)

        return self.output_scale * x

class SCSFM:

    DEFAULT_PARAMS=ParamDict(
        img_size=(160, 320),
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params

        self.depth_encoder = Resnet18Encoder(name="depth_encoder")
        self.depth_decoder = DepthDecoder(name="depth_decoder")
        self.pose_encoder = Resnet18Encoder(name="pose_encoder")
        self.pose_decoder = PoseDecoder(name="pose_decoder")
        self.concat = tfk.layers.Concatenate()

        self.depth_net = self._build_depth_model()
        self.pose_net = self._build_pose_model()

    def _build_depth_model(self) -> tfk.Model:
        image = tfk.layers.Input(self.p.img_size + (3,), name="image")

        feats = self.depth_encoder(image)
        depth = self.depth_decoder(feats)

        return tfk.Model(inputs=image, outputs=depth)

    def _build_pose_model(self) -> tfk.Model:
        image1 = tfk.layers.Input(self.p.img_size + (3,), name="image1")
        image2 = tfk.layers.Input(self.p.img_size + (3,), name="image2")
        image_concat = self.concat([image1, image2])
        c1_T_c2 = self.pose_decoder(self.pose_encoder(image_concat))

        inputs = {"image1": image1, "image2": image2}

        return tfk.Model(inputs=inputs, outputs=c1_T_c2)

    def save(self, output_dir: str) -> None:
        self.depth_net.save(os.path.join(output_dir, "depth_net"))
        self.pose_net.save(os.path.join(output_dir, "pose_net"))

    def load_weights(self, weights_dir: str) -> None:
        depth_dir = os.path.join(weights_dir, "depth_net")
        pose_dir = os.path.join(weights_dir, "pose_net")
        if os.path.isdir(depth_dir):
            self.depth_net.load_weights(depth_dir)
            print(f"[Weight Loaded] DepthNet: {depth_dir}")
        if os.path.isdir(pose_dir):
            self.pose_net.load_weights(pose_dir)
            print(f"[Weight Loaded] PoseNet: {pose_dir}")

    @property
    def trainable_variables(self):
        return self.depth_net.trainable_variables + self.pose_net.trainable_variables

if __name__ == "__main__":
    sc_sfm = SCSFM()
    sc_sfm.depth_net.summary()
    sc_sfm.pose_net.summary()
