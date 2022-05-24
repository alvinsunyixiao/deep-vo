import tensorflow as tf
import tensorflow.keras as tfk
import typing as T

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

class Conv3x3Upsample(tfk.layers.Layer):
    def __init__(self, filters: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.filters = filters
        # TODO(alvin): try reflection padding
        self.conv = tfk.layers.Conv2D(filters, 3, activation="relu", padding="same", name="conv3x3")
        self.upsample = tfk.layers.UpSampling2D(2, interpolation="bilinear", name="upsample")

    def get_config(self) -> T.Dict[str, T.Any]:
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    def call(self, x):
        x = self.conv(x)
        x = self.upsample(x)

        return x

class DisparityDecoder(tfk.layers.Layer):
    def __init__(self,
        num_channels: T.Sequence[int] = [16, 32, 64, 128, 256],
        alpha: float = 2.,
        beta: float = 2e-3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.alpha = alpha
        self.beta = beta

        self.upsamples = []
        self.proj_convs = []
        self.disp_conv = tfk.layers.Conv2D(1, 3,
            padding="same", activation="sigmoid", name="disp_conv")

        for i, ch in enumerate(num_channels):
            self.upsamples.append(Conv3x3Upsample(ch, name=f"upsample_{i}"))
            self.proj_convs.append(
                tfk.layers.Conv2D(ch, 3, padding="same", activation="relu", name=f"proj_conv_{i}"))

    def get_config(self) -> T.Dict[str, T.Any]:
        config = super().get_config()
        config.update({
            "num_channels": self.num_channels,
            "alpha": self.alpha,
            "beta": self.beta,
        })
        return config

    def call(self, features):
        x = features[-1]

        for i in range(len(self.num_channels) - 1, -1, -1):
            x = self.upsamples[i](x)
            if i > 0:
                x = tf.concat([x, features[i - 1]], axis=-1)
            x = self.proj_convs[i](x)
            if i == 0:
                disp = self.alpha * self.disp_conv(x) + self.beta

        return disp

class PoseDecoder(tfk.layers.Layer):
    def __init__(self, filters=256, num_layers=3, output_scale=1e-2, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.num_layers = num_layers
        self.output_scale = output_scale

        self.relu1 = tfk.layers.ReLU(name="relu1")
        self.convs = []
        for i in range(num_layers):
            self.convs.append(tfk.layers.Conv2D(
                filters, 3, 1, "same",
                activation="relu",
                name=f"conv_{i}",
            ))
        self.pose_conv = tfk.layers.Conv2D(6, 1, name="pose_conv")
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
        img_size=(160, 256),
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params

        self.disp_encoder = Resnet18Encoder(name="disp_encoder")
        self.disp_decoder = DisparityDecoder(name="disp_decoder")
        self.pose_encoder = Resnet18Encoder(name="pose_encoder")
        self.pose_decoder = PoseDecoder(name="pose_decoder")
        self.concat = tfk.layers.Concatenate()
        self.inverse = tfk.layers.Lambda(lambda x: 1. / x, name="inverse")

    def build_model(self) -> tfk.Model:
        image1 = tfk.layers.Input(self.p.img_size + (3,), name="image1")
        image2 = tfk.layers.Input(self.p.img_size + (3,), name="image2")

        # disparity from image1
        disp1 = self.disp_decoder(self.disp_encoder(image1))

        # disparity from image2
        disp2 = self.disp_decoder(self.disp_encoder(image2))

        # pose from image1 to image2
        image_concat12 = self.concat([image1, image2])
        c1_T_c2 = self.pose_decoder(self.pose_encoder(image_concat12))

        image_concat21 = self.concat([image2, image1])
        c2_T_c1 = self.pose_decoder(self.pose_encoder(image_concat21))

        inputs = {"image1": image1, "image2": image2}
        outputs = {
            "disp1": disp1,
            "disp2": disp2,
            "depth1": self.inverse(disp1),
            "depth2": self.inverse(disp2),
            "c1_T_c2": c1_T_c2,
            "c2_T_c1": c2_T_c1,
        }

        return tfk.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    sc_sfm = SCSFM()
    model = sc_sfm.build_model()
    model.summary()
