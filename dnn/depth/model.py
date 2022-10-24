import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import typing as T

if T.TYPE_CHECKING:
    from keras.api._v2 import keras as tfk

from dnn.sc_sfm.model import Resnet18Encoder, Conv3x3Upsample, Conv2DReflect

class DepthDecoder(tfk.layers.Layer):
    def __init__(self,
        num_channels: T.Sequence[int] = [16, 32, 64, 128, 256],
        num_scales: int = 4,
        weight_decay: T.Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.num_scales = num_scales
        self.weight_decay = weight_decay

        self.upsamples = []
        self.proj_convs = []
        self.depth_convs = []

        self.normal_layer = tfp.layers.IndependentNormal()

        for i, ch in enumerate(num_channels):
            self.upsamples.append(Conv3x3Upsample(ch, weight_decay=weight_decay, name=f"upsample_{i}"))
            self.proj_convs.append(Conv2DReflect(ch, 3, activation=None, weight_decay=weight_decay, name=f"proj_conv_{i}"))

        for i in range(self.num_scales):
            self.depth_convs.append(tfk.layers.Conv2D(
                filters=2,
                kernel_size=3,
                padding="same",
                name=f"depth_conv_{i}",
            ))

    def get_config(self) -> T.Dict[str, T.Any]:
        config = super().get_config()
        config.update({
            "num_channels": self.num_channels,
            "num_scales": self.num_scales,
            "weight_decay": self.weight_decay,
        })
        return config

    def call(self, features):
        x = features[-1]

        depths = []
        depth_means = []
        depth_scales = []
        depth_feats = []

        for i in range(len(self.num_channels) - 1, -1, -1):
            x = self.upsamples[i](x)
            if i > 0:
                x = tf.concat([x, features[i - 1]], axis=-1)
            x = self.proj_convs[i](x)
            depth_feats.insert(0, x)
            x = tf.nn.elu(x)

            if i < self.num_scales:
                depth_params = self.depth_convs[i](x)
                depth_mean, depth_scale = tf.unstack(depth_params, num=2, axis=-1)
                depth_mean = tf.nn.softplus(depth_mean)
                depth_scale = tf.maximum(depth_scale, tfp.math.softplus_inverse(1e-3))
                depth_params = tf.stack([depth_mean, depth_scale], axis=-1)
                depths.insert(0, self.normal_layer(depth_params))
                depth_means.insert(0, depth_mean)
                depth_scales.insert(0, depth_scale)

        return depths, depth_means, depth_scales, depth_feats

def get_model(img_size: T.Tuple[int, int],
              weight_decay: T.Optional[float] = None,
              name: T.Optional[str] = None) -> tfk.Model:
    image = tfk.layers.Input(img_size + (3,), name="image")

    feats = Resnet18Encoder(weight_decay=weight_decay, name="encoder")(image)
    depths, depth_means, depth_scales, depth_feats = DepthDecoder(weight_decay=weight_decay,
                                                                  name="decoder")(feats)

    outputs = {
        "depths": depths,
        "depth_means": depth_means,
        "depth_scales": depth_scales,
        "depth_feats": depth_feats,
    }

    return tfk.Model(inputs=image, outputs=outputs, name=name)

if __name__ == "__main__":
    model = get_model((160, 256), 1e-4)
    model.summary()
