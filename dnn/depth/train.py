import argparse
import os
import time
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow_probability import distributions as tfd
import typing as T

if T.TYPE_CHECKING:
    from keras.api._v2 import keras as tfk

from dnn.depth.data import DepthDataPipe
from dnn.depth.model import get_model
from utils.params import ParamDict

class Trainer:
    def __init__(self):
        self.args = self._parse_args()
        self.p = ParamDict.from_file(self.args.params)

        # model
        self.model = get_model(self.p.data.img_size, self.p.trainer.weight_decay)
        if self.args.load is not None:
            self.model.load_weights(self.args.load)

        # data
        self.data_pipe = DepthDataPipe(self.p.data)
        self.train_ds = self.data_pipe.build_train_ds()
        self.val_ds = self.data_pipe.build_val_ds()

        # output dir
        self.sess_dir = os.path.join(self.args.output, time.strftime("sess_%y-%m-%d_%H-%M-%S"))
        self.ckpt_dir = os.path.join(self.sess_dir, "ckpts")
        self.log_dir = os.path.join(self.sess_dir, "logs")
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir, "train"), name="train")
        self.val_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir, "validation"), name="validation")
        self.global_step = tf.Variable(0, dtype=tf.int64)
        self.lr = tf.Variable(0, dtype=tf.float32)
        tf.summary.experimental.set_step(self.global_step)

    def _parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--params", type=str, required=True,
                            help="path to the parameter file")
        parser.add_argument("-o", "--output", type=str, required=True,
                            help="path to store weights")
        parser.add_argument("-l", "--load", type=str, default=None,
                            help="path to load weights from")

        return parser.parse_args()

    def _forward(self,
        data: T.Dict[str, T.Any],
        training: bool = True
    ) -> T.Tuple[T.List[tfd.Distribution], tf.Tensor]:
        depth_dists = self.model(data["image"], training=training)

        model_loss = 0.
        for i, depth_dist in enumerate(depth_dists):
            depth_true = data["depth"]
            if i > 0:
                depth_true = tf.image.resize(
                    depth_true, (depth_dist.shape[1], depth_dist.shape[2]), antialias=True)
            depth_loss = -depth_dist.log_prob(depth_true[..., 0])
            depth_mask = (depth_true[..., 0] < self.p.data.max_depth) | \
                         (depth_dist.mean() < self.p.data.max_depth)
            depth_mask = tf.stop_gradient(tf.cast(depth_mask, tf.float32))
            loss = tf.reduce_mean(depth_loss * depth_mask)

            tf.summary.scalar(f"Level {i} Loss", loss)

            model_loss += loss / (4**i)
            tf.print(f"Loss-{i}:", loss, end=" ")

        tf.print("Model Loss:", model_loss)
        tf.summary.scalar("Model Loss", model_loss)

        return depth_dists, model_loss

    @tf.function
    def _train_step(self, data: T.Dict[str, T.Any], optimizer: tfk.optimizers.Optimizer) -> None:
        with tf.GradientTape() as tape:
            tf.print("[Train]", end=" ")

            reg_loss = tf.reduce_sum(self.model.losses)
            tf.print("Reg Loss:", reg_loss, end=" ")
            tf.summary.scalar("Weight Decay Loss", reg_loss)

            depth_dists, model_loss = self._forward(data, training=True)

            total_loss = model_loss + reg_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.global_step % self.p.trainer.img_log_freq == 0:
            tf.summary.image("image", data["image"])
            tf.summary.image("depth true", tf.clip_by_value(data["depth"], 0., 100.) / 100.)
            tf.summary.image("depth std",
                             tf.clip_by_value(depth_dists[0].stddev()[..., tf.newaxis], 0., 10.) / 10.)
            tf.summary.image("depth pred",
                             tf.clip_by_value(depth_dists[0].mean()[..., tf.newaxis], 0., 100.) / 100.)

    @tf.function
    def _val_step(self, data: T.Dict[str, T.Any]) -> None:
        tf.print("[Validation]", end=" ")
        depth_dists, _ = self._forward(data, training=False)

        # visualize the last batch
        tf.summary.image("image", data["image"])
        tf.summary.image("depth true", tf.clip_by_value(data["depth"], 0., 100.) / 100.)
        tf.summary.image("depth std",
                         tf.clip_by_value(depth_dists[0].stddev()[..., tf.newaxis], 0., 10.) / 10.)
        tf.summary.image("depth pred",
                         tf.clip_by_value(depth_dists[0].mean()[..., tf.newaxis], 0., 100.) / 100.)

    def train(self):
        optimizer = tfk.optimizers.Adam(self.lr, clipnorm=self.p.trainer.clipnorm)
        for i in range(self.p.trainer.num_epochs):
            self.lr.assign(self.p.trainer.lr_schedule(i))
            # save model
            if i % self.p.trainer.save_freq == 0:
                print(f"------ Saving Checkpoint for Epoch {i} ------")
                self.model.save(os.path.join(self.ckpt_dir, f"epoch-{i}"))

            print(f"------ Training Epoch {i} ------")
            with self.train_writer.as_default():
                tf.summary.scalar("Learning Rate", self.lr)
                for data in self.train_ds:
                    self.global_step.assign_add(1)
                    self._train_step(data, optimizer)

            print(f"------ Validating Epoch {i} ------")
            with self.val_writer.as_default():
                for data in self.val_ds:
                    self.global_step.assign_add(1)
                    self._val_step(data)

if __name__ == "__main__":
    Trainer().train()
