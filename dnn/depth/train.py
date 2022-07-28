import argparse
import os
import time
import tensorflow as tf
import tensorflow.keras as tfk
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
        self.model = get_model(self.p.data.img_size)
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
        self.train_writer = tf.summary.create_file_writer(self.log_dir, name="train")
        self.val_writer = tf.summary.create_file_writer(self.log_dir, name="validation")
        self.global_step = tf.Variable(0, dtype=tf.int64)
        tf.summary.experimental.set_step(self.global_step)

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--params", type=str, required=True,
                            help="path to the parameter file")
        parser.add_argument("-o", "--output", type=str, required=True,
                            help="path to store weights")
        parser.add_argument("-l", "--load", type=str, default=None,
                            help="path to load weights from")

        return parser.parse_args()

    @tf.function
    def _train_step(self, data: T.Dict[str, T.Any], optimizer: tfk.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            depth_dists = self.model(data["image"])

            total_loss = 0.
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

                total_loss += loss / (4**i)
                tf.print(f"Loss-{i}:", loss, end=" ")

        tf.print("Total Loss:", total_loss)
        tf.summary.scalar("Total Loss", total_loss)

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.global_step % self.p.trainer.img_log_freq == 0:
            tf.summary.image("image", data["image"])
            tf.summary.image("depth true", tf.clip_by_value(data["depth"], 0., 100.) / 100.)
            tf.summary.image("depth std",
                             tf.clip_by_value(depth_dists[0].stddev()[..., tf.newaxis], 0., 10.) / 10.)
            tf.summary.image("depth pred",
                             tf.clip_by_value(depth_dists[0].mean()[..., tf.newaxis], 0., 100.) / 100.)

    def train(self):
        optimizer = tfk.optimizers.Adam(1e-4, clipnorm=1.)
        for i in range(self.p.trainer.num_epochs):
            print(f"------ Saving Checkpoint ------")
            self.model.save(os.path.join(self.ckpt_dir, f"epoch-{i}"))
            print(f"------ Starting Epoch {i} ------")
            with self.train_writer.as_default():
                for data in self.train_ds:
                    self.global_step.assign_add(1)
                    self._train_step(data, optimizer)

if __name__ == "__main__":
    Trainer().train()
