from utils.tf_utils import set_tf_memory_growth
set_tf_memory_growth(True)

import argparse
import math
import os
import time
import typing as T

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
if T.TYPE_CHECKING:
    from keras.api._v2 import keras as tfk

from tqdm import tqdm, trange

from dnn.implicit.data import PointLoader, T_DATA_DICT
from dnn.implicit.model import NeRD
from utils.pose3d import Pose3D
from utils.params import ParamDict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", type=str, default="params/default.py",
                        help="path to the parameter file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output directory to store checkpoints and logs")
    parser.add_argument("-w", "--weights", type=str, default=None,
                        help="path to load weights from (optional)")
    return parser.parse_args()

class Trainer:

    DEFAULT_PARAMS = ParamDict(
        num_epochs=1000,
        save_freq=10,
        log_freq=50,
        lr=ParamDict(
            init=1e-3,
            end=1e-5,
            delay=None,
        ),
    )

    def __init__(self, params: ParamDict) -> None:
        self.p = params.trainer
        self.data = PointLoader(params.data)
        self.model = NeRD(params.model)

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.lr = tf.Variable(0., trainable=False, dtype=tf.float32)
        self.optimizer = tfk.optimizers.Adam(self.lr)

    def compute_loss(self, data_dict: T_DATA_DICT) -> T_DATA_DICT:
        # transform from cam to reference
        pos_cam_b3, dir_cam_b3, inv_range_gt_b = self.data.generate_samples(
            data_dict["points_cam_b3"])

        # TODO(alvin): support not using GT here
        pos_ref_b3 = data_dict["ref_T_cam_b"] @ pos_cam_b3
        dir_ref_b3 = data_dict["ref_T_cam_b"].R @ dir_cam_b3

        # MLP inference
        mlp_input = self.model.input_encoding(pos_ref_b3, dir_ref_b3)
        inv_range_dist = self.model.logits_to_dist(self.model.mlp(mlp_input))

        loss_b = -inv_range_dist.log_prob(inv_range_gt_b)

        return {
            "loss": tf.reduce_mean(loss_b),
            "pos_ref_b3": pos_ref_b3,
            "dir_ref_b3": dir_ref_b3,
        }

    def lr_schedule(self, step: tf.Tensor) -> tf.Tensor:
        step_float = tf.cast(step, tf.float32)

        # compute no delay base lr
        num_steps = self.p.num_epochs * self.data.p.epoch_size
        step_diff = (math.log(self.p.lr.end) - math.log(self.p.lr.init)) / num_steps
        base_lr = tf.exp(math.log(self.p.lr.init) + step_float * step_diff)

        # apply warm start multiplier
        mult = 1.
        if self.p.lr.delay is not None and step < self.p.lr.delay:
            mult = (tf.sin(step_float / float(self.p.lr.delay) * math.pi - math.pi / 2) + 1.) / 2.

        return mult * base_lr

    @tf.function(jit_compile=True)
    def train_step(self, data_dict: T_DATA_DICT) -> T_DATA_DICT:
        self.lr.assign(self.lr_schedule(self.global_step))

        with tf.GradientTape() as tape:
            meta_dict = self.compute_loss(data_dict)

        self.global_step.assign_add(1)

        grad = tape.gradient(meta_dict["loss"], self.model.mlp.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.mlp.trainable_variables))

        meta_dict.update(data_dict, grad=grad)
        return meta_dict

    def normalize_inv_range(self, inv_range_khw1: tf.Tensor) -> tf.Tensor:
        inv_min = tf.reduce_min(inv_range_khw1, axis=(1, 2, 3), keepdims=True)
        inv_max = tf.reduce_max(inv_range_khw1, axis=(1, 2, 3), keepdims=True)
        return (inv_range_khw1 - inv_min) / (inv_max - inv_min)

    @tf.function(jit_compile=True)
    def validate_step(self) -> tf.Tensor:
        inv_range_khw1 = self.model.render_inv_range(
            self.data.p.img_size, self.data.p.cam, self.data.data_dict["ref_T_cams"])
        return self.normalize_inv_range(inv_range_khw1)

    @tf.function
    def log_step(self, meta_dict: T_DATA_DICT) -> None:
        with tf.name_scope("losses"):
            tf.summary.scalar("total loss", meta_dict["loss"], step=self.global_step)

        with tf.name_scope("misc"):
            tf.summary.scalar("learning rate", self.lr, step=self.global_step)

        with tf.name_scope("data"):
            with tf.name_scope("pos_ref"):
                tf.summary.histogram("x", meta_dict["pos_ref_b3"][:, 0], step=self.global_step)
                tf.summary.histogram("y", meta_dict["pos_ref_b3"][:, 1], step=self.global_step)
                tf.summary.histogram("z", meta_dict["pos_ref_b3"][:, 2], step=self.global_step)
            with tf.name_scope("dir_ref"):
                tf.summary.histogram("x", meta_dict["dir_ref_b3"][:, 0], step=self.global_step)
                tf.summary.histogram("y", meta_dict["dir_ref_b3"][:, 1], step=self.global_step)
                tf.summary.histogram("z", meta_dict["dir_ref_b3"][:, 2], step=self.global_step)

        with tf.name_scope("weights"):
            for var in self.model.mlp.trainable_variables:
                tf.summary.histogram(var.name, var, step=self.global_step)

        with tf.name_scope("gradients"):
            for var, grad in zip(self.model.mlp.trainable_variables, meta_dict["grad"]):
                tf.summary.histogram(var.name, grad, step=self.global_step)

    def run(self, output_dir: str, weights: T.Optional[str] = None, debug: bool = False) -> None:
        sess_dir = time.strftime("sess_%y-%m-%d_%H-%M-%S")
        ckpt_dir = os.path.join(output_dir, sess_dir, "ckpts")
        log_dir = os.path.join(output_dir, sess_dir, "logs")

        train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
        val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "validation"))

        if weights is not None:
            self.model.load_weights(weights)

        with val_writer.as_default():
            tf.summary.image("gt inverse range",
                             self.normalize_inv_range(self.data.data_dict["inv_range_imgs"]),
                             step=0)

        for epoch in trange(self.p.num_epochs, desc="Epoch"):
            with train_writer.as_default():
                for data_dict in tqdm(self.data.dataset, desc="Iteration", leave=False):
                    meta_dict = self.train_step(data_dict)
                    if self.global_step % self.p.log_freq == 0:
                        self.log_step(meta_dict)

            inv_range_norm_khw1 = self.validate_step()
            with val_writer.as_default():
                tf.summary.image("rendered inverse range",
                    inv_range_norm_khw1, step=self.global_step)

            if not debug and epoch % self.p.save_freq == self.p.save_freq - 1:
                self.model.save(os.path.join(ckpt_dir, f"epoch-{epoch+1}"))

if __name__ == "__main__":
    args = parse_args()
    params = ParamDict.from_file(args.params)

    trainer = Trainer(params)
    trainer.run(args.output, args.weights)
