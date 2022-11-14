import argparse
import math
import time
import os
import typing as T

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as tfk
if T.TYPE_CHECKING:
    from keras.api._v2 import keras as tfk

from dnn.implicit.data import PointLoader, T_DATA_DICT
from dnn.implicit.model import NeRD
from utils.pose3d import Pose3D
from utils.params import ParamDict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", type=str, default="params.py",
                        help="path to the parameter file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output directory to store checkpoints and logs")
    return parser.parse_args()

class Trainer:

    DEFAULT_PARAMS = ParamDict(
        num_epochs=1000,
        save_freq=20,
        lr=ParamDict(
            init=1e-3,
            end=1e-5,
            delay=100000,
        ),
    )

    def __init__(self, params: ParamDict, output_dir: str) -> None:
        self.p = params.trainer
        self.data = PointLoader(params.data)
        self.model = NeRD(params.model)
        sess_dir = time.strftime("sess_%y-%m-%d_%H-%M-%S")
        self.ckpt_dir = os.path.join(output_dir, sess_dir, "ckpts")
        self.log_dir = os.path.join(output_dir, sess_dir, "logs")
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.lr = tf.Variable(0., trainable=False, dtype=tf.float32)
        self.optimizer = tfk.optimizers.Adam(self.lr)

    def compute_loss(self, data_dict: T_DATA_DICT) -> tf.Tensor:
        # transform from cam to reference
        pos_cam_b3, dir_cam_b3, inv_range_gt_b = self.data.generate_samples(
            data_dict["points_cam_b3"])

        # TODO(alvin): support not using GT here
        pos_ref_b3 = data_dict["ref_T_cam_b"] @ pos_cam_b3
        dir_ref_b3 = data_dict["ref_T_cam_b"].R @ dir_cam_b3

        # MLP inference
        mlp_input = self.model.input_encoding(pos_ref_b3, dir_ref_b3)
        inv_range_pred_b1 = tf.maximum(self.model.mlp(mlp_input), 1e-3) # max range 1000 meters
        inv_range_pred_b = inv_range_pred_b1[:, 0]

        # forward direction loss
        loss_f_b = tf.boolean_mask(tf.square(inv_range_pred_b - inv_range_gt_b),
            (inv_range_pred_b <= inv_range_gt_b) & \
            (tf.math.acos(tf.einsum("ij,ij->i", dir_cam_b3, data_dict["directions_cam_b3"])) <= math.radians(30)))

        # backward direction loss
        points_ref_pred_b3 = pos_ref_b3 + dir_ref_b3 / inv_range_pred_b1
        # TODO(alvin): again, support not using GT here
        points_virtual_pred_b3 = data_dict["ref_T_virtual"].inv() @ points_ref_pred_b3
        depth_virtual_proj_b = tf.linalg.norm(points_virtual_pred_b3, axis=-1)
        pixels_uv_b2 = self.data.p.cam.project(points_virtual_pred_b3)

        # filter out invalid pixels
        valid_mask_b = (depth_virtual_proj_b >= self.data.p.min_depth) & \
                       (points_virtual_pred_b3[:, -1] > 0) & \
                       (pixels_uv_b2[:, 0] >= 0) & (pixels_uv_b2[:, 1] >= 0) & \
                       (pixels_uv_b2[:, 0] <= self.data.p.img_size[0] - 1.) & \
                       (pixels_uv_b2[:, 1] <= self.data.p.img_size[0] - 1.)
        pixels_uv_b2 = tf.boolean_mask(pixels_uv_b2, valid_mask_b)
        points_virtual_pred_b3 = tf.boolean_mask(points_virtual_pred_b3, valid_mask_b)
        depth_virtual_proj_b = tf.boolean_mask(depth_virtual_proj_b, valid_mask_b)
        inv_range_proj_b = 1. / depth_virtual_proj_b

        # sample from depth image
        inv_range_sampled_b = tfa.image.resampler(data_dict["inv_range_virtual_hw1"][tf.newaxis],
                                              pixels_uv_b2[tf.newaxis])[0, :, 0]
        loss_b_b = tf.boolean_mask(tf.square(inv_range_proj_b - inv_range_sampled_b),
                                   inv_range_sampled_b <= inv_range_proj_b)

        return loss_f_b, loss_b_b

    def lr_schedule(self, step: tf.Tensor) -> tf.Tensor:
        step_float = tf.cast(step, tf.float32)

        # compute no delay base lr
        num_steps = self.p.num_epochs * self.data.p.epoch_size
        step_diff = (math.log(self.p.lr.end) - math.log(self.p.lr.init)) / num_steps
        base_lr = tf.exp(math.log(self.p.lr.init) + step_float * step_diff)

        # apply warm start multiplier
        mult = 1.
        if step < self.p.lr.delay:
            mult = (tf.sin(step_float / float(self.p.lr.delay) * math.pi - math.pi / 2) + 1.) / 2.

        return mult * base_lr

    @tf.function
    def train_step(self, data_dict: T_DATA_DICT):
        self.lr.assign(self.lr_schedule(self.global_step))

        with tf.GradientTape() as tape:
            loss_f_b, loss_b_b = self.compute_loss(data_dict)

            loss_f = 0.
            loss_b = 0.
            if tf.shape(loss_f_b)[0] > 0:
                loss_f = tf.reduce_mean(loss_f_b)
            if tf.shape(loss_b_b)[0] > 0:
                loss_b = tf.reduce_mean(loss_b_b)

            loss = loss_f + loss_b

        with tf.name_scope("Losses"):
            tf.summary.scalar("forward loss", loss_f, step=self.global_step)
            tf.summary.scalar("backword loss", loss_b, step=self.global_step)
            tf.summary.scalar("total loss", loss, step=self.global_step)

        with tf.name_scope("Misc"):
            tf.summary.scalar("learning rate", self.lr, step=self.global_step)

        self.global_step.assign_add(1)

        grad = tape.gradient(loss, self.model.mlp.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.mlp.trainable_variables))

    def normalize_inv_range(self, inv_range_khw1):
        inv_min = tf.reduce_min(inv_range_khw1, axis=(1, 2, 3), keepdims=True)
        inv_max = tf.reduce_max(inv_range_khw1, axis=(1, 2, 3), keepdims=True)
        return (inv_range_khw1 - inv_min) / (inv_max - inv_min)

    @tf.function
    def validate_step(self):
        inv_range_khw1 = self.model.render_inv_range(
            self.data.p.img_size, self.data.p.cam, self.data.data_dict["ref_T_cams"])
        tf.summary.image("rendered inverse range", self.normalize_inv_range(inv_range_khw1),
                         step=self.global_step)

    def run(self, debug: bool = False) -> None:
        train_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "train"))
        val_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "validation"))

        with val_writer.as_default():
            tf.summary.image("gt inverse range",
                             self.normalize_inv_range(self.data.data_dict["inv_range_imgs"]),
                             step=0)

        for epoch in range(self.p.num_epochs):
            with train_writer.as_default():
                for data_dict in self.data.dataset:
                    self.train_step(data_dict)

            with val_writer.as_default():
                self.validate_step()

            if not debug and epoch % self.p.save_freq == 0:
                self.model.mlp.save(os.path.join(self.ckpt_dir, f"epoch-{epoch}"))

if __name__ == "__main__":
    args = parse_args()
    params = ParamDict.from_file(args.params)

    trainer = Trainer(params, args.output)
    trainer.run()
