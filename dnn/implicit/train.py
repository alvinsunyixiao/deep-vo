import argparse
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
        save_freq=5,
    )

    def __init__(self, params: ParamDict, output_dir: str) -> None:
        self.p = params.trainer
        self.data = PointLoader(params.data)
        self.model = NeRD(params.model)
        self.optimizer = tfk.optimizers.Adam(1e-3)
        sess_dir = time.strftime("sess_%y-%m-%d_%H-%M-%S")
        self.ckpt_dir = os.path.join(output_dir, sess_dir, "ckpts")
        self.log_dir = os.path.join(output_dir, sess_dir, "logs")
        self.train_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "train"))
        self.val_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "validation"))
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def compute_loss(self, data_dict: T_DATA_DICT) -> tf.Tensor:
        # transform from cam to reference
        pos_cam_b3, dir_cam_b3, inv_range_gt_b = self.data.generate_samples(
            data_dict["points_cam_b3"], data_dict["directions_cam_b3"])

        # TODO(alvin): support not using GT here
        pos_ref_b3 = data_dict["ref_T_cam_b"] @ pos_cam_b3
        dir_ref_b3 = data_dict["ref_T_cam_b"].R @ dir_cam_b3

        # MLP inference
        mlp_input = self.model.input_encoding(pos_ref_b3, dir_ref_b3)
        inv_range_pred_b1 = tf.maximum(self.model.mlp(mlp_input), 1e-3)
        inv_range_pred_b = inv_range_pred_b1[:, 0]

        # forward direction loss
        loss_f_b = tf.boolean_mask(tf.square(inv_range_pred_b - inv_range_gt_b),
                                   inv_range_pred_b <= inv_range_gt_b)

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

    @tf.function
    def train_step(self, data_dict: T_DATA_DICT):
        with tf.GradientTape() as tape:
            loss_f_b, loss_b_b = self.compute_loss(data_dict)
            loss_f = tf.reduce_mean(loss_f_b)
            loss_b = tf.reduce_mean(loss_b_b)
            loss = loss_f + loss_b

        tf.summary.scalar("forward loss", loss_f, step=self.global_step)
        tf.summary.scalar("backword loss", loss_b, step=self.global_step)
        tf.summary.scalar("total loss", loss, step=self.global_step)
        self.global_step.assign_add(1)

        grad = tape.gradient(loss, self.model.mlp.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.mlp.trainable_variables))

    @tf.function
    def validate_step(self):
        inv_range = self.model.render_inv_range(
            self.data.p.img_size, self.data.p.cam, Pose3D.identity())
        inv_min = tf.reduce_min(inv_range)
        inv_max = tf.reduce_max(inv_range)
        inv_range_norm = (inv_range - inv_min) / (inv_max - inv_min)
        tf.summary.image("rendered inverse range", inv_range_norm[tf.newaxis],
                         step=self.global_step)

    def run(self, debug: bool = False) -> None:
        for epoch in range(self.p.num_epochs):
            with self.train_writer.as_default():
                for data_dict in self.data.dataset:
                    self.train_step(data_dict)

            with self.val_writer.as_default():
                self.validate_step()

            if not debug and epoch % self.p.save_freq == 0:
                self.model.mlp.save(os.path.join(self.ckpt_dir, f"epoch-{epoch}"))

if __name__ == "__main__":
    args = parse_args()
    params = ParamDict.from_file(args.params)

    trainer = Trainer(params, args.output)
    trainer.run()
