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
        lr=ParamDict(
            init=1e-3,
            end=1e-5,
            delay=50000,
        ),
    )

    def __init__(self, params: ParamDict) -> None:
        self.p = params.trainer
        self.data = PointLoader(params.data)
        self.model = NeRD(params.model)

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
        cos_dir_diff_b = tf.einsum("ij,ij->i", dir_cam_b3, data_dict["directions_cam_b3"])
        cos_dir_diff_b = tf.clip_by_value(cos_dir_diff_b, -1., 1.)
        loss_f_b = tf.boolean_mask(tf.square(inv_range_pred_b - inv_range_gt_b),
            (inv_range_pred_b <= inv_range_gt_b) & \
            (tf.math.acos(cos_dir_diff_b) <= math.radians(30)))

        # backward direction loss
        points_ref_pred_b3 = pos_ref_b3 + dir_ref_b3 / inv_range_pred_b1
        # TODO(alvin): again, support not using GT here
        points_virtual_pred_b3 = data_dict["ref_T_virtual"].inv() @ points_ref_pred_b3
        points_virtual_pred_b3 = tf.boolean_mask(points_virtual_pred_b3,
            points_virtual_pred_b3[..., -1] >= self.data.p.min_depth)
        depth_virtual_proj_b = tf.linalg.norm(points_virtual_pred_b3, axis=-1)
        pixels_uv_b2 = self.data.p.cam.project(points_virtual_pred_b3)

        # filter out invalid pixels
        valid_mask_b = (pixels_uv_b2[:, 0] >= 0) & (pixels_uv_b2[:, 1] >= 0) & \
                       (pixels_uv_b2[:, 0] <= self.data.p.img_size[0] - 1.) & \
                       (pixels_uv_b2[:, 1] <= self.data.p.img_size[1] - 1.)
        pixels_uv_b2 = tf.boolean_mask(pixels_uv_b2, valid_mask_b)
        points_virtual_pred_b3 = tf.boolean_mask(points_virtual_pred_b3, valid_mask_b)
        depth_virtual_proj_b = tf.boolean_mask(depth_virtual_proj_b, valid_mask_b)
        inv_range_proj_b = 1. / depth_virtual_proj_b

        # bilinear interpolate from depth image
        pixels_lower_x_b = tf.math.floor(pixels_uv_b2[:, 0])
        pixels_lower_y_b = tf.math.floor(pixels_uv_b2[:, 1])
        pixels_upper_x_b = tf.math.ceil(pixels_uv_b2[:, 0])
        pixels_upper_y_b = tf.math.ceil(pixels_uv_b2[:, 1])

        pixels_tl_b2 = tf.stack([pixels_lower_y_b, pixels_lower_x_b], axis=-1)
        pixels_tr_b2 = tf.stack([pixels_lower_y_b, pixels_upper_x_b], axis=-1)
        pixels_bl_b2 = tf.stack([pixels_upper_y_b, pixels_lower_x_b], axis=-1)
        pixels_br_b2 = tf.stack([pixels_upper_y_b, pixels_upper_x_b], axis=-1)

        inv_range_tl_b = tf.gather_nd(data_dict["inv_range_virtual_hw1"],
                                      tf.cast(pixels_tl_b2, tf.int32))[:, 0]
        inv_range_tr_b = tf.gather_nd(data_dict["inv_range_virtual_hw1"],
                                      tf.cast(pixels_tr_b2, tf.int32))[:, 0]
        inv_range_bl_b = tf.gather_nd(data_dict["inv_range_virtual_hw1"],
                                      tf.cast(pixels_bl_b2, tf.int32))[:, 0]
        inv_range_br_b = tf.gather_nd(data_dict["inv_range_virtual_hw1"],
                                      tf.cast(pixels_br_b2, tf.int32))[:, 0]

        x_offset_b = pixels_uv_b2[:, 0] - pixels_lower_x_b
        y_offset_b = pixels_uv_b2[:, 1] - pixels_lower_y_b
        inv_range_t_b = (1. - x_offset_b) * inv_range_tl_b + x_offset_b * inv_range_tr_b
        inv_range_b_b = (1. - x_offset_b) * inv_range_bl_b + x_offset_b * inv_range_br_b
        inv_range_interp_b = (1. - y_offset_b) * inv_range_t_b + y_offset_b * inv_range_b_b

        loss_b_b = tf.boolean_mask(tf.square(inv_range_proj_b - inv_range_interp_b),
                                   inv_range_interp_b <= inv_range_proj_b)

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

    @tf.function(jit_compile=True)
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

        self.global_step.assign_add(1)

        grad = tape.gradient(loss, self.model.mlp.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.mlp.trainable_variables))

        return loss, loss_f, loss_b

    def normalize_inv_range(self, inv_range_khw1):
        inv_min = tf.reduce_min(inv_range_khw1, axis=(1, 2, 3), keepdims=True)
        inv_max = tf.reduce_max(inv_range_khw1, axis=(1, 2, 3), keepdims=True)
        return (inv_range_khw1 - inv_min) / (inv_max - inv_min)

    @tf.function(jit_compile=True)
    def validate_step(self):
        inv_range_khw1 = self.model.render_inv_range(
            self.data.p.img_size, self.data.p.cam, self.data.data_dict["ref_T_cams"])
        return self.normalize_inv_range(inv_range_khw1)

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
                    loss, loss_f, loss_b = self.train_step(data_dict)

                    with tf.name_scope("Losses"):
                        tf.summary.scalar("forward loss", loss_f, step=self.global_step)
                        tf.summary.scalar("backword loss", loss_b, step=self.global_step)
                        tf.summary.scalar("total loss", loss, step=self.global_step)

                    with tf.name_scope("Misc"):
                        tf.summary.scalar("learning rate", self.lr, step=self.global_step)

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
