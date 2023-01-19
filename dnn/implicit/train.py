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
        estimate_pose=True,
        # set to None to disable frequency band masking
        freq_mask=ParamDict(
            start=100,
            end=600,
        ),
        loss=ParamDict(
            max_angle_diff=30.,
        ),
    )

    def __init__(self, params: ParamDict) -> None:
        self.p = params.trainer
        self.data = PointLoader(params.data)
        self.model = NeRD(params.model)

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.lr = tf.Variable(0., trainable=False, dtype=tf.float32)
        self.optimizer = tfk.optimizers.Adam(self.lr)

        self.ref_T_cam_k = self.data.data_dict["ref_T_cams"][:self.data.p.num_images] @ \
                           Pose3D.random(max_angle=math.radians(30),
                                         max_translate=.5,
                                         size=(self.data.p.num_images,))

    def compute_loss(self, data_dict: T_DATA_DICT, ref_T_cam_k: Pose3D) -> T_DATA_DICT:
        # transform from cam to reference
        pos_cam_b3, dir_cam_b3, inv_range_gt_b = self.data.generate_samples(
            data_dict["points_cam_b3"])

        if self.p.estimate_pose:
            ref_T_cam_b = tf.gather(ref_T_cam_k, data_dict["img_idx_b"])
            ref_T_virtual = tf.gather(ref_T_cam_k, data_dict["virtual_idx"])
        else:
            ref_T_cam_b = data_dict["ref_T_cam_b"]
            ref_T_virtual = data_dict["ref_T_virtual"]

        pos_ref_b3 = ref_T_cam_b @ pos_cam_b3
        dir_ref_b3 = ref_T_cam_b.R @ dir_cam_b3

        # MLP inference
        mlp_input = self.model.input_encoding(pos_ref_b3, dir_ref_b3)
        inv_range_pred_b1 = tf.maximum(self.model.mlp(mlp_input), 1e-3) # max range 1000 meters
        inv_range_pred_b = inv_range_pred_b1[:, 0]

        # forward direction loss
        cos_dir_diff_b = tf.einsum("ij,ij->i", dir_cam_b3, data_dict["directions_cam_b3"])
        cos_dir_diff_b = tf.clip_by_value(cos_dir_diff_b, -1., 1.)
        forward_mask_b = (inv_range_pred_b <= inv_range_gt_b) & \
                         (tf.math.acos(cos_dir_diff_b) <= math.radians(self.p.loss.max_angle_diff))
        loss_f_b = tf.boolean_mask(tf.square(inv_range_pred_b - inv_range_gt_b), forward_mask_b)

        # backward direction loss
        points_ref_pred_b3 = pos_ref_b3 + dir_ref_b3 / inv_range_pred_b1
        points_virtual_pred_b3 = ref_T_virtual.inv() @ points_ref_pred_b3
        points_virtual_pred_b3 = tf.boolean_mask(points_virtual_pred_b3, ~forward_mask_b)
        points_virtual_pred_b3 = tf.boolean_mask(points_virtual_pred_b3,
            points_virtual_pred_b3[..., -1] >= self.data.p.min_depth)
        pixels_uv_b2 = self.data.p.cam.project(points_virtual_pred_b3)

        # filter out invalid pixels
        valid_mask_b = (pixels_uv_b2[:, 0] >= 0) & (pixels_uv_b2[:, 1] >= 0) & \
                       (pixels_uv_b2[:, 0] <= self.data.p.img_size[0] - 1.) & \
                       (pixels_uv_b2[:, 1] <= self.data.p.img_size[1] - 1.)
        pixels_uv_b2 = tf.boolean_mask(pixels_uv_b2, valid_mask_b)
        points_virtual_pred_b3 = tf.boolean_mask(points_virtual_pred_b3, valid_mask_b)
        depth_virtual_proj_b = tf.linalg.norm(points_virtual_pred_b3, axis=-1)
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

        loss_f = 0.
        loss_b = 0.
        loss_f_size = tf.shape(loss_f_b)[0]
        loss_b_size = tf.shape(loss_b_b)[0]
        if loss_f_size > 0:
            loss_f = tf.reduce_mean(loss_f_b)
        if loss_b_size > 0:
            loss_b = tf.reduce_mean(loss_b_b)

        # pose loss for logging
        cams_T_cams_pred = self.data.data_dict["ref_T_cams"][:self.data.p.num_images].inv() @ \
                           ref_T_cam_k
        pose_delta = cams_T_cams_pred.to_se3()

        return {
            "loss": loss_f + loss_b,
            "loss_f": loss_f,
            "loss_b": loss_b,
            "loss_f_size": loss_f_size,
            "loss_b_size": loss_b_size,
            "loss_pose_R": tf.reduce_mean(tf.square(pose_delta[..., :3])),
            "loss_pose_t": tf.reduce_mean(tf.square(pose_delta[..., 3:])),
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

    def freq_alpha_schedule(self, epoch: int) -> tf.Tensor:
        if self.p.freq_mask is None:
            return 1.

        alpha = (epoch - self.p.freq_mask.start) / (self.p.freq_mask.end - self.p.freq_mask.start)
        return tf.clip_by_value(alpha, 0., 1.)

    @tf.function(jit_compile=True)
    def train_step(self, data_dict: T_DATA_DICT, ref_T_cam_k: Pose3D) -> T_DATA_DICT:
        self.lr.assign(self.lr_schedule(self.global_step))

        with tf.GradientTape() as tape:
            ref_T_cam_k.watch(tape)
            meta_dict = self.compute_loss(data_dict, ref_T_cam_k)

        self.global_step.assign_add(1)

        variables = {
            "model": self.model.mlp.trainable_variables,
            "pose_R": ref_T_cam_k.R.quat,
            "pose_t": ref_T_cam_k.t,
        }
        grad = tape.gradient(meta_dict["loss"], variables)
        self.optimizer.apply_gradients(zip(grad["model"], variables["model"]))

        # update pose parameter
        grad_R_tanget = tf.expand_dims(grad["pose_R"], axis=-2) @ \
                        ref_T_cam_k.R.storage_D_tangent()
        grad_R_tanget = grad_R_tanget[..., 0, :]
        grad_t = tf.convert_to_tensor(grad["pose_t"])
        grad_se3 = tf.concat([grad_R_tanget, grad_t], axis=-1)
        #TODO(alvin): make pose lr a variable
        ref_T_cam_new_k = ref_T_cam_k @ Pose3D.from_se3(-grad_se3 * self.lr * 1e2)

        meta_dict.update(data_dict,
            grad_model=grad["model"],
            grad_R=grad_R_tanget,
            grad_t=grad_t,
            ref_T_cam_k=ref_T_cam_new_k
        )
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
            tf.summary.scalar("forward loss", meta_dict["loss_f"], step=self.global_step)
            tf.summary.scalar("backword loss", meta_dict["loss_b"], step=self.global_step)
            tf.summary.scalar("total loss", meta_dict["loss"], step=self.global_step)
            tf.summary.scalar("pose R loss", meta_dict["loss_pose_R"], step=self.global_step)
            tf.summary.scalar("pose t loss", meta_dict["loss_pose_t"], step=self.global_step)

        with tf.name_scope("stats"):
            tf.summary.scalar("forward count", meta_dict["loss_f_size"], step=self.global_step)
            tf.summary.scalar("backward count", meta_dict["loss_b_size"], step=self.global_step)

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
            with tf.name_scope("weights"):
                for var, grad in zip(self.model.mlp.trainable_variables, meta_dict["grad_model"]):
                    tf.summary.histogram(var.name, grad, step=self.global_step)
            with tf.name_scope("pose"):
                tf.summary.histogram("R", meta_dict["grad_R"], step=self.global_step)
                tf.summary.histogram("t", meta_dict["grad_t"], step=self.global_step)

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
                freq_alpha = self.freq_alpha_schedule(epoch)
                tf.summary.scalar("freq_alpha", freq_alpha, step=self.global_step)
                self.model.set_freq_alpha(freq_alpha)

                for data_dict in tqdm(self.data.dataset, desc="Iteration", leave=False):
                    meta_dict = self.train_step(data_dict, self.ref_T_cam_k)
                    self.ref_T_cam_k = meta_dict["ref_T_cam_k"]
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
