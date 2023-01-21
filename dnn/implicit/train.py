from utils.tf_utils import set_tf_memory_growth
set_tf_memory_growth(True)

import argparse
import math
import os
import time
import typing as T

import numpy as np
import open3d as o3d
import tensorflow as tf
import tensorflow.keras as tfk
if T.TYPE_CHECKING:
    from keras.api._v2 import keras as tfk

from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from tqdm import tqdm, trange

from dnn.implicit.data import PointLoader, T_DATA_DICT
from dnn.implicit.model import NeRD
from utils.pose3d import Pose3D, Rot3D, Pose3DVar
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

class PseudoSE3Adam(tf.Module):
    def __init__(self,
        lr: T.Union[float, tf.Variable, tf.Tensor] = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.lr = lr
        if not isinstance(lr, tf.Variable):
            self.lr = tf.Variable(lr, trainable=False)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.is_built = False

    def apply_gradients(self, quat: tf.Variable, grad_R: tf.Tensor) -> None:
        if not self.is_built:
            self.mt = tf.Variable(tf.zeros_like(grad_R), trainable=False)
            self.vt = tf.Variable(tf.zeros_like(grad_R), trainable=False)
            self.t = tf.Variable(0.0, trainable=False)
            self.is_built = True

        self.t.assign_add(1.)
        self.mt.assign(self.beta1 * self.mt + (1 - self.beta1) * grad_R)
        self.vt.assign(self.beta2 * self.vt + (1 - self.beta2) * tf.square(grad_R))
        mt_hat = self.mt / (1. - tf.pow(self.beta1, self.t))
        vt_hat = self.vt / (1. - tf.pow(self.beta2, self.t))

        R_old = Rot3D(quat)
        R_new = R_old @ Rot3D.from_so3(-self.lr * mt_hat / (tf.sqrt(vt_hat) + self.eps))
        quat.assign(R_new.quat)

    def set_lr(self, lr: float) -> None:
        self.lr.assign(lr)

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
        #TODO(alvin): make pose lr a seperate variable
        self.pose_optimizer = PseudoSE3Adam(self.lr)

        self.ref_T_cam_k = Pose3DVar(Pose3D.identity(size=(self.data.p.num_images - 1,)))

        # generate ground truth poses
        self.cam_intrinsics = self.data.p.cam.to_matrix().numpy()
        self.cam_poses_gt = o3d.geometry.LineSet()
        for i in range(self.data.p.num_images):
            cam = o3d.geometry.LineSet.create_camera_visualization(
                self.data.p.img_size[0], self.data.p.img_size[1],
                self.cam_intrinsics, self.data.data_dict["ref_T_cams"][i].inv().to_matrix().numpy(),
            )
            cam.paint_uniform_color((0.0, 1.0, 0.0))
            self.cam_poses_gt += cam

    def compute_loss(self, data_dict: T_DATA_DICT) -> T_DATA_DICT:
        # transform from cam to reference
        pos_cam_b3, dir_cam_b3, range_gt_b = self.data.generate_samples(
            data_dict["points_cam_b3"])

        if self.p.estimate_pose:
            ref_T_cam_k = tf.concat([Pose3D.identity((1,)), self.ref_T_cam_k.to_Pose3D()], axis=0)
            ref_T_cam_b = tf.gather(ref_T_cam_k, data_dict["img_idx_b"])
        else:
            ref_T_cam_b = data_dict["ref_T_cam_b"]

        pos_ref_b3 = ref_T_cam_b @ pos_cam_b3
        dir_ref_b3 = ref_T_cam_b.R @ dir_cam_b3

        # MLP inference
        mlp_input = self.model.input_encoding(pos_ref_b3, dir_ref_b3)
        range_pred_b1 = self.model.mlp(mlp_input)
        range_pred_b = range_pred_b1[:, 0]

        #cos_dir_diff_b = tf.einsum("ij,ij->i", dir_cam_b3, data_dict["directions_cam_b3"])
        #cos_dir_diff_b = tf.clip_by_value(cos_dir_diff_b, -1., 1.)
        #forward_mask_b = (tf.math.acos(cos_dir_diff_b) <= math.radians(self.p.loss.max_angle_diff))
        loss_b = tf.abs(range_pred_b - range_gt_b) / range_gt_b

        # pose loss for logging
        cams_T_cams_pred = self.data.data_dict["ref_T_cams"][1:self.data.p.num_images].inv() @ \
                           self.ref_T_cam_k.to_Pose3D()
        pose_delta = cams_T_cams_pred.to_se3()

        return {
            "loss": tf.reduce_mean(loss_b),
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

    @tf.function(jit_compile=False)
    def train_step(self, data_dict: T_DATA_DICT) -> T_DATA_DICT:
        self.lr.assign(self.lr_schedule(self.global_step))

        with tf.GradientTape() as tape:
            meta_dict = self.compute_loss(data_dict)

        self.global_step.assign_add(1)

        variables = {
            "normal": self.model.mlp.trainable_variables + [self.ref_T_cam_k.t],
            "quat": self.ref_T_cam_k.quat,
        }
        grad = tape.gradient(meta_dict["loss"], variables)
        self.optimizer.apply_gradients(zip(grad["normal"], variables["normal"]))
        meta_dict.update(data_dict, grad_model=grad["normal"])

        # update pose parameter
        if self.p.estimate_pose:
            ref_T_cam_k = self.ref_T_cam_k.to_Pose3D()

            grad_R_tanget = tf.expand_dims(grad["quat"], axis=-2) @ \
                            ref_T_cam_k.R.storage_D_tangent()
            grad_R_tanget = tf.squeeze(grad_R_tanget, axis=-2)
            self.pose_optimizer.apply_gradients(self.ref_T_cam_k.quat, grad_R_tanget)

            meta_dict.update(
                grad_R=grad_R_tanget,
                grad_t=grad["normal"][-1],
            )

        return meta_dict

    def normalize_inv_range(self, inv_range_khw1: tf.Tensor) -> tf.Tensor:
        inv_min = tf.reduce_min(inv_range_khw1, axis=(1, 2, 3), keepdims=True)
        inv_max = tf.reduce_max(inv_range_khw1, axis=(1, 2, 3), keepdims=True)
        inv_max = tf.minimum(inv_max, 4.) # minimum range is 1/4

        inv_range_norm_khw1 = (inv_range_khw1 - inv_min) / (inv_max - inv_min)
        inv_range_norm_khw1 = tf.clip_by_value(inv_range_norm_khw1, 0., 1.)

        return inv_range_norm_khw1

    @tf.function(jit_compile=True)
    def validate_step(self) -> tf.Tensor:
        #if self.p.estimate_pose:
        #    ref_T_cam_k = self.ref_T_cam_k.to_Pose3D()
        #else:
        #    ref_T_cam_k = self.data.data_dict["ref_T_cams"][:self.data.p.num_images]

        ref_T_cam_k = self.data.data_dict["ref_T_cams"][:self.data.p.num_images]

        inv_range_khw1 = self.model.render_inv_range(
            self.data.p.img_size, self.data.p.cam, ref_T_cam_k)
        return self.normalize_inv_range(inv_range_khw1)

    @tf.function
    def log_step(self, meta_dict: T_DATA_DICT) -> None:
        with tf.name_scope("losses"):
            tf.summary.scalar("L1 loss", meta_dict["loss"], step=self.global_step)
            tf.summary.scalar("pose R loss", meta_dict["loss_pose_R"], step=self.global_step)
            tf.summary.scalar("pose t loss", meta_dict["loss_pose_t"], step=self.global_step)

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
            if self.p.estimate_pose:
                with tf.name_scope("pose"):
                    tf.summary.histogram("R", meta_dict["grad_R"], step=self.global_step)
                    tf.summary.histogram("t", meta_dict["grad_t"], step=self.global_step)

    def pose_visualization_step(self, logdir: str, epoch: int) -> None:
        ref_T_cam_k = self.ref_T_cam_k.to_Pose3D()
        cam_poses_pred = o3d.geometry.LineSet()
        for i in range(1, self.data.p.num_images):
            cam = o3d.geometry.LineSet.create_camera_visualization(
                self.data.p.img_size[0], self.data.p.img_size[1],
                self.cam_intrinsics, ref_T_cam_k[i-1].inv().to_matrix()
            )
            cam.paint_uniform_color((1.0, 0.0, 0.0))
            cam_poses_pred += cam

        summary.add_3d("pose gt", to_dict_batch([self.cam_poses_gt]),
                       step=self.epoch, logdir=logdir)
        summary.add_3d("pose pred", to_dict_batch([cam_poses_pred]),
                       step=self.epoch, logdir=logdir)

        # generate correspondence lines
        corr = o3d.geometry.LineSet()
        points_gt = self.data.data_dict["ref_T_cams"][1:self.data.p.num_images].t.numpy()
        points_pred = ref_T_cam_k.t.numpy()
        corr.points = o3d.utility.Vector3dVector(np.concatenate([points_gt, points_pred], axis=0))
        corr.lines = o3d.utility.Vector2iVector(
            np.column_stack([np.arange(points_pred.shape[0], dtype=int),
                             np.arange(points_pred.shape[0], dtype=int) + points_pred.shape[0]]))
        corr.paint_uniform_color((0.0, 0.0, 1.0))
        summary.add_3d("correspondence", to_dict_batch([corr]), step=epoch, logdir=logdir)

    def run(self, output_dir: str, weights: T.Optional[str] = None, debug: bool = False) -> None:
        sess_dir = time.strftime("sess_%y-%m-%d_%H-%M-%S")
        ckpt_dir = os.path.join(output_dir, sess_dir, "ckpts")
        log_dir = os.path.join(output_dir, sess_dir, "logs")
        pose_dir = os.path.join(log_dir, "pose")

        train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
        val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "validation"))
        pose_writer = tf.summary.create_file_writer(pose_dir)

        if weights is not None:
            self.model.load_weights(os.path.join(weights, "model"))
            ckpt = tf.train.Checkpoint(self.ref_T_cam_k)
            ckpt.read(os.path.join(weights, "pose"))

        with val_writer.as_default():
            tf.summary.image("gt inverse range",
                             self.normalize_inv_range(1. / self.data.data_dict["range_imgs"]),
                             step=0,
                             max_outputs=6)

        for epoch in trange(self.p.num_epochs, desc="Epoch"):
            with pose_writer.as_default():
                self.pose_visualization_step(pose_dir, epoch)

            with train_writer.as_default():
                freq_alpha = self.freq_alpha_schedule(epoch)
                tf.summary.scalar("freq_alpha", freq_alpha, step=self.global_step)
                self.model.set_freq_alpha(freq_alpha)

                for data_dict in tqdm(self.data.dataset, desc="Iteration", leave=False):
                    meta_dict = self.train_step(data_dict)

                    if self.global_step % self.p.log_freq == 0:
                        self.log_step(meta_dict)

            inv_range_norm_khw1 = self.validate_step()
            with val_writer.as_default():
                tf.summary.image("rendered inverse range",
                    inv_range_norm_khw1, step=self.global_step, max_outputs=6)

            if not debug and epoch % self.p.save_freq == self.p.save_freq - 1:
                epoch_dir = os.path.join(ckpt_dir, f"epoch-{epoch+1}")
                # save mlp
                self.model.save(os.path.join(epoch_dir, "model"))

                # save pose
                ckpt = tf.train.Checkpoint(self.ref_T_cam_k)
                ckpt.write(os.path.join(epoch_dir, "pose"))

if __name__ == "__main__":
    args = parse_args()
    params = ParamDict.from_file(args.params)

    trainer = Trainer(params)
    trainer.run(args.output, args.weights)
