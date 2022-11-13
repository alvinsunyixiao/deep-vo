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
from utils.camera import PinholeCam
from utils.pose3d import Pose3D
from utils.params import ParamDict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", type=str, default="params.py",
                        help="path to the parameter file")
    return parser.parse_args()

class Trainer:

    DEFAULT_PARAMS = ParamDict(
        num_epochs=1000,
        save_freq=10,
    )

    def __init__(self, params: ParamDict, output_dir: str) -> None:
        self.p = params
        self.data = PointLoader(self.p.data)
        self.model = NeRD(self.p.model)
        self.optimizer = tfk.optimizers.Adam(1e-3)
        sess_dir = time.strftime("sess_%y-%m-%d_%H-%M-%S")
        self.ckpt_dir = os.path.join(output_dir, sess_dir, "ckpts")
        self.log_dir = os.path.join(output_dir, sess_dir, "logs")
        self.train_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "train"))
        self.val_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "validation"))
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def warp_loss(self,
        inv_range_src_bhw1: tf.Tensor,
        inv_range_tgt_bhw1: tf.Tensor,
        src_T_tgt_b: Pose3D
    ) -> tf.Tensor:
        cam: PinholeCam = self.p.data.cam
        grid_bhw2 = cam.compute_grid(inv_range_src_bhw1)
        unit_ray_bhw3 = cam.unit_ray(grid_bhw2)

        src_T_tgt_b11 = tf.expand_dims(tf.expand_dims(src_T_tgt_b, -1), -1)
        src_T_tgt_bhw = tf.broadcast_to(src_T_tgt_b11, tf.shape(inv_range_src_bhw1)[:-1])

        # project from target frame to source frame
        points_tgt_bhw3 = unit_ray_bhw3 / tf.maximum(inv_range_tgt_bhw1, 1e-3)
        points_src_bhw3 = src_T_tgt_bhw @ points_tgt_bhw3
        pixel_src_bhw2 = cam.project(points_src_bhw3)

        # compute projected and sampled inverse range
        inv_range_computed_bhw = 1. / tf.linalg.norm(points_src_bhw3, axis=-1)
        inv_range_sampled_bhw = tfa.image.resampler(inv_range_src_bhw1, pixel_src_bhw2)[..., 0]
        loss_bhw = tf.square(inv_range_computed_bhw - inv_range_sampled_bhw)

        valid_mask_bhw = (pixel_src_bhw2[..., 0] >= 0.) & \
                         (pixel_src_bhw2[..., 0] <= self.p.data.img_size[0] - 1.) & \
                         (pixel_src_bhw2[..., 1] >= 0.) & \
                         (pixel_src_bhw2[..., 1] <= self.p.data.img_size[1] - 1.) & \
                         (points_src_bhw3[..., -1] >= self.p.data.min_depth) & \
                         (inv_range_computed_bhw >= inv_range_sampled_bhw)


        return tf.boolean_mask(loss_bhw, valid_mask_bhw)


    def compute_loss(self, data_dict: T_DATA_DICT) -> tf.Tensor:
        # TODO(alvin): support not using GT here
        ref_T_virtual_b = data_dict["ref_T_cam_b"] @ data_dict["cam_T_virtual_b"]
        rendered_inv_render_bhw1 = self.model.render_inv_range(
            self.p.data.img_size, self.p.data.cam, ref_T_virtual_b)

        loss_f_k = self.warp_loss(data_dict["inv_range_imgs_bhw1"], rendered_inv_render_bhw1, data_dict["cam_T_virtual_b"])
        loss_b_k = self.warp_loss(rendered_inv_render_bhw1, data_dict["inv_range_imgs_bhw1"], data_dict["cam_T_virtual_b"].inv())

        return loss_f_k, loss_b_k

    @tf.function
    def train_step(self, data_dict: T_DATA_DICT):
        with tf.GradientTape() as tape:
            loss_f_k, loss_b_k = self.compute_loss(data_dict)

            loss_f = 0.
            loss_b = 0.
            if tf.shape(loss_f_k)[0] > 0:
                loss_f = tf.reduce_mean(loss_f_k)
            if tf.shape(loss_b_k)[0] > 0:
                loss_b = tf.reduce_mean(loss_b_k)

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
            self.p.data.img_size, self.p.data.cam, Pose3D.identity())
        inv_min = tf.reduce_min(inv_range)
        inv_max = tf.reduce_max(inv_range)
        inv_range_norm = (inv_range - inv_min) / (inv_max - inv_min)
        tf.summary.image("rendered inverse range", inv_range_norm[tf.newaxis],
                         step=self.global_step)

    def run(self, debug: bool = False) -> None:
        for epoch in range(self.p.trainer.num_epochs):
            with self.train_writer.as_default():
                for data_dict in self.data.dataset:
                    self.train_step(data_dict)

            with self.val_writer.as_default():
                self.validate_step()

            if not debug and epoch % self.p.trainer.save_freq == 0:
                self.model.mlp.save(os.path.join(self.ckpt_dir, f"epoch-{epoch}"))

if __name__ == "__main__":
    args = parse_args()
    params = ParamDict.from_file(args.params)

    trainer = Trainer(params)
    trainer.run()
