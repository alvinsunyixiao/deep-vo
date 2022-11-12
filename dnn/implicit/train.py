import argparse
import typing as T

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as tfk
if T.TYPE_CHECKING:
    from keras.api._v2 import keras as tfk

from dnn.implicit.data import PointLoader, T_DATA_DICT
from dnn.implicit.model import NeRD
from utils.params import ParamDict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", type=str, default="params.py",
                        help="path to the parameter file")
    return parser.parse_args()

class Trainer:

    DEFAULT_PARAMS = ParamDict(
        num_epochs=1000,
    )

    def __init__(self, params: ParamDict) -> None:
        self.p = params
        self.data = PointLoader(self.p.data)
        self.model = NeRD(self.p.model)

    def compute_loss(self, data_dict: T_DATA_DICT) -> tf.Tensor:
        # transform from cam to reference
        pos_cam_b3, dir_cam_b3, depth_gt_b = self.data.generate_samples(
            data_dict["points_cam_b3"], data_dict["directions_cam_b3"])

        # TODO(alvin): support not using GT here
        pos_ref_b3 = data_dict["ref_T_cam_b"] @ pos_cam_b3
        dir_ref_b3 = data_dict["ref_T_cam_b"].R @ dir_cam_b3

        # MLP inference
        mlp_input = self.model.input_encoding(pos_ref_b3, dir_ref_b3)
        depth_pred_b1 = self.model.mlp(mlp_input)
        depth_pred_b = depth_pred_b1[:, 0]

        # forward direction loss
        loss_f_b = tf.boolean_mask(depth_pred_b - depth_gt_b,
                                   depth_pred_b >= depth_gt_b)

        # backward direction loss
        points_ref_pred_b3 = pos_ref_b3 + dir_ref_b3 * depth_pred_b1
        # TODO(alvin): again, support not using GT here
        points_virtual_pred_b3 = data_dict["ref_T_virtual"].inv() @ points_ref_pred_b3
        depth_virtual_proj_b = tf.linalg.norm(points_virtual_pred_b3, axis=-1)
        pixels_uv_b2 = self.p.data.cam.project(points_virtual_pred_b3)

        # filter out invalid pixels
        valid_mask_b = (depth_virtual_proj_b >= self.p.data.min_depth) & \
                       (depth_virtual_proj_b <= self.p.data.max_depth) & \
                       (points_virtual_pred_b3[:, -1] > 0) & \
                       (pixels_uv_b2[:, 0] >= 0) & (pixels_uv_b2[:, 1] >= 0) & \
                       (pixels_uv_b2[:, 0] <= self.p.data.img_size[0] - 1.) & \
                       (pixels_uv_b2[:, 1] <= self.p.data.img_size[0] - 1.)
        pixels_uv_b2 = tf.boolean_mask(pixels_uv_b2, valid_mask_b)
        points_virtual_pred_b3 = tf.boolean_mask(points_virtual_pred_b3, valid_mask_b)
        depth_virtual_proj_b = tf.boolean_mask(depth_virtual_proj_b, valid_mask_b)

        # sample from depth image
        depth_sampled_b = tfa.image.resampler(data_dict["depth_virtual_hw1"][tf.newaxis],
                                              pixels_uv_b2[tf.newaxis])[0, :, 0]
        depth_sampled_b = tf.minimum(depth_sampled_b, self.p.data.max_depth)
        loss_b_b = tf.boolean_mask(depth_sampled_b - depth_virtual_proj_b,
                                   depth_sampled_b >= depth_virtual_proj_b)

        return loss_f_b, loss_b_b

    def train_step(self, data_dict: T_DATA_DICT, opt: tfk.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            loss_f_b, loss_b_b = self.compute_loss(data_dict)
            loss_f = tf.reduce_mean(loss_f_b)
            loss_b = tf.reduce_mean(loss_b_b)
            loss = loss_f + loss_b

        tf.print("Forward loss:", loss_f, "Backword loss:", loss_b, "Total loss:", loss)

        grad = tape.gradient(loss, self.model.mlp.trainable_variables)
        opt.apply_gradients(zip(grad, self.model.mlp.trainable_variables))

    def run(self) -> None:
        optimizer = tfk.optimizers.Adam(1e-3)
        for epoch in range(self.p.trainer.num_epochs):
            for data_dict in self.data.dataset:
                self.train_step(data_dict, optimizer)

if __name__ == "__main__":
    args = parse_args()
    params = ParamDict.from_file(args.params)
    Trainer(params).run()
