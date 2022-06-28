import functools
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import typing as T

from utils.params import ParamDict
from utils.camera import PinholeCam
from utils.pose3d import Pose3D

class LossManager:

    DEFAULT_PARAMS = ParamDict(
        weights = ParamDict(
            img = 1.,
            smooth = 1e-2,
        ),
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params

    def ssim(self,
        img1_bhw3: tf.Tensor,
        img2_bhw3: tf.Tensor,
        max_val: float = 1.,
        filter_size: int = 7,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> tf.Tensor:
        c1 = (k1 * max_val)**2
        c2 = (k2 * max_val)**2
        reducer = functools.partial(tfa.image.gaussian_filter2d,
                                    filter_shape=filter_size, sigma=filter_sigma)

        # SSIM luminance
        mean1_bhw3 = reducer(img1_bhw3)
        mean2_bhw3 = reducer(img2_bhw3)
        num1_bhw3 = mean1_bhw3 * mean2_bhw3 * 2.0
        den1_bhw3 = tf.square(mean1_bhw3) + tf.square(mean2_bhw3)
        luminance_bhw3 = (num1_bhw3 + c1) / (den1_bhw3 + c1)

        # SSIM contrast-structure
        num2_bhw3 = reducer(img1_bhw3 * img2_bhw3) * 2.0
        den2_bhw3 = reducer(tf.square(img1_bhw3) + tf.square(img2_bhw3))
        # TODO(alvin): c2 *= compensation -> use unbiased estimator
        cs_bhw3 = (num2_bhw3 - num1_bhw3 + c2) / (den2_bhw3 - den1_bhw3 + c2)

        return luminance_bhw3 * cs_bhw3

    def photometric_loss(self, img1_bhw3: tf.Tensor, img2_bhw3: tf.Tensor):
        img_l1_bhw3 = tf.abs(img1_bhw3 - img2_bhw3)
        img_l1_bhw3 = tf.clip_by_value(img_l1_bhw3, 0., 1.)
        img_ssim_bhw3 = self.ssim(img1_bhw3, img2_bhw3)
        img_loss_bhw3 = .15 * img_l1_bhw3 + .85 * (1. - img_ssim_bhw3) / 2.

        return self.p.weights.img * img_loss_bhw3

    def warp_loss(self,
        img_tgt_bhw3: tf.Tensor,
        depth_tgt_bhw1: tf.Tensor,
        img_src_bhw3: tf.Tensor,
        src_T_tgt_b: Pose3D,
        cam_b: PinholeCam,
    ) -> T.Tuple[tf.Tensor, tf.Tensor]:
        # warp from src to tgt
        src_T_tgt_b11 = src_T_tgt_b[:, tf.newaxis, tf.newaxis]
        cam_b11 = cam_b[:, tf.newaxis, tf.newaxis]
        pixel_src_bhw2, _ = cam_b11.reproject(depth_tgt_bhw1, src_T_tgt_b11)
        img_proj_bhw3 = tfa.image.resampler(img_src_bhw3, pixel_src_bhw2)

        # photometric loss
        return self.photometric_loss(img_proj_bhw3, img_tgt_bhw3)

    def smooth_loss(self, disp_bhw1: tf.Tensor, img_bhw3: tf.Tensor):
        # normalize
        mean_disp_b111 = tf.reduce_mean(disp_bhw1, axis=(1, 2), keepdims=True)
        norm_disp_bhw1 = tf.math.divide_no_nan(disp_bhw1, mean_disp_b111)

        grad_disp_x_bhw1 = tf.abs(norm_disp_bhw1[:, :, :-1, :] - norm_disp_bhw1[:, :, 1:, :])
        grad_disp_y_bhw1 = tf.abs(norm_disp_bhw1[:, :-1, :, :] - norm_disp_bhw1[:, 1:, :, :])

        grad_img_x_bhw3 = tf.abs(img_bhw3[:, :, :-1, :] - img_bhw3[:, :, 1:, :])
        grad_img_y_bhw3 = tf.abs(img_bhw3[:, :-1, :, :] - img_bhw3[:, 1:, :, :])

        grad_disp_x_bhw1 *= tf.exp(-tf.reduce_mean(grad_img_x_bhw3, axis=-1, keepdims=True))
        grad_disp_y_bhw1 *= tf.exp(-tf.reduce_mean(grad_img_y_bhw3, axis=-1, keepdims=True))

        loss = tf.reduce_mean(grad_disp_x_bhw1) + tf.reduce_mean(grad_disp_y_bhw1)

        return loss * self.p.weights.smooth
