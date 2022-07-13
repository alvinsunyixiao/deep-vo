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
            smooth = 1e-3,
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
        depth_src_bhw1: tf.Tensor,
        src_T_tgt_b: Pose3D,
        cam_b: PinholeCam,
    ) -> tf.Tensor:
        # warp from src to tgt
        src_T_tgt_b11 = src_T_tgt_b[:, tf.newaxis, tf.newaxis]
        cam_b11 = cam_b[:, tf.newaxis, tf.newaxis]
        pixel_src_bhw2, depth_computed_bhw1 = cam_b11.reproject(depth_tgt_bhw1, src_T_tgt_b11)

        img_proj_bhw3 = tfa.image.resampler(img_src_bhw3, pixel_src_bhw2)
        depth_proj_bhw1 = tfa.image.resampler(depth_src_bhw1, pixel_src_bhw2)

        h = tf.shape(img_tgt_bhw3)[1]
        w = tf.shape(img_tgt_bhw3)[2]
        valid_mask_bhw = (pixel_src_bhw2[..., 0] >= 0.) & \
                         (pixel_src_bhw2[..., 0] <= tf.cast(w, tf.float32) - 0.) & \
                         (pixel_src_bhw2[..., 1] >= 0.) & \
                         (pixel_src_bhw2[..., 1] <= tf.cast(h, tf.float32) - 0.) & \
                         (depth_computed_bhw1[..., 0] >= 1e-3)
        valid_mask_bhw1 = tf.cast(valid_mask_bhw[..., tf.newaxis], tf.float32)

        # occlusion aware mask
        no_occlusion_mask_bhw1 = tf.cast(depth_proj_bhw1 >= depth_computed_bhw1, tf.float32)
        valid_mask_bhw1 *= no_occlusion_mask_bhw1

        # erode valid mask to be conservative
        valid_mask_bhw1 = tf.nn.erosion2d(
            value=valid_mask_bhw1,
            filters=tf.zeros((9, 9, 1)),
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
            dilations=[1, 1, 1, 1],
        )

        # photometric loss
        img_loss_bhw3 = self.photometric_loss(img_tgt_bhw3, img_proj_bhw3) * valid_mask_bhw1
        img_loss = tf.math.divide_no_nan(tf.reduce_sum(img_loss_bhw3),
                                         3 * tf.reduce_sum(valid_mask_bhw1))

        return img_loss

    def sym_warp_loss(self,
        img_tgt_bhw3: tf.Tensor,
        depth_tgt_bhw1: tf.Tensor,
        img_src_bhw3: tf.Tensor,
        depth_src_bhw1: tf.Tensor,
        src_T_tgt_b: Pose3D,
        tgt_T_src_b: Pose3D,
        cam_b: PinholeCam,
    ) -> tf.Tensor:
        warp_loss = self.warp_loss(
            img_tgt_bhw3,
            depth_tgt_bhw1,
            img_src_bhw3,
            depth_src_bhw1,
            src_T_tgt_b,
            cam_b,
        )
        warp_loss_inv = self.warp_loss(
            img_src_bhw3,
            depth_src_bhw1,
            img_tgt_bhw3,
            depth_tgt_bhw1,
            tgt_T_src_b,
            cam_b,
        )

        return (warp_loss + warp_loss_inv) / 2.

    def multi_sym_warp_loss(self,
        img_tgt_bhw3: tf.Tensor,
        depth_tgt_fullres_kbhw1: T.Sequence[tf.Tensor],
        img_src_bhw3: tf.Tensor,
        depth_src_fullres_kbhw1: T.Sequence[tf.Tensor],
        src_T_tgt_b: Pose3D,
        cam_b: PinholeCam,
    ) -> tf.Tensor:
        tgt_T_src_b = src_T_tgt_b.inv()

        warp_loss = 0.
        for depth_tgt_bhw1, depth_src_bhw1 in zip(depth_tgt_fullres_kbhw1, depth_src_fullres_kbhw1):
            warp_loss += self.sym_warp_loss(
                img_tgt_bhw3,
                depth_tgt_bhw1,
                img_src_bhw3,
                depth_src_bhw1,
                src_T_tgt_b,
                tgt_T_src_b,
                cam_b,
            )

        return warp_loss / float(len(depth_tgt_fullres_kbhw1))

    def multi_smooth_loss(self,
        depth_kbhw1: T.Sequence[tf.Tensor],
        img_bhw3: tf.Tensor,
    ) -> tf.Tensor:
        smooth_loss = 0.
        for scale, depth_bhw1 in enumerate(depth_kbhw1):
            h = tf.shape(depth_bhw1)[1]
            w = tf.shape(depth_bhw1)[2]
            img_resized_bhw3 = tf.image.resize(img_bhw3, (h, w), antialias=True)
            smooth_loss += self.smooth_loss(depth_bhw1, img_resized_bhw3) / (2 ** scale)

        return smooth_loss / float(len(depth_kbhw1))

    def smooth_loss(self, depth_bhw1: tf.Tensor, img_bhw3: tf.Tensor):
        # inverse depth
        disp_bhw1 = tf.math.divide_no_nan(1.0, depth_bhw1)

        # normalize
        mean_disp_b111 = tf.reduce_mean(disp_bhw1, axis=(1, 2), keepdims=True)
        norm_disp_bhw1 = tf.math.divide_no_nan(disp_bhw1, mean_disp_b111)

        # inverse depth gradient
        grad_disp_x_bhw1 = tf.abs(norm_disp_bhw1[:, :, :-1, :] - norm_disp_bhw1[:, :, 1:, :])
        grad_disp_y_bhw1 = tf.abs(norm_disp_bhw1[:, :-1, :, :] - norm_disp_bhw1[:, 1:, :, :])

        # image gradient (edge intensity)
        grad_img_x_bhw3 = tf.abs(img_bhw3[:, :, :-1, :] - img_bhw3[:, :, 1:, :])
        grad_img_y_bhw3 = tf.abs(img_bhw3[:, :-1, :, :] - img_bhw3[:, 1:, :, :])

        # edge-aware weighting
        grad_disp_x_bhw1 *= tf.exp(-tf.reduce_mean(grad_img_x_bhw3, axis=-1, keepdims=True))
        grad_disp_y_bhw1 *= tf.exp(-tf.reduce_mean(grad_img_y_bhw3, axis=-1, keepdims=True))

        # weighted smooth loss
        loss = tf.reduce_mean(grad_disp_x_bhw1) + tf.reduce_mean(grad_disp_y_bhw1)

        return loss
