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
        camera = ParamDict(
            fov_xy = (89.903625, 58.633181),
            size_xy = (256, 160),
        ),
        weights = ParamDict(
            img=1.,
            smooth=0.1,
            geo=0.5,
        ),
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params
        self.cam = PinholeCam.from_size_and_fov(
            size_xy=np.array(self.p.camera.size_xy),
            fov_xy=np.array(self.p.camera.fov_xy),
        )

    def ssim(self,
        img1_bhw3: tf.Tensor,
        img2_bhw3: tf.Tensor,
        max_val: float = 1.,
        filter_size: int = 11,
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

    def warp_loss(self,
        img_tgt_bhw3: tf.Tensor,
        depth_tgt_bhw1: tf.Tensor,
        img_src_bhw3: tf.Tensor,
        depth_src_bhw1: tf.Tensor,
        src_T_tgt_b: Pose3D,
    ) -> T.Tuple[tf.Tensor, tf.Tensor]:
        h = tf.shape(img_tgt_bhw3)[1]
        w = tf.shape(img_tgt_bhw3)[2]
        pixel_src_bhw2, depth_computed_bhw1 = self.cam.reproject(depth_tgt_bhw1, src_T_tgt_b)

        valid_mask_bhw = (pixel_src_bhw2[..., 0] >= 5.) & \
                         (pixel_src_bhw2[..., 0] <= tf.cast(w, tf.float32) - 6.) & \
                         (pixel_src_bhw2[..., 1] >= 5.) & \
                         (pixel_src_bhw2[..., 1] <= tf.cast(h, tf.float32) - 6.)
        valid_mask_bhw1 = tf.cast(valid_mask_bhw[..., None], tf.float32)

        img_proj_bhw3 = tfa.image.resampler(img_src_bhw3, pixel_src_bhw2)
        depth_proj_bhw1 = tfa.image.resampler(depth_src_bhw1, pixel_src_bhw2)

        # geometric loss
        depth_l1_bhw1 = tf.abs(depth_computed_bhw1 - depth_proj_bhw1)
        depth_diff_bhw1 = depth_l1_bhw1 / (depth_computed_bhw1 + depth_proj_bhw1)
        depth_diff_bhw1 = tf.clip_by_value(depth_diff_bhw1, 0., 1.)
        geo_loss = tf.reduce_sum(depth_diff_bhw1 * valid_mask_bhw1)
        geo_loss = tf.math.divide_no_nan(geo_loss, tf.reduce_sum(valid_mask_bhw1))

        # photometric loss
        img_l1_bhw3 = tf.abs(img_proj_bhw3 - img_tgt_bhw3)
        img_l1_bhw3 = tf.clip_by_value(img_l1_bhw3, 0., 1.)
        img_ssim_bhw3 = self.ssim(img_proj_bhw3, img_tgt_bhw3)
        img_diff_bhw3 = .15 * img_l1_bhw3 + .85 * (1. - img_ssim_bhw3) / 2.
        img_diff_bhw3 *= (1. - depth_diff_bhw1)
        img_loss = tf.reduce_sum(img_diff_bhw3 * valid_mask_bhw1)
        img_loss = tf.math.divide_no_nan(img_loss, tf.reduce_sum(valid_mask_bhw1) * 3)

        return img_loss, geo_loss

    def smooth_loss(self, depth_bhw1: tf.Tensor, img_bhw3: tf.Tensor):
        # normalize
        mean_depth_b111 = tf.reduce_mean(depth_bhw1, axis=(1, 2), keepdims=True)
        norm_depth_bhw1 = tf.math.divide_no_nan(depth_bhw1, mean_depth_b111)

        grad_depth_x_bhw1 = tf.abs(norm_depth_bhw1[:, :, :-1, :] - norm_depth_bhw1[:, :, 1:, :])
        grad_depth_y_bhw1 = tf.abs(norm_depth_bhw1[:, :-1, :, :] - norm_depth_bhw1[:, 1:, :, :])

        grad_img_x_bhw3 = tf.abs(img_bhw3[:, :, :-1, :] - img_bhw3[:, :, 1:, :])
        grad_img_y_bhw3 = tf.abs(img_bhw3[:, :-1, :, :] - img_bhw3[:, 1:, :, :])

        grad_depth_x_bhw1 *= tf.exp(-tf.reduce_mean(grad_img_x_bhw3, axis=-1, keepdims=True))
        grad_depth_y_bhw1 *= tf.exp(-tf.reduce_mean(grad_img_y_bhw3, axis=-1, keepdims=True))

        return tf.reduce_mean(grad_depth_x_bhw1) + tf.reduce_mean(grad_depth_y_bhw1)

    def all_loss(self,
        img1_bhw3: tf.Tensor,
        img2_bhw3: tf.Tensor,
        depth1_bhw1: tf.Tensor,
        depth2_bhw1: tf.Tensor,
        disp1_bhw1: tf.Tensor,
        disp2_bhw1: tf.Tensor,
        c1_T_c2: Pose3D,
        c2_T_c1: Pose3D,
    ):
        smooth_loss1 = self.smooth_loss(depth1_bhw1, img1_bhw3)
        smooth_loss2 = self.smooth_loss(depth2_bhw1, img2_bhw3)

        img_loss1, geo_loss1 = self.warp_loss(img2_bhw3, depth2_bhw1, img1_bhw3, depth1_bhw1, c1_T_c2)
        img_loss2, geo_loss2 = self.warp_loss(img1_bhw3, depth1_bhw1, img2_bhw3, depth2_bhw1, c2_T_c1)

        return img_loss1 + img_loss2, geo_loss1 + geo_loss2, smooth_loss1 + smooth_loss2
