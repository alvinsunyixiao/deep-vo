import argparse
import os
import time
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons as tfa
import typing as T

from dnn.sc_sfm.data import DataAirsimSeq
from dnn.sc_sfm.loss import LossManager
from dnn.sc_sfm.model import SCSFM
from utils.params import ParamDict
from utils.pose3d import Pose3D, Rot3D
from utils.camera import PinholeCam

class Trainer:
    def __init__(self):
        self.args = self._parse_args()
        self.p = ParamDict.from_file(self.args.params)

        # output dir
        self.sess_dir = os.path.join(self.args.output, time.strftime("sess_%y-%m-%d_%H-%M-%S"))
        self.ckpt_dir = os.path.join(self.sess_dir, "ckpts")
        self.log_dir = os.path.join(self.sess_dir, "logs")
        self.train_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "train"))
        self.val_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "validation"))
        self.global_step = tf.Variable(0, dtype=tf.int64)

        # model
        self.sc_sfm = SCSFM(self.p.model)
        if self.args.load is not None:
            self.sc_sfm.load_weights(self.args.load)

        # loss
        self.loss = LossManager(self.p.loss)

        # data
        self.data_pipe = DataAirsimSeq(self.p.data)
        self.train_ds = self.data_pipe.build_train_ds()
        self.val_ds = self.data_pipe.build_val_ds()

        tf.summary.experimental.set_step(self.global_step)

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--params", type=str, required=True,
                            help="path to the parameter file")
        parser.add_argument("-o", "--output", type=str, required=True,
                            help="path to store weights")
        parser.add_argument("-l", "--load", type=str, default=None,
                            help="path to load weights from")
        parser.add_argument("--debug", action="store_true",
                            help="dump TF2 debug info")

        return parser.parse_args()

    def _multiscale_upsample(self, depth_kbhw1: T.Sequence[tf.Tensor]) -> T.List[tf.Tensor]:
        num_scales = len(depth_kbhw1)
        img_size = tf.shape(depth_kbhw1[0])[1:3]

        depth_fullres_kbhw1 = [depth_kbhw1[0]]
        for i in range(1, num_scales):
            depth_fullres_kbhw1.append(tf.image.resize(depth_kbhw1[i], img_size))

        return depth_fullres_kbhw1

    def _generate_output(self, data: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        outputs = {
            "depth_tgt": self.sc_sfm.depth_net(data["img_tgt"]),
            "depth_tgt_prev": self.sc_sfm.depth_net(data["img_tgt_prev"]),
            "depth_tgt_next": self.sc_sfm.depth_net(data["img_tgt_next"]),
            "depth_src": self.sc_sfm.depth_net(data["img_src"]),
            # temporal pass prev
            "tgt_prev_T_tgt": Pose3D.from_se3(self.sc_sfm.pose_net({
                "image1": data["img_tgt_prev"],
                "image2": data["img_tgt"],
            })),
            # temporal pass next
            "tgt_next_T_tgt": Pose3D.from_se3(self.sc_sfm.pose_net({
                "image1": data["img_tgt_next"],
                "image2": data["img_tgt"],
            })),
        }
        outputs["depth_tgt_fullres"] = self._multiscale_upsample(outputs["depth_tgt"])
        outputs["depth_tgt_prev_fullres"] = self._multiscale_upsample(outputs["depth_tgt_prev"])
        outputs["depth_tgt_next_fullres"] = self._multiscale_upsample(outputs["depth_tgt_next"])
        outputs["depth_src_fullres"] = self._multiscale_upsample(outputs["depth_src"])

        return outputs

    def _generate_losses(self, data: T.Dict[str, T.Any], outputs: T.Dict[str, T.Any]) -> tf.Tensor:
        src_T_tgt = Pose3D.from_storage(data["src_T_tgt"])
        camera = PinholeCam.from_storage(data["cam_param"])

        smooth_loss = 0.
        smooth_loss += self.loss.multi_smooth_loss(outputs["depth_tgt"], data["img_tgt"])
        smooth_loss += self.loss.multi_smooth_loss(outputs["depth_tgt_prev"], data["img_tgt_prev"])
        smooth_loss += self.loss.multi_smooth_loss(outputs["depth_tgt_next"], data["img_tgt_next"])
        smooth_loss += self.loss.multi_smooth_loss(outputs["depth_src"], data["img_tgt"])
        smooth_loss /= 4.

        stereo_warp_loss = self.loss.multi_sym_warp_loss(
            img_tgt_bhw3=data["img_tgt"],
            depth_tgt_fullres_kbhw1=outputs["depth_tgt_fullres"],
            img_src_bhw3=data["img_src"],
            depth_src_fullres_kbhw1=outputs["depth_src_fullres"],
            src_T_tgt_b=src_T_tgt,
            cam_b=camera,
        )

        prev_warp_loss = self.loss.multi_sym_warp_loss(
            img_tgt_bhw3=data["img_tgt"],
            depth_tgt_fullres_kbhw1=outputs["depth_tgt_fullres"],
            img_src_bhw3=data["img_tgt_prev"],
            depth_src_fullres_kbhw1=outputs["depth_tgt_prev_fullres"],
            src_T_tgt_b=outputs["tgt_prev_T_tgt"],
            cam_b=camera,
        )

        next_warp_loss = self.loss.multi_sym_warp_loss(
            img_tgt_bhw3=data["img_tgt"],
            depth_tgt_fullres_kbhw1=outputs["depth_tgt_fullres"],
            img_src_bhw3=data["img_tgt_next"],
            depth_src_fullres_kbhw1=outputs["depth_tgt_next_fullres"],
            src_T_tgt_b=outputs["tgt_next_T_tgt"],
            cam_b=camera,
        )

        loss = self.p.loss.weights.smooth * smooth_loss + \
               self.p.loss.weights.img * (stereo_warp_loss + prev_warp_loss + next_warp_loss) / 3.

        tf.summary.scalar("smooth_loss", smooth_loss)
        tf.summary.scalar("stereo_warp_loss", stereo_warp_loss)
        tf.summary.scalar("prev_warp_loss", prev_warp_loss)
        tf.summary.scalar("next_warp_loss", next_warp_loss)
        tf.summary.scalar("loss", loss)

        return loss

    @tf.function
    def _val_step(self, data: T.Dict[str, T.Any]) -> None:
        outputs = self._generate_output(data)
        self._generate_losses(data, outputs)

    @tf.function
    def _train_step(self, data: T.Dict[str, T.Any], optimizer: tfk.optimizers.Optimizer) -> None:
        with tf.GradientTape() as tape:
            outputs = self._generate_output(data)
            loss = self._generate_losses(data, outputs)

            tf.print("Loss:", loss)
            #tf.print(tf.histogram_fixed_width(depth_tgt, (0, 100), 5))

        grads = tape.gradient(loss, self.sc_sfm.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.sc_sfm.trainable_variables))

        # plot image and disparity
        if self.global_step % self.p.trainer.img_log_freq == 0:
            tf.summary.image("img_tgt", data["img_tgt"])
            disp_tgt = 1.0 / outputs["depth_tgt"][0]
            disp_max = tf.reduce_max(disp_tgt, axis=(1, 2, 3), keepdims=True)
            disp_min = tf.reduce_min(disp_tgt, axis=(1, 2, 3), keepdims=True)
            tf.summary.image("inverse depth",
                (disp_tgt - disp_min) / (disp_max - disp_min))

            # warp tgt_prev viz
            tgt_prev_T_tgt_b11 = outputs["tgt_prev_T_tgt"][:, tf.newaxis, tf.newaxis]
            cam_b11 = PinholeCam.from_storage(data["cam_param"])[:, tf.newaxis, tf.newaxis]
            pixel_src_bhw2, _ = cam_b11.reproject(outputs["depth_tgt"][0], tgt_prev_T_tgt_b11)
            img_proj_bhw3 = tfa.image.resampler(data["img_tgt_prev"], pixel_src_bhw2)
            tf.summary.image("img_tgt_prev", data["img_tgt_prev"])
            tf.summary.image("img_tgt_prev_warp", img_proj_bhw3)

            # warp src
            src_T_tgt_b11 = Pose3D.from_storage(data["src_T_tgt"])[:, tf.newaxis, tf.newaxis]
            pixel_src_bhw2, _ = cam_b11.reproject(outputs["depth_tgt"][0], src_T_tgt_b11)
            img_proj_bhw3 = tfa.image.resampler(data["img_src"], pixel_src_bhw2)
            tf.summary.image("img_src", data["img_src"])
            tf.summary.image("img_src_warp", img_proj_bhw3)

    def train(self):
        optimizer = tfk.optimizers.Adam(1e-4)
        for i in range(self.p.trainer.num_epochs):
            print(f"------ Saving Checkpoint ------")
            if self.args.debug:
                tf.debugging.disable_check_numerics()
            self.sc_sfm.save(os.path.join(self.ckpt_dir, f"epoch-{i}"))
            if self.args.debug:
                tf.debugging.enable_check_numerics()
            print(f"------ Starting Epoch {i} ------")

            with self.train_writer.as_default():
                for data in self.train_ds:
                    self.global_step.assign_add(1)
                    self._train_step(data, optimizer)

            print(f"------ Validating ------")
            with self.val_writer.as_default():
                for data in self.val_ds:
                    self.global_step.assign_add(1)
                    self._val_step(data)

if __name__ == "__main__":
    Trainer().train()
