import argparse
import os
import time
import tensorflow as tf
import tensorflow.keras as tfk
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

        # output dir
        self.sess_dir = os.path.join(self.args.output, time.strftime("sess_%y-%m-%d_%H-%M-%S"))
        self.ckpt_dir = os.path.join(self.sess_dir, "ckpts")
        self.log_dir = os.path.join(self.sess_dir, "logs")
        self.train_writer = tf.summary.create_file_writer(self.log_dir, name="train")
        self.val_writer = tf.summary.create_file_writer(self.log_dir, name="validation")
        self.global_step = tf.Variable(0, dtype=tf.int64)
        tf.summary.experimental.set_step(self.global_step)

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--params", type=str, required=True,
                            help="path to the parameter file")
        parser.add_argument("-o", "--output", type=str, required=True,
                            help="path to store weights")
        parser.add_argument("-l", "--load", type=str, default=None,
                            help="path to load weights from")

        return parser.parse_args()

    @tf.function
    def _train_step(self, data: T.Dict[str, T.Any], optimizer: tfk.optimizers.Optimizer):
        # some constants
        w = self.p.loss.weights
        src_T_tgt = Pose3D.from_storage(data["src_T_tgt"])
        camera = PinholeCam.from_storage(data["cam_param"])

        with tf.GradientTape() as tape:
            depth_tgt = self.sc_sfm.depth_net(data["img_tgt"])

            # disparity smoothness loss
            smooth_loss = self.loss.smooth_loss(depth_tgt["disparity"], data["img_tgt"])
            tf.summary.scalar("smooth_loss", smooth_loss)

            # stereo pass
            stereo_warp_loss = self.loss.warp_loss(
                img_tgt_bhw3=data["img_tgt"],
                depth_tgt_bhw1=depth_tgt["depth"],
                img_src_bhw3=data["img_src"],
                src_T_tgt_b=src_T_tgt,
                cam_b=camera,
            )

            # temporal pass prev
            tgt_prev_T_tgt = Pose3D.from_se3(self.sc_sfm.pose_net({
                "image1": data["img_tgt_prev"],
                "image2": data["img_tgt"],
            }))
            prev_warp_loss = self.loss.warp_loss(
                img_tgt_bhw3=data["img_tgt"],
                depth_tgt_bhw1=depth_tgt["depth"],
                img_src_bhw3=data["img_tgt_prev"],
                src_T_tgt_b=tgt_prev_T_tgt,
                cam_b=camera,
            )

            # temporal pass next
            tgt_next_T_tgt = Pose3D.from_se3(self.sc_sfm.pose_net({
                "image1": data["img_tgt_next"],
                "image2": data["img_tgt"],
            }))
            next_warp_loss = self.loss.warp_loss(
                img_tgt_bhw3=data["img_tgt"],
                depth_tgt_bhw1=depth_tgt["depth"],
                img_src_bhw3=data["img_tgt_next"],
                src_T_tgt_b=tgt_next_T_tgt,
                cam_b=camera,
            )

            # per-pixel minimum
            warp_diff = tf.reduce_min([stereo_warp_loss, prev_warp_loss, next_warp_loss], axis=0)
            raw_diff = tf.reduce_min([
                self.loss.photometric_loss(data["img_tgt"], data["img_src"]),
                self.loss.photometric_loss(data["img_tgt"], data["img_tgt_prev"]),
                self.loss.photometric_loss(data["img_tgt"], data["img_tgt_next"]),
            ], axis=0)
            auto_mask = tf.cast(warp_diff < raw_diff, tf.float32)
            warp_loss = tf.reduce_sum(warp_diff * auto_mask) / tf.reduce_sum(auto_mask)
            tf.summary.scalar("warp_loss", warp_loss)

            loss = smooth_loss + warp_loss
            tf.summary.scalar("loss", loss)

            tf.print("Loss:", loss)
            #tf.print(tf.histogram_fixed_width(depth_tgt["depth"], (0, 100), 5))

        grads = tape.gradient(loss, self.sc_sfm.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.sc_sfm.trainable_variables))

        # plot image and disparity
        if self.global_step % self.p.trainer.img_log_freq == 0:
            tf.summary.image("img_tgt_prev", data["img_tgt_prev"])
            tf.summary.image("img_tgt", data["img_tgt"])
            disp_max = tf.reduce_max(depth_tgt["disparity"], axis=(1, 2, 3), keepdims=True)
            disp_min = tf.reduce_min(depth_tgt["disparity"], axis=(1, 2, 3), keepdims=True)
            tf.summary.image("disparity",
                (depth_tgt["disparity"] - disp_min) / (disp_max - disp_min))
            tf.summary.image("warp_diff", warp_diff)
            tf.summary.image("auto_mask", auto_mask)

    def train(self):
        optimizer = tfk.optimizers.Adam(1e-4)
        for i in range(self.p.trainer.num_epochs):
            print(f"------ Saving Checkpoint ------")
            self.sc_sfm.save(os.path.join(self.ckpt_dir, f"epoch-{i}"))
            print(f"------ Starting Epoch {i} ------")
            with self.train_writer.as_default():
                for data in self.train_ds:
                    self.global_step.assign_add(1)
                    self._train_step(data, optimizer)

if __name__ == "__main__":
    Trainer().train()
