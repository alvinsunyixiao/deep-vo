import argparse
import os
import time
import tensorflow as tf
import tensorflow.keras as tfk
import typing as T

from dnn.sc_sfm.data import VODataPipe
from dnn.sc_sfm.loss import LossManager
from dnn.sc_sfm.model import SCSFM
from utils.params import ParamDict
from utils.pose3d import Pose3D, Rot3D

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
        self.data_pipe = VODataPipe(self.p.data)
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
        ned_T_edn = Pose3D.identity()
        ned_T_edn.R = Rot3D.from_matrix([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ])
        pose1 = Pose3D.from_storage(data["pose1"]) @ ned_T_edn
        pose2 = Pose3D.from_storage(data["pose2"]) @ ned_T_edn
        with tf.GradientTape() as tape:
            depth1 = self.sc_sfm.depth_net(data["image1"])
            depth2 = self.sc_sfm.depth_net(data["image2"])
            c1_T_c2 = Pose3D.from_se3(self.sc_sfm.pose_net({
                "image1": data["image1"],
                "image2": data["image2"],
            }))
            c2_T_c1 = Pose3D.from_se3(self.sc_sfm.pose_net({
                "image1": data["image2"],
                "image2": data["image1"],
            }))
            w = self.p.loss.weights

            # full unsupervised pass
            img_loss, geo_loss, smooth_loss = self.loss.all_loss(
                img1_bhw3=data["image1"],
                img2_bhw3=data["image2"],
                depth1_bhw1=depth1["depth"],
                depth2_bhw1=depth2["depth"],
                disp1_bhw1=depth1["disparity"],
                disp2_bhw1=depth2["disparity"],
                c1_T_c2=c1_T_c2,
                c2_T_c1=c2_T_c1,
            )
            unsup_loss = w.img * img_loss + w.geo * geo_loss + w.smooth * smooth_loss
            tf.summary.scalar("unsup_img_loss", img_loss)
            tf.summary.scalar("unsup_geo_loss", geo_loss)
            tf.summary.scalar("unsup_smooth_loss", smooth_loss)
            tf.summary.scalar("unsup_loss", unsup_loss)

            # semi-supervised pass with gt poses
            img_loss, geo_loss, smooth_loss = self.loss.all_loss(
                img1_bhw3=data["image1"],
                img2_bhw3=data["image2"],
                depth1_bhw1=depth1["depth"],
                depth2_bhw1=depth2["depth"],
                disp1_bhw1=depth1["disparity"],
                disp2_bhw1=depth2["disparity"],
                c1_T_c2=pose1.inv() @ pose2,
                c2_T_c1=pose2.inv() @ pose1,
            )
            sup_loss = w.img * img_loss + w.geo * geo_loss + w.smooth * smooth_loss
            tf.summary.scalar("sup_img_loss", img_loss)
            tf.summary.scalar("sup_geo_loss", geo_loss)
            tf.summary.scalar("sup_smooth_loss", smooth_loss)
            tf.summary.scalar("sup_loss", sup_loss)

            loss = unsup_loss + sup_loss
            tf.summary.scalar("loss", loss)

            tf.print("Loss:", loss)
            #tf.print(tf.histogram_fixed_width(depth1["depth"], (0, 100), 5))
            #tf.print(c1_T_c2.to_se3())

        grads = tape.gradient(loss, self.sc_sfm.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.sc_sfm.trainable_variables))

        if self.global_step % self.p.trainer.img_log_freq == 0:
            tf.summary.image("image", data["image1"])
            disp_max = tf.reduce_max(depth1["disparity"], axis=(1, 2, 3), keepdims=True)
            disp_min = tf.reduce_min(depth1["disparity"], axis=(1, 2, 3), keepdims=True)
            tf.summary.image("disparity", (depth1["disparity"] - disp_min) / \
                                          (disp_max - disp_min))

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
