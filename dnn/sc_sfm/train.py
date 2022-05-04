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
from utils.pose3d import Pose3D

class Trainer:

    DEFAULT_PARAMS = ParamDict(
        num_epochs = 300,
        save_freq = 2,
    )

    def __init__(self):
        self.args = self._parse_args()
        self.p = ParamDict.from_file(self.args.params)

        # model
        self.sc_sfm = SCSFM(self.p.model)
        self.model = self.sc_sfm.build_model()

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

        return parser.parse_args()

    @tf.function
    def _train_step(self, data: T.Dict[str, T.Any], optimizer: tfk.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            outputs = self.model(data)
            img_loss, geo_loss, smooth_loss = self.loss.all_loss(
                img1_bhw3=data["image1"],
                img2_bhw3=data["image2"],
                depth1_bhw1=outputs["depth1"],
                depth2_bhw1=outputs["depth2"],
                #c1_T_c2=Pose3D.from_se3(outputs["c1_T_c2"]),
                #c2_T_c1=Pose3D.from_se3(outputs["c2_T_c1"]),
                c1_T_c2=Pose3D.from_storage(data["pose1"]).inv() @ Pose3D.from_storage(data["pose2"]),
                c2_T_c1=Pose3D.from_storage(data["pose2"]).inv() @ Pose3D.from_storage(data["pose1"]),
            )
            tf.summary.scalar("img_loss", img_loss)
            tf.summary.scalar("geo_loss", geo_loss)
            tf.summary.scalar("smooth_loss", smooth_loss)

            w = self.p.loss.weights
            loss = w.img * img_loss + w.geo * geo_loss + w.smooth * smooth_loss
            tf.summary.scalar("loss", loss)
            tf.print("Loss:", loss, "img_loss:", img_loss, "geo_loss:", geo_loss, "smooth_loss:", smooth_loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train(self):
        optimizer = tfk.optimizers.Adam(1e-4)
        for i in range(self.p.trainer.num_epochs):
            print(f"------ Starting Epoch {i} ------")
            with self.train_writer.as_default():
                for data in self.train_ds:
                    self.global_step.assign_add(1)
                    self._train_step(data, optimizer)

if __name__ == "__main__":
    Trainer().train()
