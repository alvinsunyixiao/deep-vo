import argparse
import math

import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow_probability import distributions as tfd
from tqdm import trange

from utils.pose3d import Pose3D
from utils.params import ParamDict
from dnn.data import VODataPipe

class ScaleCalibrator:
    def __init__(self):
        self.args = self._parse_args()
        self.p = ParamDict.from_file(self.args.params)

        self.model = tfk.models.load_model(self.args.model)
        self.cal_ds = VODataPipe(self.p.data(batch_size=self.args.batch_size)).build_cal_ds()
        self.optimizer = tfk.optimizers.Adam(1e-3)

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--model", type=str, required=True,
                            help="path to the scod model")
        parser.add_argument("-p", "--params", type=str, default="params.py",
                            help="path to parameter file")
        parser.add_argument("--num-epochs", type=int, default=20,
                            help="number of epochs to run on calibration data")
        parser.add_argument("--batch-size", type=int, default=8,
                            help="batch size to use for the optimization")

        return parser.parse_args()

    @tf.function
    def _optimize_once(self, data):
        with tf.GradientTape() as tape:
            output, S = self.model({"image1": data["image1"], "image2": data["image2"]})

            cov = tf.linalg.diag(output["sigma"]**2 + 1e-5) + S
            w_T_c1 = Pose3D.from_storage(data["pose1"])
            w_T_c2 = Pose3D.from_storage(data["pose2"])
            c1_T_q = Pose3D.from_se3(output["mu"])
            w_T_q = w_T_c1 @ c1_T_q
            c2_T_q = w_T_c2.inv() @ w_T_q

            dist = tfd.MultivariateNormalTriL(scale_tril=tf.linalg.cholesky(cov))
            log_ll = dist.log_prob(c2_T_q.to_se3())
            loss = tf.reduce_mean(-log_ll)

        grads = tape.gradient(loss, self.model.log_prior)
        self.optimizer.apply_gradients([(grads, self.model.log_prior)])

        return loss

    def calibrate(self):
        pbar = trange(self.args.num_epochs)
        for epoch in pbar:
            for data in self.cal_ds:
                loss = self._optimize_once(data)
                pbar.set_postfix({"Loss": loss,
                                  "Log prior": self.model.log_prior.numpy()})
        self.model.save(self.args.model)

if __name__ == "__main__":
    ScaleCalibrator().calibrate()
