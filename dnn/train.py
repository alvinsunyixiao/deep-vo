import argparse
import os
import time

import tensorflow as tf
import tensorflow.keras as tfk

from dnn.data import VODataPipe
from dnn.model import DeepPose
from utils.params import ParamDict
from utils.tf_utils import set_tf_memory_growth

def default_lr_schedule(epoch: int) -> float:
    # warm start
    if epoch < 200:
        return epoch / 200 * (1e-3 - 1e-6) + 1e-6
    elif epoch < 1000:
        return 1e-3
    elif epoch < 1600:
        return 1e-4
    elif epoch < 2000:
        return 1e-5

TRAINER_DEFAULT_PARAMS = ParamDict(
    num_epochs=2000,
    log_freq=50,
    lr_schedule=default_lr_schedule,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", type=str, default="params.py",
                        help="path to load the parameter file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output directory to store checkpoints and logs")
    parser.add_argument("-l", "--load", type=str, default=None,
                        help="load weight from path")

    return parser.parse_args()

def get_session_dir(root: str) -> str:
    return os.path.join(root, time.strftime("sess_%m-%d-%y_%H-%M-%S"))

if __name__ == "__main__":
    set_tf_memory_growth(True)

    args = parse_args()
    p = ParamDict.from_file(args.params)
    sess_dir = get_session_dir(args.output)

    data_pipe = VODataPipe(p.data)
    train_ds = data_pipe.build_train_ds()
    val_ds = data_pipe.build_val_ds()

    model = DeepPose(p.model)
    # load from checkpoint if provided
    if args.load is not None:
        model.load_weights(args.load)
    model.compile(tfk.optimizers.Adam(1e-5))

    model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=p.trainer.num_epochs,
        callbacks=[
            tfk.callbacks.ModelCheckpoint(
                os.path.join(sess_dir, "ckpts", "epoch-{epoch:02d}")
            ),
            tfk.callbacks.TensorBoard(
                log_dir=os.path.join(sess_dir, "logs"),
                update_freq=p.trainer.log_freq,
                profile_batch=0,
                histogram_freq=1,
            ),
            tfk.callbacks.LearningRateScheduler(p.trainer.lr_schedule, verbose=1),
        ],
    )
