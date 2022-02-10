import argparse
import os
import time

import tensorflow as tf
import tensorflow.keras as tfk

from utils.tf_utils import set_tf_memory_growth
from dnn.data import VODataPipe
from dnn.model import DeepPose

def parse_args():
    parser = argparse.ArgumentParser()
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
    sess_dir = get_session_dir(args.output)

    data_pipe = VODataPipe()
    train_ds = data_pipe.build_train_ds()
    val_ds = data_pipe.build_val_ds()

    model = DeepPose()
    # load from checkpoint if provided
    if args.load is not None:
        model.load_weights(args.load)
    model.compile(tfk.optimizers.Adam(1e-5))

    model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=2000,
        callbacks=[
            tfk.callbacks.ModelCheckpoint(
                os.path.join(sess_dir, "ckpts", "epoch-{epoch:02d}")
            ),
            tfk.callbacks.TensorBoard(
                log_dir=os.path.join(sess_dir, "logs"),
                update_freq=50,
                profile_batch=0,
                histogram_freq=1,
            ),
        ],
    )
