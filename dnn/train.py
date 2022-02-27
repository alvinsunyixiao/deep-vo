import argparse
import math
import os
import time

import tensorflow as tf
import tensorflow.keras as tfk

from dnn.data import VODataPipe
from dnn.model import DeepPoseTrain
from utils.params import ParamDict
from utils.tf_utils import set_tf_memory_growth

def default_lr_schedule(epoch: int) -> float:
    if epoch < 1000:
        return 3e-5
    elif epoch < 1600:
        return 1e-5
    else:
        return 1e-6

TRAINER_DEFAULT_PARAMS = ParamDict(
    num_epochs=2000,
    log_freq=50,
    save_freq=10,
    lr_schedule=default_lr_schedule,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", type=str, default="params.py",
                        help="path to load the parameter file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output directory to store checkpoints and logs")
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help="resume from training session directory")

    return parser.parse_args()

def get_session_dir(root: str) -> str:
    return os.path.join(root, time.strftime("sess_%m-%d-%y_%H-%M-%S"))

if __name__ == "__main__":
    set_tf_memory_growth(True)

    args = parse_args()
    p = ParamDict.from_file(args.params)

    data_pipe = VODataPipe(p.data)
    train_ds = data_pipe.build_train_ds()
    val_ds = data_pipe.build_val_ds()

    model = DeepPoseTrain(p.model).build_model()
    model.compile(tfk.optimizers.Adam(epsilon=1e-4, clipnorm=2.))

    initial_epoch = 0
    if args.resume is not None:
        sess_dir = args.resume
        ckpts_dir = os.path.join(sess_dir, "ckpts")
        ckpts_files = os.listdir(ckpts_dir)
        last_ckpt_file = max(ckpts_files, key=lambda f: int(f.split("-")[1]))
        last_ckpt_file_path = os.path.join(ckpts_dir, last_ckpt_file)
        initial_epoch = int(last_ckpt_file.split("-")[1])
        print(f"Restored weights from {last_ckpt_file_path}")
        model.load_weights(last_ckpt_file_path)
    else:
        sess_dir = get_session_dir(args.output)

    model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=p.trainer.num_epochs,
        initial_epoch=initial_epoch,
        callbacks=[
            tfk.callbacks.ModelCheckpoint(
                os.path.join(sess_dir, "ckpts", "epoch-{epoch:03d}"),
                period=p.trainer.save_freq,
            ),
            tfk.callbacks.TensorBoard(
                log_dir=os.path.join(sess_dir, "logs"),
                update_freq=p.trainer.log_freq,
                profile_batch=(100, 500),
                histogram_freq=1,
            ),
            tfk.callbacks.LearningRateScheduler(p.trainer.lr_schedule),
        ],
    )
