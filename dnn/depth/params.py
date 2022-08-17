from dnn.depth.data import DepthDataPipe
from utils.params import ParamDict

def lr_schedule(epoch: int) -> float:
    if epoch < 90:
        return 1e-4
    elif epoch < 120:
        return 1e-5
    else:
        return 1e-6

PARAMS = ParamDict(
    data=DepthDataPipe.DEFAULT_PARAMS,
    trainer=ParamDict(
        num_epochs = 140,
        save_freq = 10,
        img_log_freq = 2000,
        weight_decay = 1e-4,
        lr_schedule = lr_schedule,
        clipnorm = 1.,
    ),
)
