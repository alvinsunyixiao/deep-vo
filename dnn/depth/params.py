from dnn.depth.data import DepthDataPipe
from utils.params import ParamDict

PARAMS = ParamDict(
    data=DepthDataPipe.DEFAULT_PARAMS,
    trainer=ParamDict(
        num_epochs = 1000,
        save_freq = 5,
        img_log_freq = 500,
    ),
)
