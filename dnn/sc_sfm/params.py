from dnn.sc_sfm.data import DataAirsimSeq
from dnn.sc_sfm.model import SCSFM
from dnn.sc_sfm.loss import LossManager
from utils.params import ParamDict

PARAMS = ParamDict(
    data=DataAirsimSeq.DEFAULT_PARAMS,
    model=SCSFM.DEFAULT_PARAMS,
    loss=LossManager.DEFAULT_PARAMS,
    trainer=ParamDict(
        num_epochs = 1000,
        save_freq = 5,
        img_log_freq = 500,
    ),
)
