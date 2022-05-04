from dnn.sc_sfm.data import VODataPipe
from dnn.sc_sfm.model import SCSFM
from dnn.sc_sfm.train import Trainer
from dnn.sc_sfm.loss import LossManager
from utils.params import ParamDict

PARAMS = ParamDict(
    data=VODataPipe.DEFAULT_PARAMS,
    model=SCSFM.DEFAULT_PARAMS,
    trainer=Trainer.DEFAULT_PARAMS,
    loss=LossManager.DEFAULT_PARAMS,
)
