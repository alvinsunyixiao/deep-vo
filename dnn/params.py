from dnn.data import VODataPipe
from dnn.model import DeepPose
from dnn.train import TRAINER_DEFAULT_PARAMS
from utils.params import ParamDict

PARAMS = ParamDict(
    data=VODataPipe.DEFAULT_PARAMS,
    model=DeepPose.DEFAULT_PARAMS,
    trainer=TRAINER_DEFAULT_PARAMS,
)
