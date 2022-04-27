from dnn.baseline.data import VODataPipe
from dnn.baseline.model import DeepPose
from dnn.baseline.train import TRAINER_DEFAULT_PARAMS
from utils.params import ParamDict

PARAMS = ParamDict(
    data=VODataPipe.DEFAULT_PARAMS,
    model=DeepPose.DEFAULT_PARAMS,
    trainer=TRAINER_DEFAULT_PARAMS,
)
