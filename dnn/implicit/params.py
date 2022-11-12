from dnn.implicit.data import PointLoader
from dnn.implicit.model import NeRD
from dnn.implicit.train import Trainer
from utils.params import ParamDict

PARAMS = ParamDict(
    data = PointLoader.DEFAULT_PARAMS,
    model = NeRD.DEFAULT_PARAMS,
    trainer = Trainer.DEFAULT_PARAMS,
)
