from dnn.implicit.data import PointLoader
from dnn.implicit.model import NeRD
from dnn.implicit.train import Trainer
from utils.params import ParamDict

PARAMS = ParamDict(
    data = PointLoader.DEFAULT_PARAMS(
        batch_size=2**16,
    ),

    model = NeRD.DEFAULT_PARAMS(
        mlp_layers=5,
        mlp_width=128,
        num_dir_freq=12,
        num_pos_freq=10,
    ),

    trainer = Trainer.DEFAULT_PARAMS(
        num_epochs=200,
        lr=Trainer.DEFAULT_PARAMS.lr(
            init=1e-3,
            end=5e-6,
            delay=20000,
        )
    ),
)
