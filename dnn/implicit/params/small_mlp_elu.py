from dnn.implicit.params.default import PARAMS as DEFAULT_PARAMS

PARAMS = DEFAULT_PARAMS(
    data = DEFAULT_PARAMS.data(
        batch_size=2**18,
        epoch_size=1000,
    ),

    model = DEFAULT_PARAMS.model(
        mlp_layers=5,
        mlp_width=128,
        num_dir_freq=10,
        num_pos_freq=10,
        mlp_activation="elu",
    ),
)
