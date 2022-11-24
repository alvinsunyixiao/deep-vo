from dnn.implicit.params.default import PARAMS as DEFAULT_PARAMS

PARAMS = DEFAULT_PARAMS(
    data = DEFAULT_PARAMS.data(
        batch_size=2**18,
        epoch_size=1000,
    ),

    model = DEFAULT_PARAMS.model(
        mlp_layers=6,
        mlp_width=256,
        num_dir_freq=10,
        max_dir_freq=18,
        num_pos_freq=10,
    ),

    trainer = DEFAULT_PARAMS.trainer(
        num_epochs=1000,
        lr=DEFAULT_PARAMS.trainer.lr(
            init=1e-3,
            end=1e-5,
            delay=50000,
        )
    ),
)
