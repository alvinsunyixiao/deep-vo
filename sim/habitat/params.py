import math

from utils.params import ParamDict

PARAMS = ParamDict(
    img_size_hw = (144, 256),
    img_hfov = 120.,
    random_seed = 4242,
    num_samples = 200,
    obstacle_search_radius = 1.0,
    sensor_height = 1.5,
    perturb = ParamDict(
        roll = math.radians(30.),
        pitch = math.radians(30.),
    ),
)
