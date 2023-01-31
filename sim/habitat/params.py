import math

from utils.params import ParamDict

PARAMS = ParamDict(
    img_size_hw = (144, 256),
    img_hfov = 90.,
    random_seed = 4242,
    obstacle_search_radius = 1.0,
    sensor_height = 1.5,
    perturb = ParamDict(
        R = math.radians(20.),
        t = 0.4,
    ),
)
