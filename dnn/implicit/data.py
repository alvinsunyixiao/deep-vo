import numpy as np
import tensorflow as tf

from utils.camera import PinholeCam
from utils.params import ParamDict

class DataTransformer:

    DEFATUL_PARAMS = ParamDict(
    )

    def __init__(self, params=DEFATUL_PARAMS):
        self.p = params

    def filter_depth(self, depth_hw1):
        cam: PinholeCam = self.p.cam

