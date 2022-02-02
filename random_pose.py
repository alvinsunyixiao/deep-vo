import functools
import numpy as np

import typing as T

from scipy.spatial.transform import Rotation
from pose3d import Pose3D

class RandomPoseGen:
    def __init__(self,
        min_pos: T.Tuple[float, float, float],
        max_pos: T.Tuple[float, float, float],
        min_rp: T.Tuple[float, float],
        max_rp: T.Tuple[float, float],
    ):
        self.pos_gen = functools.partial(np.random.uniform, low=min_pos, high=max_pos)

        min_rpy = min_rp + (-np.pi,)
        max_rpy = max_rp + (np.pi,)
        self.rpy_gen = functools.partial(np.random.uniform, low=min_rpy, high=max_rpy)

    def gen(self) -> Pose3D:
        pos = self.pos_gen()
        rpy = self.rpy_gen()

        return Pose3D(
            position=pos,
            orientation=Rotation.from_euler("xyz", rpy),
        )
