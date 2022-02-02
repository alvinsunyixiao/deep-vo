from __future__ import annotations

import numpy as np
import typing as T

from scipy.spatial.transform import Rotation

class Pose3D:
    def __init__(self, orientation: Rotation, position: np.ndarray):
        assert position.shape == (3,), "Position has to be a size-3 vector"
        self.R = orientation
        self.t = position

    def inv(self) -> Pose3D:
        R_inv = self.R.inv()
        return Pose3D(
            position=-R_inv.apply(self.t),
            orientation=R_inv,
        )

    def __matmul__(self, other: T.Union[Pose3D, np.ndarray]):
        if isinstance(other, Pose3D):
            return Pose3D(
                position=self.R.apply(other.t) + self.t,
                orientation=self.R * other.R
            )
        elif isinstance(other, np.ndarray):
            assert other.shape == (3,), f"Shape {other.shape} vector not supported"
            return self.R.apply(other) + self.t

    @classmethod
    def from_se3(cls, se3: np.ndarray) -> Pose3D:
        assert se3.shape == (6,), "se3 vectors must be size-6"
        t = se3[:3]
        w = se3[3:]
        theta = np.linalg.norm(w)
        V = np.eye(3)
        if abs(theta) >= 1e-3:
            wx = self._skew_sym(w)
            wx2 = wx @ wx
            V += (1 - np.cos(theta)) / theta**2 * wx + \
                 (theta - np.sin(theta)) / theta**3 * wx2

        return Pose3D(
            position=V @ t,
            orientation=Rotation.from_rotvec(w),
        )

    def as_se3(self) -> np.ndarray:
        w = self.R.as_rotvec()
        V_inv = np.eye(3)
        theta = np.linalg.norm(w)
        if abs(theta) >= 1e-3:
            wx = self._skew_sym(w)
            wx2 = wx @ wx
            V_inv += -.5 * wx + \
                (1 - (theta * np.cos(theta / 2)) / (2 * np.sin(theta / 2))) / theta**2 * wx2

        return np.concatenate([V_inv @ self.t, w])

    def _skew_sym(self, w: np.ndarray) -> np.ndarray:
        assert w.shape == (3,)
        return np.array([
            [0., -w[2], w[1]],
            [w[2], 0., -w[0]],
            [-w[1], w[0], 0.],
        ])

    @classmethod
    def random(cls, max_rotation: float, max_translation: float) -> Pose3D:
        w = np.random.normal(size=3)
        theta = np.random.uniform(high=max_rotation)
        w = w / np.linalg.norm(w) * theta
        R = Rotation.from_rotvec(w)
        t = np.random.uniform(low=-max_translation, high=-max_translation, size=3)

        return Pose3D(R, t)