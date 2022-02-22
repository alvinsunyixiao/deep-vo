import gtsam

from utils.pose3d import Pose3D, Rot3D

def rot3d_to_gtasm(rot: Rot3D) -> gtsam.Rot3:
    q = rot.quat.numpy()
    return gtsam.Rot3(x=q[0], y=q[1], z=q[2], w=q[3])

def pose3d_to_gtsam(pose: Pose3D) -> gtsam.Pose3:
    return gtsam.Pose3(r=rot3d_to_gtasm(pose.R), t=pose.t.numpy())
