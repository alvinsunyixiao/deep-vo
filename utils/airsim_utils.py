import airsim
import tensorflow as tf

from utils.pose3d import Pose3D, Rot3D

def rot3d_to_airsim(rot: Rot3D) -> airsim.Quaternionr:
    tf.assert_equal(rot.shape, tf.cast((), tf.int32),
                    "convertion to airsim only supports single pose")
    q = rot.quat.numpy().astype(float)
    return airsim.Quaternionr(q[0], q[1], q[2], q[3])

def pose3d_to_airsim(pose: Pose3D) -> airsim.Pose:
    tf.assert_equal(pose.shape, tf.cast((), tf.int32),
                    "convertion to airsim only supports single pose")
    t = pose.t.numpy().astype(float)

    return airsim.Pose(
        position_val=airsim.Vector3r(t[0], t[1], t[2]),
        orientation_val=rot3d_to_airsim(pose.R),
    )

def airsim_to_rot3d(quaternion: airsim.Quaternionr) -> Rot3D:
    return Rot3D(quaternion.to_numpy_array())

def airsim_to_pose3d(pose: airsim.Pose) -> Pose3D:
    return Pose3D(
        orientation=airsim_to_rot3d(pose.orientation),
        position=pose.position.to_numpy_array(),
    )
