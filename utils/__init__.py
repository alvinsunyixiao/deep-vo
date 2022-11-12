import tensorflow as tf
import typing as T

from .pose3d import Rot3D, Pose3D

################### API Dispatch for Rot3D ###################

@tf.experimental.dispatch_for_api(tf.broadcast_to)
def Rot3D_broadcast_to(input: Rot3D, shape) -> Rot3D:
    return input.broadcast_to(shape)

@tf.experimental.dispatch_for_api(tf.cast)
def Rot3D_cast(x: Rot3D, dtype) -> Rot3D:
    return Rot3D(x.quat, dtype)

@tf.experimental.dispatch_for_api(tf.shape)
def Rot3D_shape(input: Rot3D, out_type=tf.int32) -> tf.Tensor:
    return tf.shape(input.quat, out_type=out_type)[:-1]

@tf.experimental.dispatch_for_api(tf.stack)
def Rot3D_stack(values: T.List[Rot3D], axis=0) -> Rot3D:
    assert len(values) > 0, "Attempting to stack empty array"
    axis = axis if axis >= 0 else axis - 1
    quats = tf.stack([R.quat for R in values], axis=axis)
    return Rot3D(quats, renormalize=False)

@tf.experimental.dispatch_for_api(tf.concat)
def Rot3D_concat(values: T.List[Rot3D], axis) -> Rot3D:
    assert len(values) > 0, "Attempting to concat empty array"
    axis = axis if axis >= 0 else axis - 1
    quats = tf.concat([R.quat for R in values], axis=axis)
    return Rot3D(quats, renormalize=False)

@tf.experimental.dispatch_for_api(tf.gather)
def Rot3D_gather(params: Rot3D, indices, validate_indices=None, axis=None, batch_dims=0) -> Rot3D:
    return Rot3D(tf.gather(params.quat, indices), renormalize=False)


################### API Dispatch for Pose3D ###################

@tf.experimental.dispatch_for_api(tf.broadcast_to)
def Pose3D_broadcast_to(input: Pose3D, shape) -> Pose3D:
    return input.broadcast_to(shape)

@tf.experimental.dispatch_for_api(tf.cast)
def Pose3D_cast(x: Pose3D, dtype) -> Pose3D:
    return Pose3D(x.R, x.t, dtype=dtype)

@tf.experimental.dispatch_for_api(tf.shape)
def Pose3D_shape(input: Pose3D, out_type=tf.int32) -> tf.Tensor:
    return tf.shape(input.t, out_type=out_type)[:-1]

@tf.experimental.dispatch_for_api(tf.stack)
def Pose3D_stack(values: T.List[Pose3D], axis=0) -> Pose3D:
    assert len(values) > 0, "Attempting to stack empty array"

    axis = axis if axis >= 0 else axis - 1
    ts = tf.stack([pose.t for pose in values], axis=axis)
    quats = tf.stack([pose.R.quat for pose in values], axis=axis)

    return Pose3D(R=Rot3D(quats, renormalize=False), t=ts)

@tf.experimental.dispatch_for_api(tf.concat)
def Pose3D_concat(values: T.List[Pose3D], axis=0) -> Pose3D:
    assert len(values) > 0, "Attempting to concat empty array"

    axis = axis if axis >= 0 else axis - 1
    ts = tf.concat([pose.t for pose in values], axis=axis)
    quats = tf.concat([pose.R.quat for pose in values], axis=axis)

    return Pose3D(R=Rot3D(quats, renormalize=False), t=ts)

@tf.experimental.dispatch_for_api(tf.gather)
def Pose3D_gather(params: Pose3D, indices, validate_indices=None, axis=None, batch_dims=0) -> Pose3D:
    return Pose3D(
        R = Rot3D(tf.gather(params.R.quat, indices), renormalize=False),
        t = tf.gather(params.t, indices),
    )
