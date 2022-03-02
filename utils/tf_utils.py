import os

import typing as T
import numpy as np
import tensorflow as tf

def set_mixed_precision():
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)

def set_tf_memory_growth(mode: bool = True):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, mode)

def bytes_to_feature(value: T.Union[str, bytes, tf.Tensor]) -> tf.train.Feature:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if type(value) == str:
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tensor_to_feature(tensor: T.Union[np.ndarray, tf.Tensor]) -> tf.train.Feature:
    return bytes_to_feature(tf.io.serialize_tensor(tensor))

def const_to_feature(data: T.List[T.Any], dtype=None):
    return tensor_to_feature(tf.constant(data, dtype=dtype))

class ShardedTFRecordWriter:
    def __init__(self, output_dir: str, shard_size: int):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self._shard_cnt = 0
        self._sample_cnt = 0

    def __enter__(self):
        self._writer = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writer is not None:
            self._shard_cnt += 1
            self._writer.close()

    def write(self, record):
        if self._sample_cnt % self.shard_size == 0:
            if self._writer is not None:
                self._writer.close()
            self._writer = tf.io.TFRecordWriter(
                os.path.join(self.output_dir, f"shard-{self._shard_cnt}.tfrecord"))
            self._shard_cnt += 1
        self._writer.write(record)
        self._sample_cnt += 1

