import math
import os

import tensorflow as tf
import typing as T

from utils.params import ParamDict
from utils.pose3d import Pose3D

class DepthDataPipe:

    DEFAULT_PARAMS = ParamDict(
        data_root = "/home/alvin/data/tfrecords/deep_depth",
        batch_size = 32,
        prefetch_size = 4,
        shuffle_size = 512,
        cycle_length = 16,
        img_size = (160, 256),
        max_depth=300.,
        num_parallel_reads=8,
        num_parallel_calls=8,
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS) -> None:
        self.p = params

    def build_train_ds(self, dirname: str = "train") -> tf.data.Dataset:
        file_pattern = os.path.join(self.p.data_root, dirname, "*.tfrecord")
        files = tf.data.Dataset.list_files(file_pattern)
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=self.p.num_parallel_reads)
        ds = ds.map(self._process_train, num_parallel_calls=self.p.num_parallel_calls)
        ds = ds.shuffle(self.p.shuffle_size)
        ds = ds.batch(self.p.batch_size, drop_remainder=True)
        ds = ds.prefetch(self.p.prefetch_size)

        return ds

    def build_val_ds(self, dirname: str = "validation") -> tf.data.Dataset:
        file_pattern = os.path.join(self.p.data_root, dirname, "*.tfrecord")
        files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
        ds = tf.data.TFRecordDataset(files)
        ds = ds.map(self._process_val, num_parallel_calls=self.p.num_parallel_calls)
        ds = ds.batch(self.p.batch_size)
        ds = ds.prefetch(self.p.prefetch_size)

        return ds

    def _image_aug(self, images_hw3: tf.Tensor) -> tf.Tensor:
        if tf.random.uniform(()) < 0.3:
            images_hw3 = tf.image.random_brightness(images_hw3, 0.2)
        if tf.random.uniform(()) < 0.3:
            images_hw3 = tf.image.random_contrast(images_hw3, .5, 2.)
        if tf.random.uniform(()) < 0.3:
            images_hw3 = tf.image.random_saturation(images_hw3, 0.5, 2.)
        if tf.random.uniform(()) < 0.3:
            images_hw3 = tf.image.random_hue(images_hw3, 0.1)
        if tf.random.uniform(()) < 0.3:
            stddev = tf.random.uniform((), maxval=0.06)
            # same noise all channel
            if tf.random.uniform(()) < 0.5:
                noise = tf.random.normal(tf.shape(images_hw3)[:-1], stddev=stddev)
                noise = noise[..., tf.newaxis]
            # independent noise across channels
            else:
                noise = tf.random.normal(tf.shape(images_hw3), stddev=stddev)
            images_hw3 += noise

        return images_hw3

    def _parse_raw(self,
        example_proto,
        train: bool = True
    ) -> T.Dict[str, tf.Tensor]:
        # build feature descriptor
        feature_desc = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "pose": tf.io.FixedLenFeature([], tf.string),
            "depth": tf.io.FixedLenFeature([], tf.string),
        }

        # parse example proto
        data_dict = tf.io.parse_single_example(example_proto, feature_desc)

        data_dict["image"] = tf.io.parse_tensor(data_dict["image"], tf.string)
        data_dict["image"] = tf.io.decode_image(data_dict["image"], channels=3)
        data_dict["pose"] = tf.io.parse_tensor(data_dict["pose"], tf.float32)
        data_dict["depth"] = tf.io.parse_tensor(data_dict["depth"], tf.float32)

        return data_dict

    def _center_crop_box(self, data_dict: T.Dict[str, tf.Tensor]) -> tf.Tensor:
        h_in = tf.cast(tf.shape(data_dict["image"])[0], tf.float32)
        w_in = tf.cast(tf.shape(data_dict["image"])[1], tf.float32)

        h_out = float(self.p.img_size[0])
        w_out = float(self.p.img_size[1])

        h_ratio = h_out / h_in
        w_ratio = w_out / w_in

        ratio = tf.maximum(h_ratio, w_ratio)

        h_in_norm = h_ratio / ratio
        w_in_norm = w_ratio / ratio

        box = tf.stack([0.5 - h_in_norm / 2., 0.5 - w_in_norm / 2.,
                        0.5 + h_in_norm / 2., 0.5 + w_in_norm / 2.])

        return box

    def _crop_single_image(self, image: tf.Tensor, box: tf.Tensor) -> tf.Tensor:
        return tf.image.crop_and_resize(
            image=image[tf.newaxis],
            boxes=box[tf.newaxis],
            box_indices=tf.constant([0], dtype=tf.int32),
            crop_size=self.p.img_size,
        )[0]

    def _parse_func(self,
        example_proto,
        train: bool = True
    ) -> T.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        data_dict = self._parse_raw(example_proto)

        # crop and resize
        if train:
            # TODO(alvin): add random crop + rotation for training
            crop_box = self._center_crop_box(data_dict)
        else:
            crop_box = self._center_crop_box(data_dict)
        data_dict["image"] = self._crop_single_image(data_dict["image"], crop_box)
        data_dict["depth"] = self._crop_single_image(data_dict["depth"], crop_box)

        # augmentation
        data_dict["image"] /= 255.
        data_dict["depth"] = tf.clip_by_value(data_dict["depth"], 0., self.p.max_depth)
        if train:
            data_dict["image"] = self._image_aug(data_dict["image"])

        return data_dict

    @tf.function
    def _process_train(self, example_proto) -> T.Dict[str, tf.Tensor]:
        return self._parse_func(example_proto, True)

    @tf.function
    def _process_val(self, example_proto) -> T.Dict[str, tf.Tensor]:
        return self._parse_func(example_proto, False)
