import math
import os

import tensorflow as tf
import typing as T

from utils.params import ParamDict
from utils.pose3d import Pose3D

class VODataPipe:

    DEFAULT_PARAMS = ParamDict(
        data_root = "/mnt/data/deep_vo_clean",
        num_parallel_calls = 12,
        batch_size = 64,
        num_perturb = 16,
        prefetch_size = 4,
        shuffle_size = 512,
        img_size = (144, 256),
        cycle_length = 16,
        distance_bins = 300,
        # maximum allowed sky portion
        max_sky_ratio = 0.35,
        # maximum allowed close range occlusion
        max_occlusion_ratio = 0.02,
        # maximum distance to be considered as occlusion
        max_occlusion_dist = 4,
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        self.p = params

    def build_train_ds(self, dirname: str = "train") -> tf.data.Dataset:
        file_pattern = os.path.join(self.p.data_root, dirname, "*.tfrecord")
        files = tf.data.Dataset.list_files(file_pattern)
        ds = files.interleave(self._train_interleave,
            cycle_length=self.p.cycle_length,
            num_parallel_calls=self.p.num_parallel_calls,
            deterministic=False,
        )
        ds = ds.shuffle(self.p.shuffle_size)
        ds = ds.batch(self.p.batch_size, drop_remainder=True)
        ds = ds.prefetch(self.p.prefetch_size)

        return ds

    def build_val_ds(self, dirname: str = "validation") -> tf.data.Dataset:
        file_pattern = os.path.join(self.p.data_root, dirname, "*.tfrecord")
        files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
        ds = tf.data.TFRecordDataset(files)
        ds = ds.map(self._process_val, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.p.batch_size)
        ds = ds.prefetch(self.p.prefetch_size)

        return ds

    def build_cal_ds(self) -> tf.data.Dataset:
        file_pattern = os.path.join(self.p.data_root, "calibration", "*.tfrecord")
        files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
        ds = tf.data.TFRecordDataset(files)
        ds = ds.map(self._process_cal, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.p.batch_size)

        return ds

    def _train_interleave(self, file: tf.Tensor) -> tf.data.Dataset:
        ds = tf.data.TFRecordDataset(file)
        ds = ds.map(self._process_train, num_parallel_calls=4)
        ds = ds.unbatch()

        return ds

    def _parse_single_image(self, img_str: tf.Tensor) -> tf.Tensor:
        image = tf.image.decode_image(img_str, channels=3)
        image.set_shape(self.p.img_size + (3,))

        return tf.cast(image, tf.float32) / 127.5 - 1.

    def _image_aug(self, images_vhw3: tf.Tensor) -> tf.Tensor:
        images_vhw3 = tf.image.random_brightness(images_vhw3, 0.1)
        images_vhw3 = tf.image.random_brightness(images_vhw3, 0.3)
        images_vhw3 = tf.image.random_contrast(images_vhw3, .5, 2.)
        images_vhw3 += tf.random.normal(tf.shape(images_vhw3), stddev=0.05)
        return images_vhw3

    def _filter_poses(self,
        data_dict: T.Dict[str, tf.Tensor],
        loc: T.Literal["front", "back", "bottom"],
        train: bool = True,
    ) -> T.Dict[str, tf.Tensor]:
        poses_k7 = tf.io.parse_tensor(data_dict[f"{loc}_poses"], tf.float32)
        images_k = tf.io.parse_tensor(data_dict[f"{loc}_images"], tf.string)
        dists_kb = tf.io.parse_tensor(data_dict[f"{loc}_dists"], tf.float32)
        poses_k7.set_shape((self.p.num_perturb, 7))
        images_k.set_shape((self.p.num_perturb,))
        dists_kb.set_shape((self.p.num_perturb, self.p.distance_bins))

        poses = Pose3D.from_storage(poses_k7)

        # filter out invalid samples
        occlusion_ratio = tf.reduce_sum(dists_kb[:, :self.p.max_occlusion_dist], axis=-1)
        sky_ratio = 1. - tf.reduce_sum(dists_kb, axis=-1)
        valid_k = (occlusion_ratio <= self.p.max_occlusion_ratio) & (sky_ratio <= 0.3)

        images_v = tf.boolean_mask(images_k, valid_k)
        poses_v7 = tf.boolean_mask(poses_k7, valid_k)

        num_samples = tf.shape(images_v)[0]
        indics = tf.range(num_samples)
        if train:
            indics = tf.random.shuffle(indics)
        num_pairs = tf.bitwise.right_shift(num_samples, 1)
        num_samples_even = tf.bitwise.left_shift(num_pairs, 1)
        indics = indics[:num_samples_even]

        images_v = tf.gather(images_v, indics)
        images_vhw3 = tf.map_fn(self._parse_single_image, images_v,
                                fn_output_signature=tf.float32)
        if train:
            images_vhw3 = self._image_aug(images_vhw3)
        poses_v7 = tf.gather(poses_v7, indics)

        return {
            "image1": images_vhw3[:num_pairs],
            "image2": images_vhw3[num_pairs:],
            "pose1": poses_v7[:num_pairs],
            "pose2": poses_v7[num_pairs:],
        }

    def _parse_raw(self,
        example_proto,
        train: bool = True
    ) -> T.Dict[str, tf.Tensor]:
        # build feature descriptor
        locations = ["front", "back", "bottom"]
        feature_desc = {}
        for loc in locations:
            feature_desc[f"{loc}_images"] = tf.io.FixedLenFeature([], tf.string)
            feature_desc[f"{loc}_poses"] = tf.io.FixedLenFeature([], tf.string)
            feature_desc[f"{loc}_dists"] = tf.io.FixedLenFeature([], tf.string)

        # parse example proto
        return tf.io.parse_single_example(example_proto, feature_desc)

    def _parse_func(self,
        example_proto,
        train: bool = True
    ) -> T.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        raw_dict = self._parse_raw(example_proto)

        # parse images (use front and back camera for now)
        front_data = self._filter_poses(raw_dict, "front", train)
        back_data = self._filter_poses(raw_dict, "back", train)
        bottom_data = self._filter_poses(raw_dict, "bottom", train)

        return tf.nest.map_structure(
            lambda x, y, z: tf.concat([x, y, z], axis=0),
            front_data, back_data, bottom_data,
        )

    @tf.function
    def _process_cal(self, example_proto) -> T.Dict[str, tf.Tensor]:
        raw_dict = self._parse_raw(example_proto)
        poses_27 = tf.io.parse_tensor(raw_dict["front_poses"], tf.float32)
        images_2 = tf.io.parse_tensor(raw_dict["front_images"], tf.string)
        poses_27.set_shape((2, 7))
        images_2.set_shape((2,))

        return {
            "image1": self._parse_single_image(images_2[0]),
            "image2": self._parse_single_image(images_2[1]),
            "pose1": poses_27[0],
            "pose2": poses_27[1],
        }

    @tf.function
    def _process_train(self, example_proto) -> T.Dict[str, tf.Tensor]:
        return self._parse_func(example_proto, True)

    @tf.function
    def _process_val(self, example_proto) -> T.Dict[str, tf.Tensor]:
        return tf.nest.map_structure(
            lambda x: x[0],
            self._parse_func(example_proto, False),
        )
