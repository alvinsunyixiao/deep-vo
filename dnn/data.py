import os

import tensorflow as tf
import typing as T

from utils.params import ParamDict

class VODataPipe:

    DEFAULT_PARAMS = ParamDict(
        data_root = "/data/tfrecords/deep_vo",
        num_parallel_reads = 16,
        num_parallel_calls = 8,
        batch_size = 64,
        num_perturb = 16,
        prefetch_size = 4,
        img_size = (144, 256),
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        self.p = params

    def build_train_ds(self) -> tf.data.Dataset:
        file_pattern = os.path.join(self.p.data_root, "train", "*.tfrecord")
        files = tf.data.Dataset.list_files(file_pattern)
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=self.p.num_parallel_reads)
        ds = ds.map(self._process_train, num_parallel_calls=self.p.num_parallel_calls)
        ds = ds.batch(self.p.batch_size)
        ds = ds.prefetch(self.p.prefetch_size)

        return ds

    def build_val_ds(self) -> tf.data.Dataset:
        file_pattern = os.path.join(self.p.data_root, "validation", "*.tfrecord")
        files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
        ds = tf.data.TFRecordDataset(files)
        ds = ds.map(self._process_val, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.p.batch_size)
        ds = ds.prefetch(self.p.prefetch_size)

        return ds

    def _parse_func(self, example_proto) -> T.Tuple[tf.Tensor, tf.Tensor]:
        # build feature descriptor
        locations = ["front", "back", "bottom"]
        feature_desc = {}
        for loc in locations:
            feature_desc[f"{loc}_images"] = tf.io.FixedLenFeature([], tf.string)
            feature_desc[f"{loc}_poses"] = tf.io.FixedLenFeature([], tf.string)

        # parse example proto
        raw_dict = tf.io.parse_single_example(example_proto, feature_desc)

        # parse images (use front camera for now)
        images_k = tf.io.parse_tensor(raw_dict["front_images"], tf.string)
        poses_k7 = tf.io.parse_tensor(raw_dict["front_poses"], tf.float32)
        images_k.set_shape((self.p.num_perturb,))
        poses_k7.set_shape((self.p.num_perturb, 7))

        return images_k, poses_k7

    @tf.function
    def _process_train(self, example_proto) -> T.Tuple[tf.Tensor, tf.Tensor]:
        images_k, poses_k7 = self._parse_func(example_proto)

        # randomly pick two perturbed poses
        perturb_indics = tf.random.shuffle(tf.range(tf.shape(images_k)[0], dtype=tf.int32))
        i1 = perturb_indics[0]
        i2 = perturb_indics[1]

        # combine image pairs
        image1 = tf.cast(tf.io.decode_image(images_k[i1], 3), tf.float32)
        image2 = tf.cast(tf.io.decode_image(images_k[i2], 3), tf.float32)
        image1.set_shape(self.p.img_size + (3,))
        image2.set_shape(self.p.img_size + (3,))

        return {
            "image1": image1 / 127.5 - 1.,
            "image2": image2 / 127.5 - 1.,
            "pose1": poses_k7[i1],
            "pose2": poses_k7[i2],
        }

    @tf.function
    def _process_val(self, example_proto) -> T.Tuple[tf.Tensor, tf.Tensor]:
        images_k, poses_k7 = self._parse_func(example_proto)

        # combine image pairs
        image1 = tf.cast(tf.io.decode_image(images_k[0], 3), tf.float32)
        image2 = tf.cast(tf.io.decode_image(images_k[1], 3), tf.float32)
        image1.set_shape(self.p.img_size + (3,))
        image2.set_shape(self.p.img_size + (3,))

        return {
            "image1": image1 / 127.5 - 1.,
            "image2": image2 / 127.5 - 1.,
            "pose1": poses_k7[0],
            "pose2": poses_k7[1],
        }
