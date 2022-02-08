import os

import tensorflow as tf
import typing as T

from utils.params import ParamDict
from utils.pose3d import Pose3D

class VODataPipe:

    DEFAULT_PARAMS = ParamDict(
        data_root = "/mnt/data/deep_vo",
        num_parallel_reads = 8,
        batch_size = 64,
        num_perturb = 16,
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        self.p = params

    def build_train_ds(self) -> tf.data.Dataset:
        file_pattern = os.path.join(self.p.data_root, "train", "*.tfrecord")
        files = tf.data.Dataset.list_files(file_pattern)
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=self.p.num_parallel_reads)
        ds = ds.map(self._process_train, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.p.batch_size)

        return ds

    def build_val_ds(self) -> tf.data.Dataset:
        file_pattern = os.path.join(self.p.data_root, "validation", "*.tfrecord")
        files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
        ds = tf.data.TFRecordDataset(files)
        ds = ds.map(self._process_val, num_parallel_calls=tf.data.AUTOTUNE)

        return ds

    def _parse_func(self, example_proto) -> T.Tuple[tf.Tensor, Pose3D]:
        # build feature descriptor
        locations = ["front", "back", "bottom"]
        feature_desc = {}
        for loc in locations:
            feature_desc[f"{loc}_images"] = tf.io.FixedLenFeature([], tf.string)
            feature_desc[f"{loc}_poses"] = tf.io.FixedLenFeature([], tf.string)

        # parse example proto
        raw_dict = tf.io.parse_single_example(example_proto, feature_desc)

        # parse images
        images_lk = tf.stack([
            tf.io.parse_tensor(raw_dict[f"{loc}_images"], tf.string) for loc in locations])
        poses_lk7 = tf.stack([
            tf.io.parse_tensor(raw_dict[f"{loc}_poses"], tf.float32) for loc in locations])
        images_lk.set_shape((len(locations), self.p.num_perturb))
        poses_lk7.set_shape((len(locations), self.p.num_perturb, 7))

        return images_lk, Pose3D.from_storage(poses_lk7)

    def _process_train(self, example_proto) -> T.Tuple[tf.Tensor, tf.Tensor]:
        images_lk, poses_lk = self._parse_func(example_proto)

        # randomly pick a camera location
        loc = tf.random.uniform((), maxval=3, dtype=tf.int32) # 3 == len(locations)
        images_k = images_lk[loc]
        poses_k = poses_lk[loc]

        # randomly pick two perturbed poses
        perturb_indics = tf.random.shuffle(tf.range(tf.shape(images_k)[0], dtype=tf.int32))
        i1 = perturb_indics[0]
        i2 = perturb_indics[1]

        # combine image pairs
        image1 = tf.cast(tf.io.decode_image(images_k[i1], 3), tf.float32)
        image2 = tf.cast(tf.io.decode_image(images_k[i2], 3), tf.float32)
        image1.set_shape((None, None, 3))
        image2.set_shape((None, None, 3))
        w_T_c1 = poses_k[i1]
        w_T_c2 = poses_k[i2]
        c1_T_c2 = w_T_c1.inv() @ w_T_c2

        # convert to model input / output
        x = tf.concat([image1, image2], axis=-1)
        y = c1_T_c2.to_se3()

        return x, y

    def _parse_images(self, images_m: tf.Tensor) -> tf.Tensor:
        def decode_image(img_str):
            img = tf.io.decode_image(img_str, 3)
            img.set_shape((None, None, 3))
            return img

        images_mhw3 = tf.nest.map_structure(decode_image, tf.unstack(images_m))
        images_mhw3 = tf.stack(images_mhw3)
        images_mhw3 = tf.cast(images_mhw3, tf.float32)

        return images_mhw3

    def _process_val(self, example_proto) -> T.Tuple[tf.Tensor, tf.Tensor]:
        images_lk, poses_lk = self._parse_func(example_proto)

        # generate pairs with adjacent samples
        images1_mhw3 = self._parse_images(tf.reshape(images_lk[:, :-1], (-1,)))
        images2_mhw3 = self._parse_images(tf.reshape(images_lk[:, 1:], (-1,)))
        w_T_c1 = poses_lk[:, :-1].flatten()
        w_T_c2 = poses_lk[:, 1:].flatten()
        c1_T_c2 = w_T_c1.inv() @ w_T_c2

        # convert to model input / output (batched)
        x = tf.concat([images1_mhw3, images2_mhw3], axis=-1)
        y = c1_T_c2.to_se3()

        return x, y
