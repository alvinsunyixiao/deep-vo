import glob
import functools
import os
import random
import typing as T

import numpy as np
import tensorflow as tf

from utils.params import ParamDict
from utils.pose3d import Pose3D, Rot3D
from utils.camera import PinholeCam

class DataAirsimSeq:

    DEFAULT_PARAMS = ParamDict(
        # root to data generated from sim/stereo_record.py
        data_root = "/data/airsim/multirotor_highres",
        # train portion
        train_split = 0.9,
        # camera intrinsics
        #camera = PinholeCam(
        #    focal = np.array([73.9, 73.9]),
        #    center = np.array([128., 80.]),
        #),
        camera = PinholeCam(
            focal = np.array([184.75, 184.75]),
            center = np.array([320, 180]),
        )
        crop_size = (160, 320),
        # stereo camera extrinsic in EDN coordinate
        left_T_right = Pose3D(
            orientation = Rot3D.identity(),
            position = np.array([.25, 0., 0.]),
        ),
        # parallel data pipeline
        num_parallel_calls=tf.data.AUTOTUNE,
        batch_size = 8,
    )

    def __init__(self, params: ParamDict=DEFAULT_PARAMS) -> None:
        self.p = params

        airsim_recs = glob.glob(
            os.path.join(self.p.data_root, "**/airsim_rec.txt"), recursive=True)

        # create data dict and shuffle
        self.all_data = []
        for one_seq in map(self._parse_file, airsim_recs):
            self.all_data.extend(one_seq)
        random.shuffle(self.all_data)

        # train / validation split
        train_size = int(len(self.all_data) * self.p.train_split)
        self.train_data = self.all_data[:train_size]
        self.val_data = self.all_data[train_size:]

        # build tf datasets
        self.train_ds = self.build_dataset(self.train_data)
        self.val_ds = self.build_dataset(self.val_data)

    def build_train_ds(self) -> tf.data.Dataset:
        return self.build_dataset(self.train_data, training=True)

    def build_val_ds(self) -> tf.data.Dataset:
        return self.build_dataset(self.val_data, training=False)

    def _read_image(self, img_path: tf.Tensor) -> tf.Tensor:
        img_file = tf.io.read_file(img_path)
        img = tf.io.decode_image(img_file, channels=3, dtype=tf.float32)
        return img

    def _augment_images(self, images: tf.Tensor) -> tf.Tensor:
        if tf.random.uniform(()) < 0.3:
            images = tf.image.random_brightness(images, 0.2)
        if tf.random.uniform(()) < 0.3:
            images = tf.image.random_contrast(images, .5, 2.)
        if tf.random.uniform(()) < 0.3:
            images = tf.image.random_saturation(images, 0.5, 2.)
        if tf.random.uniform(()) < 0.3:
            images = tf.image.random_hue(images, 0.1)

        return images

    def _map_func(self,
        data_dict: T.Dict[str, tf.Tensor],
        img_aug: bool = True,
    ) -> T.Dict[str, tf.Tensor]:
        # randomly select source and target frame from left / right
        if tf.random.uniform(()) < 0.5:
            image_paths = tf.concat([data_dict["left_imgs"], [data_dict["right_imgs"][1]]], axis=0)
            src_T_tgt = self.p.left_T_right.inv().to_storage()
        else:
            image_paths = tf.concat([data_dict["right_imgs"], [data_dict["left_imgs"][1]]], axis=0)
            src_T_tgt = self.p.left_T_right.to_storage()

        images = tf.map_fn(self._read_image, image_paths, fn_output_signature=tf.float32)

        # generate crop boxes
        if img_aug:
            # random crop box
            offset_h = tf.random.uniform((), maxval=tf.shape(images)[1] - self.p.crop_size[0], dtype=tf.int32)
            offset_w = tf.random.uniform((), maxval=tf.shape(images)[2] - self.p.crop_size[1], dtype=tf.int32)
        else:
            # center crop box
            offset_h = (tf.shape(images)[1] - self.p.crop_size[0]) // 2
            offset_w = (tf.shape(images)[2] - self.p.crop_size[1]) // 2
        images = tf.image.crop_to_bounding_box(images, offset_h, offset_w, self.p.crop_size[0], self.p.crop_size[1])

        if img_aug:
            images = self._augment_images(images)

        return {
            "src_T_tgt": src_T_tgt,
            "cam_param": self.p.camera.to_storage(),
            "img_tgt_prev": images[0],
            "img_tgt": images[1],
            "img_tgt_next": images[2],
            "img_src": images[3],
        }

    def build_dataset(self,
        data: T.List[T.Dict[str, str]],
        training: bool = True
    ) -> tf.data.Dataset:
        data_dict = {key: [] for key in data[0].keys()}
        for d in data:
            for key in d:
                data_dict[key].append(d[key])
        dataset = tf.data.Dataset.from_tensor_slices(data_dict)
        dataset = dataset.shuffle(len(data))
        dataset = dataset.map(
            map_func=functools.partial(self._map_func, img_aug=training),
            num_parallel_calls=self.p.num_parallel_calls,
            deterministic=False,
        )
        dataset = dataset.batch(self.p.batch_size, drop_remainder=True)

        return dataset

    def _parse_file(self, filepath: str) -> T.List[T.Dict[str, tf.Tensor]]:
        data_dir = os.path.dirname(filepath)
        image_dir = os.path.join(data_dir, "images")

        data = {"left_img": [], "right_img": []}
        with open(filepath, "r") as f:
            lines = f.read().splitlines()
            lines = lines[1:] # remove first line
            num_lines = len(lines)

            for line in lines:
                elems = line.split("\t")
                image_files = elems[-1].split(";")
                data["left_img"].append(os.path.join(image_dir, image_files[0]))
                data["right_img"].append(os.path.join(image_dir, image_files[1]))

        ret = []
        for i in range(1, num_lines - 1):
            ret.append({
                "left_imgs": tf.convert_to_tensor(data["left_img"][i - 1:i + 2]),
                "right_imgs": tf.convert_to_tensor(data["right_img"][i - 1:i + 2]),
            })

        return ret

