from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf  # TF2
import numpy as np
import os
from pathlib import Path
from itertools import chain
import random
# https://github.com/albu/albumentations
from albumentations import (
    Compose,
    GridDistortion,
    OpticalDistortion,
    OneOf,
    RandomRotate90,
    GaussNoise,
    HueSaturationValue,
    IAASharpen,
    IAAEmboss,
    RandomBrightnessContrast,
    MotionBlur,
    MedianBlur,
    IAAAdditiveGaussianNoise,
    Blur,
    ShiftScaleRotate,
    HorizontalFlip,
    IAAPiecewiseAffine
)


class DatasetLoader(object):
    def __init__(self, buffer_size, batch_size, output_resolution=256, max_load_resolution=320):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.output_resolution = output_resolution
        self.max_load_resolution = max_load_resolution

    def get_all_images(self, image_path, supported_extensions=None):
        if supported_extensions is None:
            supported_extensions = [".jpg", ".jpeg", ".JPEG", ".png", ".webp"]
        if isinstance(image_path, list):
            img_gen = []
            for p in image_path:
                for ext in supported_extensions:
                    img_gen.append(Path(p).glob(f"*{ext}"))
        else:
            img_gen = [Path(image_path).glob(f"*{ext}") for ext in supported_extensions]
        return chain(*img_gen)

    def setup_datasets(self, image_dir_name="image",
                       mask_dir_name="mask",
                       split_dir_name="train",
                       datasets_path="./data", shuffle_files=True):
        dataset_path_with_split_name = Path(datasets_path) / split_dir_name
        dataset_paths = sorted([p for p in dataset_path_with_split_name.glob("*/") if p.is_dir()])

        image_dirs = []
        for dataset_path in dataset_paths:
            image_dirs.append(dataset_path / image_dir_name)

        image_full_pathes = []
        mask_full_pathes = []
        for image in sorted(self.get_all_images(image_dirs)):
            image_path = str(image)
            cur_path, filename = os.path.split(image_path)
            mask_path = os.path.join(cur_path[:-len(image_dir_name)] + mask_dir_name,
                                     os.path.splitext(filename)[0] + ".png")
            if not Path(mask_path).exists():
                continue
            assert Path(mask_path).exists(), f"{mask_path} not found"
            image_full_pathes.append(image_path)
            mask_full_pathes.append(mask_path)
        if shuffle_files:
            shuffled_data = list(zip(image_full_pathes, mask_full_pathes))
            random.shuffle(shuffled_data)
            image_full_pathes, mask_full_pathes = zip(*shuffled_data)
        return list(image_full_pathes), list(mask_full_pathes)

    def _cv_load(self, filename):
        import cv2
        filename = filename.numpy().decode("UTF-8")
        image = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return np.asarray(image, np.float32)

    def _image_dimensions(self, image):
        if image.get_shape().is_fully_defined():
            return image.get_shape().as_list()
        else:
            rank = len(image.get_shape().as_list())
            static_shape = image.get_shape().with_rank(rank).as_list()
            dynamic_shape = tf.unstack(tf.shape(image), rank)
            return [
                s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
            ]

    def resize_preserve_aspect_ratio(self, image_data, max_load_output_resolution=640):
        image = tf.cast(image_data, dtype=tf.float32)
        current_height, current_width = self._image_dimensions(image)[:2]
        resize_ratio = tf.cast(max_load_output_resolution / tf.maximum(current_width, current_height), dtype=tf.float32)
        scaled_height_const = tf.cast(tf.round(resize_ratio * tf.cast(current_height, tf.float32)), tf.int32)
        scaled_width_const = tf.cast(tf.round(resize_ratio * tf.cast(current_width, tf.float32)), tf.int32)
        resized_image_tensor = tf.image.resize(image, size=[scaled_height_const, scaled_width_const])
        return resized_image_tensor

    def load_aug_images(self, image_file, label_file, apply_albumentations=False):
        def aug_dataset(image, mask):
            image = image.numpy().astype(np.float32)
            mask = mask.numpy().astype(np.float32)

            aug = Compose([
                RandomRotate90(),
                OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.2),
                OneOf([
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                OneOf([
                    OpticalDistortion(p=0.3),
                    GridDistortion(p=0.1),
                    IAAPiecewiseAffine(p=0.3),
                ], p=0.2),
                OneOf([
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(),
                ], p=0.3),
                HueSaturationValue(p=0.3),
                HorizontalFlip(),
            ], p=0.5)
            augmented = aug(image=image, mask=mask)
            mask = augmented['mask']
            image = augmented['image']
            return image.astype(np.float32), mask.astype(np.float32)

        image = tf.py_function(self._cv_load, [image_file], tf.float32)
        image.set_shape([None, None, 3])
        label = tf.image.decode_image(tf.io.read_file(label_file), channels=1)
        label.set_shape([None, None, 1])
        resize_image = self.resize_preserve_aspect_ratio(image,
                                                         max_load_output_resolution=self.max_load_resolution)
        image_shape = tf.shape(resize_image)
        resize_label = tf.image.resize(tf.cast(label, tf.float32), size=[image_shape[0], image_shape[1]])
        if apply_albumentations:
            augmented_image, augmented_label = tf.py_function(aug_dataset,
                                                              [resize_image, resize_label],
                                                              [tf.float32, tf.float32])
            augmented_label.set_shape([None, None, 1])
            augmented_image.set_shape([None, None, 3])
        else:
            augmented_image = resize_image
            augmented_label = resize_label
        augmented_label = tf.image.resize(augmented_label / 255., size=[self.output_resolution, self.output_resolution])
        augmented_image = tf.image.resize(augmented_image / 255., size=[self.output_resolution, self.output_resolution])
        return augmented_image, augmented_label

    def preprocess_test(self, image_file, label_file):
        return self.load_aug_images(image_file, label_file, apply_albumentations=False)

    def preprocess_train(self, image_file, label_file):
        return self.load_aug_images(image_file, label_file, apply_albumentations=True)

    def load(self, datasets_path="./data", train_dir_name="train", test_dir_name="test"):
        train_images, train_labels = self.setup_datasets(split_dir_name=train_dir_name, datasets_path=datasets_path)
        train_num_datasets = len(train_images)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.with_options(options)
        train_dataset = train_dataset.map(
            self.preprocess_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
            tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)
        test_images, test_labels = self.setup_datasets(split_dir_name=test_dir_name, datasets_path=datasets_path)
        test_num_datasets = len(test_images)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_dataset = test_dataset.with_options(options)
        test_dataset = test_dataset.map(self.preprocess_test,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
            tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)

        return train_dataset, test_dataset, train_num_datasets, test_num_datasets
