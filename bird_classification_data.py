import hpo
import tensorflow as tf
import os
import glob
import re

class BirdData(hpo.Data):
    def __init__(self, data_dir, cache_path, training_batch_size, validation_batch_size, test_batch_size):
        super().__init__()

        self._data_dir = data_dir
        self._cache_path = cache_path
        self._training_batch_size = training_batch_size
        self._validation_batch_size = validation_batch_size
        self._test_batch_size = test_batch_size

        self._training_image_count = 0
        self._validation_image_count = 0
        self._test_image_count = 0

        self._class_labels = self._get_class_labels()

        self._training_data = None
        self._validation_data = None
        self._test_data = None

    def _get_class_labels(self):
        label_file = open(os.path.join(self._data_dir, "BIRDS-150.txt"), "r")
        labels = label_file.read().split(",")
        for i in range(len(labels)):
            labels[i] = re.sub(r"[0-9]+:", "", labels[i])
        label_file.close()

        return labels[:-1]

    def load(self):
        def get_jpeg_from_filepath(path):
            def get_class_label_from_filepath(path):
                folders = tf.strings.split(path, os.path.sep)
                return folders[-2]

            class_label = get_class_label_from_filepath(path)
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [224, 224])

            return img, class_label == self._class_labels

        def prepare_dataset(dataset, batch_size, cache=True, repeat=True, prefetch=True, shuffle=True, shuffle_seed=42, shuffle_buffer_size=1000):
            if (cache):
                if (isinstance(cache, str)):
                    print("Opening cache or creating (%s)." % (cache))
                    dataset = dataset.cache(cache)
                else:
                    print("No cache path provided. Loading into memory.")
                    dataset = dataset.cache()
            else:
                print("Not caching data. This may be slow.")

            if shuffle:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)

            if repeat:
                dataset = dataset.repeat()

            if batch_size > 0:
                dataset = dataset.batch(batch_size)

            if prefetch:
                dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            return dataset

        training_paths = [os.path.join(os.path.join(os.path.join(self._data_dir, "train"), label), "*.jpg") for label in self._class_labels]
        validation_paths = [os.path.join(os.path.join(os.path.join(self._data_dir, "valid"), label), "*.jpg") for label in self._class_labels]
        testing_paths = [os.path.join(os.path.join(os.path.join(self._data_dir, "test"), label), "*.jpg") for label in self._class_labels]

        image_count = sum([len(glob.glob(training_path)) for training_path in training_paths])
        image_count += sum([len(glob.glob(validation_path)) for validation_path in validation_paths])
        image_count += sum([len(glob.glob(testing_path)) for testing_path in testing_paths])

        self._training_image_count = int(image_count * 0.7)
        self._validation_image_count = int(image_count * 0.3)

        self._training_image_count += (image_count - (self._training_image_count + self._validation_image_count)) # add an extra if rounding error

        print("NUMBER OF TRAINING SAMPLES:", self._training_image_count)
        print("NUMBER OF VALIDATION SAMPLES:", self._validation_image_count)

        filepaths_ds = tf.data.Dataset.list_files(training_paths, seed=42, shuffle=True)
        filepaths_ds = filepaths_ds.concatenate(tf.data.Dataset.list_files(validation_paths, seed=42, shuffle=True))
        filepaths_ds = filepaths_ds.concatenate(tf.data.Dataset.list_files(testing_paths, seed=42, shuffle=True))

        training_filepaths_ds = filepaths_ds.take(self._training_image_count)
        validation_filepaths_ds = filepaths_ds.skip(self._training_image_count)

        training_images = training_filepaths_ds.map(get_jpeg_from_filepath, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_images = validation_filepaths_ds.map(get_jpeg_from_filepath, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self._training_data = prepare_dataset(training_images, self._training_batch_size, cache=os.path.join(self._cache_path, "training_images.tfcache"))
        self._validation_data = prepare_dataset(validation_images, self._validation_batch_size, cache=os.path.join(self._cache_path, "validation_images.tfcache"))


    def training_steps(self):
        return self._training_image_count // self._training_batch_size

    def validation_steps(self):
        return self._validation_image_count // self._validation_batch_size

    def test_steps(self):
        return self._test_image_count // self._test_batch_size

    def training_data(self):
        return self._training_data

    def validation_data(self):
        return self._validation_data

    def test_data(self):
        return self._test_data