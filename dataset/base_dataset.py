# -*-coding:utf-8 -*-
import os
import pickle
import json
import tensorflow as tf
from tools.logger import logger


class BaseDataset(object):
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.raw_data = []
        self.samples = []
        self.dtypes = {}
        self.shapes = {}
        self.pads = {}
        self.feature_names = []
        self.label_names = []

    @property
    def sample_size(self):
        return len(self.raw_data)

    @property
    def steps_per_epoch(self):
        return int(self.sample_size/self.batch_size)

    def build_proto(self):
        raise NotImplementedError

    def build_single_feature(self, data):
        raise NotImplementedError

    def load_data(self, file_name):
        """
        By Default, each data folder has 3 files: train/valid/test
        """
        raw_data = []
        with open(os.path.join(self.data_dir, file_name+ '.txt'), 'rb') as f:
            for line in f:
                raw_data.append(json.loads(line.strip()))
        self.raw_data = raw_data

    def build_serving_proto(self):
        receiver_tensor = {}
        for i in self.feature_names:
            receiver_tensor[i] = tf.placeholder(dtype=self.dtypes[i],
                                                shape=[None]+self.shapes[i],
                                                name=i)
        return tf.estimator.export.ServingInputReceiver(receiver_tensor, receiver_tensor)


class SampleCache(object):
    def __init__(self, data_dir, enable_cache):
        self.data_dir = data_dir
        self.enable_cache = enable_cache

    def cache_file(self, file_name):
        return os.path.join(self.data_dir, 'cache_{}.pkl'.format(file_name))

    def _load(self, file_name):
        file = self.cache_file(file_name)
        try:
            logger.info('Loading Cache from {}'.format(file))
            with open(file, 'rb') as f:
                sample = pickle.load(f)
            return sample
        except Exception as e:
            logger.inof(e)
            return []

    def dump(self, samples, file_name):
        file = self.cache_file(file_name)
        with open(file, 'wb') as f:
            pickle.dump(samples, f)
        logger.info('Dumping Cache to {}'.format(file))

    def load(self, file_name):
        if self.enable_cache:
            return self._load(file_name)
        else:
            return []


class GeneratorDataset(BaseDataset):
    def __init__(self, data_dir, batch_size, enable_cache):
        super(GeneratorDataset, self).__init__(data_dir, batch_size)
        self.cacher = SampleCache(data_dir, enable_cache)

    def build_feature(self, file_name):
        """
        If enable_cache, 先尝试读取cache，cache为空再重新生成样本
        """
        self.load_data(file_name)

        self.samples = self.cacher.load(file_name)
        if not self.samples:
            for data in self.raw_data:
                self.samples.append(self.build_single_feature(data))

        self.cacher.dump(self.samples, file_name)

    def build_generator(self):
        for s in self.samples:
            feature = {i: s[i] for i in self.feature_names}
            label = {i:s[i] for i in self.label_names}
            yield feature, label

    def build_input_fn(self, is_predict=False, unbatch=False):
        def helper():
            shapes = ({i: self.shapes[i] for i in self.feature_names},
                      {i: self.shapes[i] for i in self.label_names})
            dtypes = ({i: self.dtypes[i] for i in self.feature_names},
                      {i: self.dtypes[i] for i in self.label_names})
            pads = ({i: self.pads[i] for i in self.feature_names},
                      {i: self.pads[i] for i in self.label_names})
            dataset = tf.data.Dataset.from_generator(
                lambda : self.build_generator(),
                output_shapes=shapes,
                output_types=dtypes
            )
            if not is_predict:
                dataset = dataset.shuffle(int(self.batch_size * 5)).repeat()

            if not unbatch:
                dataset = dataset.padded_batch(self.batch_size, shapes, pads).\
                    prefetch(tf.data.experimental.AUTOTUNE)

            return dataset
        return helper





