# -*-coding:utf-8 -*-
import tensorflow.compat.v1 as tf
import os

from dataset.base_dataset import TFRecordDataset


class ImgDataset(TFRecordDataset):
    def __init__(self, data_dir, batch_size, img_size, img_preprocess):
        super(ImgDataset, self).__init__(data_dir, batch_size)
        self.channel_size = 3  # by Default no need to change
        self.img_size = img_size
        self.preprocess = img_preprocess  # only need in training not in TFRecord Dump

    def build_proto(self):
        self.dtypes.update({
            'img_array': tf.float32,
            'img_bytes': tf.string,
            'label': tf.int32
        })
        self.shapes.update({
            'img_array': [None, None, self.channel_size],
            'label': []
        })
        self.pads.update({
            'img_array': 0.0,
            'label': 0

        })
        self.feature_names +=['img_array']
        self.label_names +=['label']

    def build_single_feature(self, data):
        """
        用gid去读区对应目录下images/gid.jpg文件, 返回img binary
        这里没有直接convert成Numpy再tostring，用binary存储size要小10倍左右
        """
        gid = data['gid']
        with tf.gfile.GFile(os.path.join(self.data_dir, 'images', gid + '.jpg'), 'rb') as fid:
            img_bytes = fid.read()
        return {
            'img_bytes': img_bytes,
            'label': int(data['label'])
        }

    def parser(self, line, is_predict):
        proto = {
            'img_bytes': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64)
        }
        feature = tf.parse_single_example(line, features=proto)
        img_array = tf.io.decode_jpeg(feature.pop('img_bytes'), channels=self.channel_size)  # image of arbitary size
        img_array = self.preprocess(img_array, self.img_size, self.img_size,
                                    is_training=(not is_predict))
        feature['img_array'] = tf.reshape(img_array, [self.img_size, self.img_size, self.channel_size])
        label = {'label': tf.cast(feature.pop('label'), tf.int32)}
        return feature, label

    def update_params(self, train_params):
        train_params.update({
            'sample_size': self.sample_size,
            'steps_per_epoch': self.steps_per_epoch,
            'num_train_steps': int(self.steps_per_epoch * train_params['epoch_size'])
        })
        return train_params
