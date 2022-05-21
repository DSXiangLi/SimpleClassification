# -*-coding:utf-8 -*-
import tensorflow.compat.v1 as tf
import os


class TFRecordDump(object):
    converters = {}

    def __init__(self, data_dir, instance_per_shard, tfrecord_type):
        self.data_dir = data_dir
        self.instance_per_shard = instance_per_shard
        self.tfrecord_type = tfrecord_type

    def register(dtype):
        def wrapper(func):
            TFRecordDump.converters[dtype] = func
        return wrapper

    def convert(self, feature):
        feat = {}
        for k, v in self.tfrecord_type.items():
            feat[k] = self.converters[v](feature[k])
        return feat

    def dump(self, feature_list, file_name, shard):
        with tf.io.TFRecordWriter(os.path.join(self.data_dir, file_name + '_{}.tfrecord'.format(shard))) as writer:
            for feature in feature_list:
                features = tf.train.Features(
                    feature=self.convert(feature)
                )
                example = tf.train.Example(features=features)
                writer.write(example.SeralizeToString())


@TFRecordDump.register(tf.string)
def string_feature(value):
    if not isinstance(value, list):
        value = [value]
    value = [i if isinstance(i, bytes) else bytes(i, encoding='UTF-8') for i in value ]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


@TFRecordDump.register(tf.int64)
def int_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


@TFRecordDump.register(tf.float32)
def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


