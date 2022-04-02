# -*-coding:utf-8 -*-
"""
    Dataset for multiple source input
"""
import tensorflow as tf


class MultiDataset(object):
    def __init__(self,  batch_size, max_seq_len, data_dir_list, dataset_cls,tokenizer, use_cache):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.data_dir_list = data_dir_list
        self.datasets = dict([(data_dir, dataset_cls(data_dir, batch_size, max_seq_len, tokenizer, use_cache))
                              for data_dir in data_dir_list])
        self.shapes = {}
        self.pads = {}
        self.dtypes = {}
        self.feature_names = []
        self.label_names = []

    @property
    def task_size(self):
        return int(len(self.data_dir_list))

    @property
    def sample_size(self):
        """
        MultiSource多个source样本之和
        """
        return int(sum([d.sample_size for d in self.datasets.values()]))

    @property
    def steps_per_epochs(self):
        return int(self.sample_size/self.batch_size)

    @property
    def task2idx(self):
        """
        数据源按传入顺序依次赋值0，1，2
        """
        return dict([(task, idx) for idx, task in enumerate(self.data_dir_list)])

    def add_task_id(self, feature, label, task_name):
        feature['task_ids'] = self.task2idx[task_name]
        return feature, label

    def build_proto(self):
        """
        因为两个任务的输入相同，所以直接继承其中一个的proto，然后加入task即可
        """
        self.shapes = self.datasets[self.data_dir_list[0]].shapes
        self.dtypes = self.datasets[self.data_dir_list[0]].dtypes
        self.pads = self.datasets[self.data_dir_list[0]].pads

        self.shapes.update({'task_ids': []})
        self.dtypes.update({'task_ids': tf.int32})
        self.pads.update({'task_ids': 0})

        self.feature_names = self.datasets[self.data_dir_list[0]].feature_names + ['task_ids']
        self.label_names = self.datasets[self.data_dir_list[0]].label_names

    def build_feature(self, file_name):
        for ds in self.datasets.values():
            ds.build_feature(file_name)

    def build_serving_proto(self):
        receiver_tensor = {}
        for i in self.feature_names:
            receiver_tensor[i] = tf.placeholder(dtype=self.dtypes[i],
                                                shape=[None]+self.shapes[i],
                                                name=i)
        return tf.estimator.export.ServingInputReceiver(receiver_tensor, receiver_tensor)

    def build_input_fn(self, is_predict=False):
        def helper():
            shapes = ({i: self.shapes[i] for i in self.feature_names},
                      {i: self.shapes[i] for i in self.label_names})
            pads = ({i: self.pads[i] for i in self.feature_names},
                      {i: self.pads[i] for i in self.label_names})

            dataset_list = [ds.build_input_fn(is_predict, unbatch=True)().\
                                map(lambda feature, label: self.add_task_id(task))
                            for task, ds in self.datasets.items()]
            choice = tf.data.Dataset.range(self.task_size).repeat()

            dataset = tf.contrib.data.choose_from_datasets(dataset_list, choice)

            if not is_predict:
                dataset = dataset.shuffle(int(self.batch_size * 5)).repeat()

            dataset = dataset.padded_batch(self.batch_size, shapes, pads).\
                prefetch(tf.data.experimental.AUTOTUNE)

            return dataset
        return helper