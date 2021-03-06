# -*-coding:utf-8 -*-
"""
    Ref1. Multi-task Domain Adaptation for Sequence Tagging
    Ref2. A survey of multi-source domain adaptation
"""
import tensorflow.compat.v1 as tf
from functools import partial
import importlib
from tools.train_utils import HpParser, add_layer_summary
from model.train_helper import build_model_fn, BaseEncoder, MultiTrainer
from dataset.multi_dataset import MultiDataset

hp_list = [HpParser.hp('share_size', 200),
           HpParser.hp('share_dropout', 0.3),
           HpParser.hp('share_activation', 'relu'),
           HpParser.hp('task_weight', '0.5,0.5',
                       lambda x: dict([(i, float(j)) for i, j in enumerate(x.split(','))])),  # 各个任务的loss权重
           HpParser.hp('task_label_size', '2,2',
                       lambda x: dict([(i, int(j)) for i, j in enumerate(x.split(','))])),  # 各个任务的label size
           ]
hp_parser = HpParser(hp_list)


class MultisourceWrapper(BaseEncoder):  # noqa
    def __init__(self, encoder):
        super(MultisourceWrapper, self).__init__()
        self.encoder = encoder
        self.task_ids = None

    def encode(self, features, is_training):
        return self.encoder.encode(features, is_training)

    def domain_layer(self, share_private, embedding, task_id):
        with tf.variable_scope('task'.format(task_id)):
            preds = tf.layers.dense(tf.concat([share_private, embedding], axis=-1),
                                    units=self.params['task_label_size'][task_id], activation=None, use_bias=True)
            add_layer_summary('task_pred_{}'.format(task_id), preds)
        return preds

    def __call__(self, features, labels, params, is_training):
        self.params = params
        self.encoder.params = params
        self.task_ids = features['task_ids']
        embedding = self.encode(features, is_training)

        with tf.variable_scope('share_embedding'):
            share_private = tf.layers.dense(embedding, units=params['share_size'],
                                            activation=params['share_activation'])

            share_private = tf.layers.dropout(share_private, rate=params['share_dropout'], seed=1234,
                                              training=is_training)

        # 一个domain一个head，预测结果只取task对应的head预测'
        predictions = []
        for id in range(self.params['task_size']):
            predictions.append(self.domain_layer(share_private, embedding, id))
        task_index = tf.stack([self.task_ids, tf.range(tf.shape(labels)[0])], axis=1)
        predictions = tf.gather_nd(predictions, task_index)  # batch_size * label_size

        return predictions, labels

    def compute_loss(self, predictions, labels):
        """
        各任务loss加权平均，这里暂不支持不同
        """
        loss_func = self.params['loss_func']
        loss = loss_func(predictions, labels)
        total_loss = 0
        for id, weight in self.params['task_weight'].items():
            task_mask = tf.cast(tf.equal(self.task_ids, id), tf.float32)
            loss = tf.reduce_sum(tf.multiply(task_mask, loss)) / (tf.reduce_sum(task_mask) + 1e-9)
            tf.summary.scalar('task_loss/loss_{}'.format(id), loss)
            total_loss += loss * weight
        return total_loss

    def init_fn(self):
        return self.encoder.init_fn()

    def optimize(self, loss):
        train_op = self.encoder.optimize(loss)
        return train_op


def get_trainer(model):
    module = importlib.import_module('model.{}.model'.format(model))
    encoder = getattr(module, '{}Encoder'.format(model.capitalize()))
    dataset = getattr(module, 'dataset')

    trainer = MultiTrainer(model_fn=build_model_fn(MultisourceWrapper(encoder())),
                           dataset_cls=partial(MultiDataset, dataset_cls=dataset))
    return trainer
