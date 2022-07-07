# -*-coding:utf-8 -*-
"""
    MMOE Share Bottom
"""
import tensorflow.compat.v1 as tf
import importlib
from tools.train_utils import HpParser, add_layer_summary
from model.train_helper import build_mtl_model_fn, BaseEncoder, MultitaskTrainer
from dataset.multi_dataset import Mtl2SeqDataset


hp_list = [HpParser.hp('share_size', 200),
           HpParser.hp('share_dropout', 0.3),
           HpParser.hp('share_activation', 'relu'),
           HpParser.hp('task_weight', '0.5,0.5',
                       lambda x: dict([(i, float(j)) for i, j in enumerate(x.split(','))])),  # 各个任务的loss权重
           HpParser.hp('task_label_size', '20,61',
                       lambda x: dict([(i, int(j)) for i, j in enumerate(x.split(','))])),  # 各个任务的label size
           ]
hp_parser = HpParser(hp_list)


class MultitaskWrapper(BaseEncoder):  # noqa
    def __init__(self, encoder):
        super(MultitaskWrapper, self).__init__()
        self.encoder = encoder

    def encode(self, features, is_training):
        return self.encoder.encode(features, is_training)

    def __call__(self, features, labels, params, is_training):
        self.params = params
        self.encoder.params = params
        embedding = self.encode(features, is_training)

        # 两个任务独立预测
        with tf.variable_scope('independent_pred'):
            predictions2 = tf.layers.dense(embedding,
                                           units=self.params['task_label_size'][1], activation=None, use_bias=True)
            predictions1 = tf.layers.dense(predictions2,
                                           units=self.params['task_label_size'][0], activation=None, use_bias=True)

            add_layer_summary('prediction1', predictions1)
            add_layer_summary('prediction2', predictions2)

        return [predictions1, predictions2], labels

    def compute_loss(self, predictions, labels):
        """
        各任务loss加权平均，这里暂不支持不同
        """
        loss_func = self.params['loss_func']
        total_loss = 0
        # independent loss
        for id, weight in self.params['task_weight'].items():
            loss = tf.reduce_mean(loss_func(predictions[id], labels[id]))
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

    trainer = MultitaskTrainer(model_fn=build_mtl_model_fn(MultitaskWrapper(encoder())),
                       dataset_cls=Mtl2SeqDataset)
    return trainer
