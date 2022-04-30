# -*-coding:utf-8 -*-

import importlib
import tensorflow as tf
from model.train_helper import build_model_fn, Trainer, BaseEncoder
from tools.train_utils import HpParser, add_layer_summary
from functools import partial
from dataset.base_dataset import BaseDataset

hp_list = [HpParser.hp('distill_weight', 0.5),
           HpParser.hp('distill_loss', 'ce'),
           HpParser.hp('temperature', 2),
           ]
hp_parser = HpParser(hp_list)


def temperature_softmax(t_logit, s_logit, temperature):
    # t is teacher logit , s is student logit
    t_prob = tf.nn.softmax(t_logit / temperature)
    s_prob = tf.nn.softmax(s_logit / temperature)
    ce = -tf.reduce_sum(t_prob * tf.log(s_prob), axis=-1)
    return ce


def mse(t_logit, s_logit):
    return tf.reduce_sum((t_logit - s_logit) ** 2, axis=-1)


class KnowledgeDistillWrapper(BaseEncoder):
    def __init__(self, encoder):
        super(KnowledgeDistillWrapper, self).__init__()
        self.encoder = encoder
        self.alpha = None
        self.teacher_logit = None  # prediction from teacher model
        self.distill_loss = None

    def get_loss(self):
        if self.params['distill_loss'] == 'ce':
            return partial(temperature_softmax, temperature=self.params['temperature'])
        elif self.params['distill_loss'] == 'mse':
            return mse

    def encode(self, features, is_training):
        return self.encoder.encode(features, is_training)

    def __call__(self, features, labels, params, is_training):
        if labels:
            # 分离teacher soft label和任务true label
            self.teacher_logit = labels['logit']
            labels = labels['label']
        self.params = params
        self.encoder.params = params
        self.weight = params['distill_weight']
        self.distill_loss = self.get_loss()

        embedding = self.encode(features, is_training)
        with tf.variable_scope('transfer'):
            preds = tf.layers.dense(embedding, units=self.params['label_size'], activation=None, use_bias=True)
            add_layer_summary('preds', preds)
        return preds, labels

    def init_fn(self):
        return self.encoder.init_fn()

    def optimize(self, loss):
        train_op = self.encoder.optimize(loss)
        return train_op

    def compute_loss(self, predictions, labels):
        # supervised loss
        supervised_loss = tf.reduce_mean(self.params['loss_func'](predictions, labels))
        tf.summary.scalar('loss/supervised_loss', supervised_loss)

        # distill loss: mse between prediction and
        distill_loss = tf.reduce_mean(self.distill_loss(self.teacher_logit, predictions))
        tf.summary.scalar('loss/mse_loss', distill_loss)

        # loss 加权
        total_loss = self.weight * distill_loss + supervised_loss
        return total_loss


def get_distill_dataset(dataset_cls: BaseDataset):
    class DistillDataset(dataset_cls):
        def __init__(self, *args, **kwargs):
            super(DistillDataset, self).__init__(*args, **kwargs)

        def build_proto(self):
            super().build_proto()
            self.dtypes.update({
                'logit': tf.float32
            })
            self.shapes.update({
                'logit': [None]
            })
            self.pads.update({
                'logit': 0.0
            })
            self.label_names.append('logit')

        def build_single_feature(self, data):
            sample = super().build_single_feature(data)
            sample.update({
                'logit': data['logit']
            })
            return sample

    return DistillDataset


def get_trainer(model):
    module = importlib.import_module('model.{}.model'.format(model))
    encoder = getattr(module, '{}Encoder'.format(model.capitalize()))
    dataset = getattr(module, 'dataset')

    trainer = Trainer(model_fn=build_model_fn(KnowledgeDistillWrapper(encoder())),
                      dataset_cls=get_distill_dataset(dataset))
    return trainer
