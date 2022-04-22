# -*-coding:utf-8 -*-

import importlib

import numpy as np
import tensorflow as tf

from dataset.tokenizer import get_tokenizer
from model.train_helper import build_model_fn, BaseTrainer, BaseEncoder
from tools.train_utils import HpParser, add_layer_summary

hp_list = [HpParser.hp('temporal_alpha', 0.6), #  ensemble alpha
           HpParser.hp('max_unsupervised', 1.0), #  w_max, when there is no unlabled data set it to 1
           HpParser.hp('ramp_up_method', 'sigmoid') #  ramp up decay type
           ]
hp_parser = HpParser(hp_list)


def ramp_up(cur_epoch, max_epoch, method):
    """
    根据训练epoch来调整无标注loss部分的权重，初始epoch无标注loss权重为0
    """
    def linear(cur_epoch, max_epoch):
        return cur_epoch / max_epoch

    def sigmoid(cur_epoch, max_epoch):
        p = 1.0 - cur_epoch / max_epoch
        return tf.exp(-5.0 * p ** 2)

    def cosine(cur_epoch, max_epoch):
        p = cur_epoch / max_epoch
        return 0.5 * (tf.cos(np.pi * p) + 1)

    if cur_epoch == 0:
        weight = tf.constant(0.0)
    else:
        if method == 'linear':
            weight = linear(cur_epoch, max_epoch)
        elif method == 'sigmoid':
            weight = sigmoid(cur_epoch, max_epoch)
        elif method == 'cosine':
            weight = cosine(cur_epoch, max_epoch)
        else:
            raise ValueError('Only linear, sigmoid, cosine method are supported')
    return tf.cast(weight, tf.float32)


class TemporalWrapper(BaseEncoder):
    def __init__(self, encoder):
        super(TemporalWrapper, self).__init__()
        self.encoder = encoder
        self.Z = None
        self.assign_op = None
        self.alpha = None  # temporal ensemble momentum
        self.wmax = None  # max weight unsupervised loss

    def encode(self, features, is_training):
        return self.encoder.encode(features, is_training)

    def __call__(self, features, labels, params, is_training):
        self.params = params
        self.encoder.params = params
        self.alpha = self.params['temporal_alpha']
        self.wmax = tf.cast(self.params['max_unsupervised'] * self.params['labeled_size'] / self.params['sample_size'], tf.float32)

        embedding = self.encode(features, is_training)

        with tf.variable_scope('mlp'):
            preds = tf.layers.dense(embedding, units=self.params['label_size'], activation=None, use_bias=True)
            add_layer_summary('preds', preds)

        with tf.variable_scope('temporal_ensemble'):
            temporal_ensemble = tf.get_variable(initializer=tf.zeros_initializer,
                                                shape=(self.params['sample_size'], self.params['label_size']),
                                                dtype=tf.float32, name='temporal_ensemble', trainable=False)
            add_layer_summary('temporal_ensemble', temporal_ensemble)

            self.Z = tf.nn.embedding_lookup(temporal_ensemble, features['idx'])  # batch_size * label_size
            self.Z = self.alpha * self.Z + (1-self.alpha) * preds
            self.assign_op = tf.scatter_update(temporal_ensemble, features['idx'], self.Z)
            add_layer_summary('ensemble', self.Z)
        return preds, labels

    def init_fn(self):
        return self.encoder.init_fn()

    def optimize(self, loss):
        train_op = self.encoder.optimize(loss)
        train_op = tf.group(train_op, self.assign_op)  # add temporal ensemble update op
        return train_op

    def compute_loss(self, predictions, labels):
        # supervised loss
        mask = tf.cast(tf.where(labels>=0, tf.ones_like(labels), tf.zeros_like(labels)), tf.float32) # select labeled sample
        loss = self.params['loss_func'](predictions, labels)
        supervised_loss = tf.reduce_mean(loss * mask)
        add_layer_summary('supervised_mask', mask)
        tf.summary.scalar('loss/supervised_loss', supervised_loss)

        # MSE between current prediction and temporal ensemble
        cur_epoch = tf.cast(tf.train.get_or_create_global_step()/self.params['steps_per_epoch'], tf.int32)
        temporal_logits = self.Z/(1-tf.pow(self.alpha, tf.cast(cur_epoch+1, tf.float32)))  # startup bias
        mse_loss = tf.reduce_mean((tf.nn.softmax(predictions) - tf.nn.softmax(temporal_logits))**2)
        tf.summary.scalar('loss/mse_loss', mse_loss)

        # Loss加权
        weight = ramp_up(cur_epoch, self.params['epoch_size'], self.params['ramp_up_method']) * self.wmax
        tf.summary.scalar('loss_weight', weight)
        total_loss = supervised_loss + mse_loss * weight
        return total_loss


class Trainer(BaseTrainer):
    def __init__(self, model_fn, dataset_cls):
        super(Trainer, self).__init__(model_fn, dataset_cls)

    def prepare(self):
        self.logger.info('Prepare dataset')
        self.input_pipe = self.dataset_cls(data_dir=self.train_params['data_dir'],
                                           batch_size=self.train_params['batch_size'],
                                           max_seq_len=self.train_params['max_seq_len'],
                                           tokenizer=get_tokenizer(self.train_params['nlp_pretrain_model']),
                                           enable_cache=self.train_params['enable_cache'],
                                           clear_cache=self.train_params['clear_cache'])
        self.input_pipe.build_feature('train')
        self.train_params = self.input_pipe.update_params(self.train_params)



def get_trainer(model):
    module = importlib.import_module('model.{}.model'.format(model))
    encoder = getattr(module, '{}Encoder'.format(model.capitalize()))
    dataset = getattr(module, 'dataset')

    trainer = Trainer(model_fn=build_model_fn(TemporalWrapper(encoder())),
                      dataset_cls=dataset)
    return trainer