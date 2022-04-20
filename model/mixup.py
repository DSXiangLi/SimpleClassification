# -*-coding:utf-8 -*-
import os
import tensorflow as tf
import importlib
from tools.train_utils import HpParser, add_layer_summary
from model.train_helper import build_model_fn, BaseEncoder

hp_list = [HpParser.hp('mixup_alpha', 0.1)]
hp_parser = HpParser(hp_list)


def mixup(input_x, input_y, label_size, alpha):
    """
    Input
        input_x: batch_size * emb_size
        input_y: batch_size * label_size
    Return:
        same shape as above
    """
    # get mixup lambda
    batch_size = tf.shape(input_x)[0]
    input_y = tf.one_hot(input_y, depth=label_size)

    mix = tf.distributions.Beta(alpha, alpha).sample(1)
    mix = tf.maximum(mix, 1 - mix)

    # get random shuffle sample
    index = tf.random_shuffle(tf.range(batch_size))
    random_x = tf.gather(input_x, index)
    random_y = tf.gather(input_y, index)

    # get mixed input
    xmix = input_x * mix + random_x * (1 - mix)
    ymix = tf.cast(input_y, tf.float32) * mix + tf.cast(random_y, tf.float32) * (1 - mix)
    return xmix, ymix


class MixupWrapper(BaseEncoder):
    def __init__(self, encoder):
        super(MixupWrapper, self).__init__()
        self.encoder = encoder

    def encode(self, features, is_training):
        return self.encoder.encode(features, is_training)

    def __call__(self, features, labels, params, is_training):
        self.params = params
        self.encoder.params = params
        embedding = self.encode(features, is_training)

        with tf.variable_scope('mixup'):
            if is_training:
                xmix, ymix = mixup(embedding, labels, self.params['label_size'], self.params['mixup_alpha'])
            else:
                # keep testing sample unchanged
                xmix, ymix = embedding, labels

        with tf.variable_scope('mlp'):
            preds = tf.layers.dense(xmix, units=self.params['label_size'], activation=None, use_bias=True)
            add_layer_summary('preds', preds)

        return preds, ymix

    def init_fn(self):
        return self.encoder.init_fn()

    def optimize(self, loss):
        train_op = self.encoder.optimize(loss)
        return train_op


def get_trainer(model):
    module = importlib.import_module('model.{}.model'.format(model))
    encoder = getattr(module, '{}Encoder'.format(model.capitalize()))
    dataset = getattr(module, 'dataset')
    Trainer = getattr(module, 'Trainer')

    trainer = Trainer(model_fn=build_model_fn(MixupWrapper(encoder())),
                      dataset_cls=dataset)
    return trainer