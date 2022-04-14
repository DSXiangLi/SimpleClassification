# -*-coding:utf-8 -*-
import os
import tensorflow as tf
import importlib
from tools.train_utils import HpParser, add_layer_summary
from model.train_helper import  build_model_fn

hp_list = [HpParser.hp('alpha', 0.1)]
hp_parser = HpParser(hp_list)


class MixupWrapper(object):
    def __init__(self, encoder):
        self.encoder = encoder

    @staticmethod
    def mixup(input_x, input_y, alpha):
        """
        Input
            input_x: batch_size * emb_size
            input_y: batch_size * label_size
        Return:
            same shape as above
        """
        # get mixup lambda
        batch_size = tf.shape(input_x)[0]
        mix = tf.distributions.Beta(alpha, alpha).sample(1)
        mix = tf.maximum(mix, 1 - mix)

        # get random shuffle sample
        index = tf.random_shuffle(tf.range(batch_size))
        random_x = tf.gather(input_x, index)
        random_y = tf.gather(input_y, index)

        # get mixed input
        xmix = input_x * mix + random_x * (1 - mix)
        ymix = input_y * mix + random_y * (1 - mix)
        return xmix, ymix

    def __call__(self, features, labels, params, is_training):
        self.params = params
        self.encoder.params = params
        embedding = self.encoder.encode(features, is_training)

        with tf.variable_scope('mixup'):
            if is_training:
                xmix, ymix = self.mixup(embedding, labels, self.params['alpha'])
            else:
                # keep testing sample unchanged
                xmix, ymix = embedding, labels

        with tf.variable_scope('mlp'):
            preds = tf.layers.dense(xmix, units=self.params['label_size'], activation=None, use_bias=True)
            add_layer_summary('preds', preds)

        return preds, ymix

    def optimize(self, loss):
        train_op = self.encoder.optimize(loss)
        return train_op


def get_trainer(model):
    module = importlib.import_module('model.{}.model'.format(model))
    encoder = getattr(module, '{}Encoder'.format(model.captialize()))
    dataset = getattr(module, 'dataset')
    Trainer = getattr(module, 'Trainer')

    trainer = Trainer(model_fn=build_model_fn(MixupWrapper(encoder)),
                      dataset_cls=dataset)
    return trainer