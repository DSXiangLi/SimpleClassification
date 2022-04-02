# -*-coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops
from collections import namedtuple
from typing import List


class HpParser(object):
    """
    Model and Loss Specific hyperparameter parser, will be added to the default arguemnt
    """
    Hp = namedtuple('AddonHP', ['field', 'default', 'action'])

    @staticmethod
    def hp(field, default, action=lambda x: x):
        # add default field value, python>3.8 namedtuple will have default ops
        return HpParser.Hp(field, default, action)

    def __init__(self, hp_list: List[namedtuple]):
        self.hp_list = hp_list

    def append(self, parser):
        for i in self.hp_list:
            parser.add_argument("--" + i.field, default=i.default, type=type(i.default))
        return parser

    def update(self, params, args):
        args = vars(args)
        for i in self.hp_list:
            params[i.field] = i.action(args[i.field])
        return params

    def parse(self, args):
        params = {}
        args = vars(args)
        for i in self.hp_list:
            params[i.field] = i.action(args[i.field])
        return params


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
        self.num_calls += 1

        return y

def get_log_hook(loss, save_steps):
    hook = {}
    hook['global_step'] = tf.train.get_or_create_global_step()
    hook['loss'] = loss
    log_hook = tf.train.LoggingTensorHook(hook, every_n_iter=save_steps)
    return log_hook


def add_layer_summary(tag, value):
    tf.summary.scalar('{}/fraction_of_zero_values'.format(tag.replace(':', '_')), tf.math.zero_fraction(value))
    tf.summary.histogram('{}/activation'.format(tag.replace(':', '_')), value)

