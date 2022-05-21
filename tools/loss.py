# -*-coding:utf-8 -*-
"""
    Loss Family for Classification
    Noisy Label Loss
        GCE: generalized cross entropy
        SCE: Symmetric cross entropy
        BoootCE: bootstrap
    Data Imbalance Loss

"""

import inspect
import tensorflow.compat.v1 as tf
from tools.train_utils import HpParser, add_layer_summary
from functools import wraps
LossHP = {}
LossFunc = {}


def add_loss_hp(func):
    """
    加入Loss Function相关hyper parameter
    """
    global LossFunc, LossHP
    hp_list = []
    sg = inspect.signature(func)  # loss default params
    for k, v in sg.parameters.items():
        hp_list.append(HpParser.hp(k, v.default))
    LossHP[func.__name__] = HpParser(hp_list)
    LossFunc[func.__name__] = func


def pre_loss(loss_func):
    @wraps(loss_func)
    def helper(logits, labels, num_labels=None):
        probs = tf.nn.softmax(logits, axis=-1)

        if not num_labels:
            num_labels = logits.get_shape().as_list()[-1]

        if len(labels.get_shape().as_list()) == 1:
            labels = tf.one_hot(labels, depth=num_labels)

        loss = loss_func(probs, labels)
        return loss

    return helper


@add_loss_hp
def ce():
    """
    Cross Entropy, all below loss function adapts to it
    """

    @pre_loss
    def helper(probs, labels):
        eps = 1e-10
        ce = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(probs, eps, 1.0 - eps)), axis=-1)
        return ce

    return helper


@add_loss_hp
def gce(q=0.7):
    @pre_loss
    def helper(probs, labels):
        # (1-f(x)^q)/q
        loss = 1 - tf.pow(tf.reduce_sum(labels * probs, axis=-1), q) / q
        return loss

    return helper


@add_loss_hp
def sce(alpha=0.1, beta=1):
    @pre_loss
    def helper(probs, labels):
        # KL(p|q) + KL(q|p)
        eps = 1e-10
        y_true = tf.clip_by_value(labels, eps, 1.0 - eps)
        y_pred = probs
        ce = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        add_layer_summary('ce', ce)

        y_true = probs
        y_pred = tf.clip_by_value(labels, eps, 1.0 - eps)
        rce = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        add_layer_summary('rce', rce)

        return alpha * ce + beta * rce

    return helper


@add_loss_hp
def bootce(beta=0.95, is_hard=0):
    def helper(logits, labels, num_labels=None):
        # (y+p) * log(p)
        if not num_labels:
            num_labels = logits.get_shape().as_list()[-1]

        if len(labels.get_shape().as_list()) == 1:
            labels = tf.one_hot(labels, depth=num_labels)
        eps = 1e-10
        probs = tf.nn.softmax(logits, axis=-1)
        probs = tf.clip_by_value(probs, eps, 1 - eps)
        if is_hard:
            pred_label = tf.one_hot(tf.argmax(probs, axis=-1), depth=num_labels)
        else:
            pred_label = probs

        loss = -tf.reduce_sum((beta * labels + (1.0 - beta) * pred_label) * tf.log(probs), axis=-1)
        return loss

    return helper


@add_loss_hp
def peerce(alpha=0.5):
    @pre_loss
    def helper(probs, labels):
        # transformer labels to one-hot
        rand_labels = tf.random.shuffle(labels)
        ce_true = ce()(probs, labels)
        ce_rand = ce()(probs, rand_labels)
        loss = alpha * ce_true + (1 - alpha) * ce_rand
        return loss

    return helper


@add_loss_hp
def focal(gamma=2, alpha=0.24):
    @pre_loss
    def helper(probs, labels):
        focus = -tf.reduce_sum(labels * tf.log(probs) * tf.pow(1 - probs, gamma), axis=-1)
        imbalance = tf.reduce_sum(labels * tf.constant([alpha, 1 - alpha]), axis=-1)
        loss = focus * imbalance
        return loss

    return helper
