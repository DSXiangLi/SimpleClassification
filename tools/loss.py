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
import tensorflow as tf
from tools.train_utils import HpParser, add_layer_summary
LossHP = {}
LossFunc = {}


def add_loss_hp(func):
    """
    加入Loss Function相关hyper parameter
    """
    global LossFunc, LossHP
    hp_list = []
    sg = inspect.signature(func) # loss default params
    for k,v in sg.parameters.items():
        hp_list.append(HpParser.hp(k, v.default))
    LossHP[func.__name__] = HpParser(hp_list)
    LossFunc[func.__name__] = func


@add_loss_hp
def ce():
    """
    Cross Entropy, all below loss function adapts to it
    """
    def helper(logits, labels, num_labels=None):
        if not num_labels:
            num_labels = logits.get_shape().as_list()[-1]
        probs = tf.nn.softmax(logits, axis=-1)
        # transformer labels to one-hot
        if len(labels.get_shape().as_list()) == 1:
            labels = tf.one_hot(labels, depth=num_labels)
        ce = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(probs, 1e-10, 1.0)), axis=-1)
        return ce
    return helper


@add_loss_hp
def gce(q=0.7):
    def helper(logits, labels, num_labels=None):
        if not num_labels:
            num_labels = logits.get_shape().as_list()[-1]
        probs = tf.nn.softmax(logits, axis=-1)
        # transformer labels to one-hot
        if len(labels.get_shape().as_list()) == 1:
            labels = tf.one_hot(labels, depth=num_labels)
        # (1-f(x)^q)/q
        loss = 1- tf.pow(tf.reduce_sum(labels * probs, axis=-1),q)/q
        return loss
    return helper

@add_loss_hp
def sce(alpha=0.1, beta=1):
    def helper(logits, labels, num_labels=None):
        if not num_labels:
            num_labels = logits.get_shape().as_list()[-1]
        probs = tf.nn.softmax(logits, axis=-1)
        # transformer labels to one-hot
        if len(labels.get_shape().as_list()) == 1:
            labels = tf.one_hot(labels, depth=num_labels)
        # KL(p|q) + KL(q|p)
        y_true = tf.clip_by_value(labels, 1e-10, 1.0)
        y_pred = probs
        ce = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        add_layer_summary('ce', ce)

        y_true = probs
        y_pred = tf.clip_by_value(labels, 1e-10, 1.0)
        rce = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        add_layer_summary('rce', rce)

        return alpha * ce + beta * rce
    return helper

@add_loss_hp
def bootce(beta=0.95, is_hard=0):
    def helper(logits, labels, num_labels=None):
        if not num_labels:
            num_labels = logits.get_shape().as_list()[-1]
        probs = tf.nn.softmax(logits, axis=-1)
        # transformer labels to one-hot
        if len(labels.get_shape().as_list()) == 1:
            labels = tf.one_hot(labels, depth=num_labels)

        # (y+p) * log(p)
        eps = 1e-10
        probs = tf.clip_by_value(probs, eps, 1-eps)
        if is_hard:
            pred_label = tf.one_hot(tf.argmax(probs, axis=-1),depth=num_labels)
        else:
            pred_label = probs

        loss = -tf.reduce_sum( (beta*labels + (1.0-beta) * pred_label) * tf.log(probs), axis=-1)
        return loss
    return helper


@add_loss_hp
def peerce(alpha=0.5):
    def helper(logits, labels, num_labels=None):
        # transformer labels to one-hot
        if not num_labels:
            num_labels = logits.get_shape().as_list()[-1]

        if len(labels.get_shape().as_list()) == 1:
            labels = tf.one_hot(labels, depth=num_labels)
        rand_labels = tf.random.shuffle(labels)
        ce_true = ce()(labels, logits)
        ce_rand = ce()(rand_labels, logits)
        loss = alpha * ce_true + (1-alpha) * ce_rand
        return loss
    return helper


@add_loss_hp
def focal(gamma=2, alpha=0.24):
    def helper(logits, labels, num_labels=None):
        if not num_labels:
            num_labels = logits.get_shape().as_list()[-1]
        probs = tf.nn.softmax(logits, axis=-1)
        # transformer labels to one-hot
        if len(labels.get_shape().as_list()) == 1:
            labels = tf.one_hot(labels, depth=num_labels)

        focus = -tf.reduce_sum(labels * tf.log(probs) * tf.pow(1-probs, gamma), axis=-1)
        imbalance = tf.reduce_sum(labels * tf.constant([alpha, 1-alpha]), axis=-1)
        loss = focus * imbalance
        return loss
    return helper
