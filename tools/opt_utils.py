# -*-coding:utf-8 -*-
"""
    常用Train OP 组合
"""
import tensorflow as tf


def lr_decay(init_lr, step_per_epoch, decay_rate):
    global_step = tf.train.get_or_create_global_step()

    lr = tf.train.exponential_decay(
        init_lr,
        global_step,
        step_per_epoch,
        staircase=True,
        decay_rate=decay_rate)

    tf.summary.scalar('lr', lr)
    return lr


def gradient_clipping(optimizer, cost, lower_clip, upper_clip):
    """
    apply gradient clipping
    """
    gradients, variables = zip(*optimizer.compute_gradients( cost ))

    clip_grad = [tf.clip_by_value( grad, lower_clip, upper_clip) for grad in gradients if grad is not None]

    train_op = optimizer.apply_gradients(zip(clip_grad, variables),
                                         global_step=tf.train.get_global_step() )

    return train_op


def train_op_clip_decay(loss, init_lr, steps_per_epoch, decay_rate, lower_clip, upper_clip):
    """
    Adam optimizer with exponential lr decay and gradient clip
    """
    lr = lr_decay(init_lr, steps_per_epoch, decay_rate)

    opt = tf.train.AdamOptimizer(lr)

    train_op = gradient_clipping(opt, loss, lower_clip, upper_clip)

    return train_op


