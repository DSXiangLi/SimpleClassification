# -*-coding:utf-8 -*-
"""
    常用Train OP 组合
"""
import tensorflow.compat.v1 as tf
from itertools import chain

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


def train_op_diff_lr(loss, init_lr, diff_lr_times, optimizer, tvars=None):
    """
    For finetune, use  different learning rate schema for different layer
    diff_lr_times: {'scope': times}
    """

    global_step = tf.train.get_or_create_global_step()

    if not tvars:
        tvars = tf.trainable_variables()

    opt_list = []
    var_list = []
    # lr/opt for other layers
    for name, times in diff_lr_times.items():
        opt = optimizer(init_lr * times)
        opt_list.append(opt)
        var_list.append([i for i in tvars if name in i.name])

    # 如果有剩余没有被diff_lr_times覆盖的variable默认都走init lr
    vars = [i for i in tvars if i not in list(chain(*var_list))]
    if vars:
        opt = optimizer(init_lr)
        opt_list.append(opt)
        var_list.append(vars)

    # calculate gradient for all vars and clip gradient
    all_grads = tf.gradients(loss, list(chain(*var_list)))
    (all_grads, _) = tf.clip_by_global_norm(all_grads, clip_norm=1.0)

    # back propagate given different learning rate
    train_op_list = []
    for vars, opt in zip(var_list, opt_list):
        num_vars = len(vars)
        grads = all_grads[:num_vars]
        all_grads = all_grads[num_vars:]
        train_op = opt.apply_gradients(zip(grads, vars), global_step=global_step)
        train_op_list.append(train_op)
    train_op = tf.group(train_op_list, [global_step.assign(global_step + 1)])

    return train_op


