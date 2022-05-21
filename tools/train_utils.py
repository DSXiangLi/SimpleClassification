# -*-coding:utf-8 -*-
import os
import re
import shutil
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from collections import namedtuple, OrderedDict
from tools.logger import logger
from typing import List
from glob import glob

RUN_CONFIG = {
    'keep_checkpoint_max': 3,
    'allow_growth': True,
    'pre_process_gpu_fraction': 0.8,
    'log_device_placement': True,
    'allow_soft_placement': True,
    'inter_op_parallel': 2,
    'intra_op_parallel': 2,

}


def get_variables(scopes=(), suffix=None, trainable_only=True):
    """
    Get trainable or global variables, given scopes and suffix
    """
    if trainable_only:
        collection = tf.GraphKeys.TRAINABLE_VARIABLES
    else:
        collection = tf.GraphKeys.GLOBAL_VARIABLES

    if not scopes:
        tvars = tf.get_collection(collection)
        return tvars

    tvars = []
    for scope in scopes:
        try:
            scope = scope.name
        except:
            pass
        if suffix is not None:
            if ':' not in suffix:
                suffix += ':'
            scope = (scope or '') + '.*' + suffix
        tvar = tf.get_collection(collection, scope)
        tvars.extend(tvar)

    return tvars


def get_assignment_from_ckpt(ckpt, load_vars, verbose=False):
    """
    取checkpoint中变量和待家在变量的交集
    """
    init_vars = tf.train.list_variables(ckpt)
    assignment_map = OrderedDict()

    # 移除 surfix
    name_to_variable = OrderedDict()
    variables = set()
    for var in load_vars:
        name = var.name
        m = re.match('^(.*):\\d+$', name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
        variables.add(name)

    for name, var in init_vars:
        if name in name_to_variable:
            assignment_map[name] = name # like Bert
        else:
            logger.info('Variable {} not in vars to restore'.format(name))
    if verbose:
        logger.info('Vars in ckpt: {}'.format(init_vars))
        logger.info('Vars to load: {}'.format(name_to_variable))

    return assignment_map


def clear_model(model_dir):
    try:
        shutil.rmtree(model_dir)
    except Exception as e:
        print('Error! {} occured at model cleaning'.format(e))
    else:
        print( '{} model cleaned'.format(model_dir) )


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


def get_log_hook(loss, save_steps):
    hook = {}
    hook['global_step'] = tf.train.get_or_create_global_step()
    hook['loss'] = loss
    log_hook = tf.train.LoggingTensorHook(hook, every_n_iter=save_steps)
    return log_hook


def add_layer_summary(tag, value):
    tf.summary.scalar('{}/fraction_of_zero_values'.format(tag.replace(':', '_')), tf.math.zero_fraction(value))
    tf.summary.histogram('{}/activation'.format(tag.replace(':', '_')), value)


def build_estimator(model_fn, train_params, run_config):
    session_config = tf.ConfigProto()

    if run_config['use_gpu']:
        # control CPU and Mem usage
        session_config.gpu_options.allow_growth = run_config['allow_growth']
        session_config.gpu_options.per_process_gpu_memory_fraction = run_config['pre_process_gpu_fraction']
        session_config.log_device_placement = run_config['log_device_placement']
        session_config.allow_soft_placement = run_config['allow_soft_placement']
        session_config.inter_op_parallelism_threads = run_config['inter_op_parallel']
        session_config.intra_op_parallelism_threads = run_config['intra_op_parallel']

    run_config = tf.estimator.RunConfig(
        save_summary_steps=run_config['summary_steps'],
        log_step_count_steps=run_config['log_steps'],
        keep_checkpoint_max=run_config['keep_checkpoint_max'],
        save_checkpoints_steps=run_config['save_steps'],
        session_config=session_config, eval_distribute=None
    )

    # 如果对应checkpoint目录有ckpt文件默认进行warm start
    if glob(os.path.join(train_params['ckpt_dir'], '*ckpt*')):
        warm_start_dir = train_params['ckpt_dir']
    else:
        warm_start_dir = None

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=train_params,
        model_dir=train_params['ckpt_dir'],
        warm_start_from=warm_start_dir
    )
    return estimator