# -*-coding:utf-8 -*-

import json
import importlib
import tensorflow as tf
import tf_slim as slim

from tools.train_utils import add_layer_summary, get_variables, get_assignment_from_ckpt, HpParser
from tools.opt_utils import train_op_diff_lr
from model.train_helper import Trainer, build_model_fn, BaseEncoder
from preprocess.slim.preprocessing_factory import get_preprocessing
from dataset.img_dataset import ImgDataset

hp_list = [HpParser.hp('img_feature_layer', 'global_pool'),
           HpParser.hp('img_pretrain_model', 'resnet_v1_50'),
           HpParser.hp('img_size', 224),
           HpParser.hp('scope_lr', '{"transfer":50, "resnet_v1_50/block4":1, "resnet_v1_50/block3":1}',
                       lambda x: json.loads(x))]
hp_parser = HpParser(hp_list)


class SlimEncoder(BaseEncoder):
    """
    Support Global Image Transfer/Fintune for Classification.
    Base Model: Vgg, Resnet, Inception
    """

    def __init__(self):
        super(SlimEncoder, self).__init__()
        self.model_name = None
        self.module = None
        self.encoder = None
        self.arg_scope = None

    def _get_module(self, model_name):
        if 'resnet_v1' in model_name:
            module = importlib.import_module('backbone.slim.resnet_v1')
        elif 'resnet_v2' in self.model_name:
            module = importlib.import_module('backbone.slim.resnet_v2')
        elif 'vgg' in self.model_name:
            module = importlib.import_module('backbone.slim.vgg')
        else:
            raise ValueError('only resnet_v1, vgg are supported')
        return module

    def _get_argscope(self, model_name):
        if 'resnet' in model_name:
            arg_scope = getattr(self.module, 'resnet_arg_scope')
        elif 'vgg' in model_name:
            arg_scope = getattr(self.module, 'vgg_arg_scope')
        else:
            raise ValueError('only resnet%, vgg% are supported')
        return arg_scope

    def encode(self, features, is_training):
        self.module = self._get_module(self.model_name)
        self.arg_scope = self._get_argscope(self.model_name)
        self.encoder = getattr(self.module, self.model_name)

        with slim.arg_scope(self.arg_scope()):
            net, end_points = self.encoder(features['img_array'], num_classes=None, is_training=is_training)

        img_embedding = end_points[self.params['img_feature_layer']]
        img_embedding = tf.reduce_mean(img_embedding, axis=[1, 2], keep_dims=False)
        add_layer_summary('img_embedding', img_embedding)
        return img_embedding

    def __call__(self, features, labels, params, is_training):
        """
        pretrain Bert model output + pretrain Image Encoder
        """
        self.params = params
        self.model_name = params['img_pretrain_model']

        img_embedding = self.encode(features, is_training)

        with tf.variable_scope('transfer'):
            logits = tf.layers.dense(img_embedding, units=params['label_size'], activation=None,
                                     use_bias=True, name='logits')
            add_layer_summary('logits', logits)

        return logits, labels

    def init_fn(self):
        # for nets with BatchNorm, load all vars instead of train vars
        all_vars = get_variables(trainable_only=False)
        assignment_map = get_assignment_from_ckpt(self.params['img_pretrain_ckpt'], all_vars)
        tf.train.init_from_checkpoint(self.params['img_pretrain_ckpt'], assignment_map)
        return None

    def optimize(self, loss):
        # UPDATE_OPS is necessary for model with Batch Norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        tvars = get_variables(self.params['scope_lr'].keys(), trainable_only=True)
        with tf.control_dependencies(update_ops):
            train_op = train_op_diff_lr(loss, self.params['lr'], self.params['scope_lr'],
                                        tf.train.AdamOptimizer, tvars=tvars)

        return train_op


class ImgTrainer(Trainer):
    def __init__(self, model_fn, dataset_cls):
        super(ImgTrainer, self).__init__(model_fn, dataset_cls)

    def prepare(self):
        self.logger.info('Prepare dataset')
        self.input_pipe = self.dataset_cls(data_dir=self.train_params['data_dir'],
                                           batch_size=self.train_params['batch_size'],
                                           img_size=self.train_params['img_size'],
                                           img_preprocess=get_preprocessing(self.train_params['img_pretrain_model'])
                                           )
        self.input_pipe.build_feature('train')
        self.train_params = self.input_pipe.update_params(self.train_params)


trainer = ImgTrainer(model_fn=build_model_fn(SlimEncoder),
                     dataset_cls=ImgDataset)
