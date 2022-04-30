# -*-coding:utf-8 -*-
"""
    Ref1. Adversarial Transfer Learning for Chinese Named Entity Recognition with Self-Attention Mechanism
    Ref2. Adversarial Multi-task Learning for Text Classification
    Ref3. Dual Adversarial Neural Transfer for Low-Resource Named Entity Recognition
"""
import tensorflow as tf
from functools import partial
import importlib
from tensorflow.python.framework import ops
from tools.train_utils import HpParser, add_layer_summary
from model.train_helper import build_model_fn, BaseEncoder, MultiTrainer
from dataset.multi_dataset import MultiDataset

hp_list = [HpParser.hp('share_size', 200),
           HpParser.hp('share_dropout', 0.3),
           HpParser.hp('share_activation', 'relu'),
           HpParser.hp('task_lambda', 0.5),  # task discriminator weight
           HpParser.hp('shrink_gradient_reverse', 0.001),  # task discriminator weight
           HpParser.hp('task_weight', '0.5,0.5',
                       lambda x: dict([(i, float(j)) for i, j in enumerate(x.split(','))])),  # 各个任务的loss权重
           HpParser.hp('task_label_size', '2,2',
                       lambda x: dict([(i, int(j)) for i, j in enumerate(x.split(','))])),  # 各个任务的label size
           ]
hp_parser = HpParser(hp_list)


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


flip_gradient = FlipGradientBuilder()


class AdversarialWrapper(BaseEncoder):  # noqa
    def __init__(self, encoder):
        super(AdversarialWrapper, self).__init__()
        self.encoder = encoder
        self.task_ids = None  # batch_size
        self.task_logits = None  # batch_size * task_size

    def encode(self, features, is_training):
        return self.encoder.encode(features, is_training)

    def domain_layer(self, share_private, embedding, task_id):
        with tf.variable_scope('task'):
            preds = tf.layers.dense(tf.concat([share_private, embedding], axis=-1),
                                    units=self.params['task_label_size'][task_id], activation=None, use_bias=True)
            add_layer_summary('task_pred_{}'.format(task_id), preds)
        return preds

    def __call__(self, features, labels, params, is_training):
        self.params = params
        self.encoder.params = params
        self.task_ids = features['task_ids']
        embedding = self.encode(features, is_training)

        with tf.variable_scope('share_embedding'):
            share_private = tf.layers.dense(embedding, units=params['share_size'],
                                            activation=params['share_activation'])

            share_private = tf.layers.dropout(share_private, rate=params['share_dropout'], seed=1234,
                                              training=is_training)

        with tf.variable_scope('task'):
            share_private = flip_gradient(share_private, params['shrink_gradient_reverse'])

            self.task_logits = tf.layers.dense(share_private, units=params['task_size'], activation=None,
                                               use_bias=True, name='task_logits')
            add_layer_summary('task_logits', self.task_logits)

        # 一个domain一个head，预测结果只取task对应的head预测'
        predictions = []
        for id in range(self.params['task_size']):
            predictions.append(self.domain_layer(share_private, embedding, id))

        task_index = tf.stack([self.task_ids, tf.range(tf.shape(labels)[0])], axis=1)
        predictions = tf.gather_nd(predictions, task_index)  # batch_size * label_size

        return predictions, labels

    def compute_loss(self, predictions, labels):
        """
        各任务loss加权平均，这里暂不支持不同
        """
        loss_func = self.params['loss_func']
        loss = loss_func(predictions, labels)
        total_loss = 0
        for id, weight in self.params['task_weight'].items():
            task_mask = tf.cast(tf.equal(self.task_ids, id), tf.float32)
            loss = tf.reduce_sum(tf.multiply(task_mask, loss)) / (tf.reduce_sum(task_mask) + 1e-9)
            tf.summary.scalar('task_loss/loss_{}'.format(id), loss)
            total_loss += loss * weight

        task_loss = loss_func(self.task_logits, self.task_ids)
        tf.summary.scalar('task_loss/loss_disc', task_loss)
        total_loss += task_loss * self.params['task_lambda']
        return total_loss

    def init_fn(self):
        return self.encoder.init_fn()

    def optimize(self, loss):
        train_op = self.encoder.optimize(loss)
        return train_op


def get_trainer(model):
    module = importlib.import_module('model.{}.model'.format(model))
    encoder = getattr(module, '{}Encoder'.format(model.capitalize()))
    dataset = getattr(module, 'dataset')

    trainer = MultiTrainer(model_fn=build_model_fn(AdversarialWrapper(encoder())),
                           dataset_cls=partial(MultiDataset, dataset_cls=dataset))
    return trainer
