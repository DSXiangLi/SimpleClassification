# -*-coding:utf-8 -*-
"""
    Ref1. Adversarial Training Methods for Semi-Supervised Text Classification
    Ref2. Distributional smoothing with virtual adversarial training
    Ref3. Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning
"""
import importlib
import tensorflow.compat.v1 as tf

from model.train_helper import build_model_fn, Trainer, BaseEncoder
from tools.train_utils import HpParser, add_layer_summary, get_variables

hp_list = [HpParser.hp('epsilon', 5.0), # norm length for adversarial training
           HpParser.hp('embedding_name','word_embeddings') #BERT扰动整体词向量，词袋模型扰动输入词向量
]
hp_parser = HpParser(hp_list)


class FGM(object):
    def __init__(self):
        self.epsilon = None
        self.embedding_name = None
        self.adv_perturb = None

    def init(self, epsilon, embedding_name):
        self.epsilon = epsilon
        self.embedding_name = embedding_name

    def get_adv_perturbation(self, embedding, loss):
        """
        embedding: vocab_size * embedding_size
        """
        grad = tf.gradients(loss, embedding, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad[0])
        add_layer_summary('grad', grad)
        add_layer_summary('embedding', embedding)
        perturb = self.epsilon * tf.nn.l2_normalize(grad)
        add_layer_summary('perturb', perturb)
        return perturb

    def find_embedding(self):
        tvars = get_variables()
        for var in tvars:
            if self.embedding_name in var.name:
                return var
        raise ValueError('embedding variable {} not found in tavrs {}'.format(
            self.embedding_name, tvars))

    def attack(self, loss, features, forward_func, is_training):
        """
        add perturb to vocab embedding
        1. generate perturb
        2. add perturb to vocab embedding
        3. recompute forward propagration
        """
        embedding = self.find_embedding()
        self.adv_perturb = self.get_adv_perturbation(embedding, loss)
        embedding.assign(embedding + self.adv_perturb)
        logit = forward_func(features, is_training)
        embedding.assign(embedding - self.adv_perturb)
        return logit


class FgmWrapper(BaseEncoder):
    def __init__(self, encoder):
        super(FgmWrapper, self).__init__()
        self.encoder = encoder
        self.fgm = FGM()
        self.features = None # for second propagate after adv
        self.is_training = None

    def encode(self, features, is_training):
        return self.encoder.encode(features, is_training)

    def forward(self, features, is_training):
        embedding = self.encode(features, is_training)

        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            preds = tf.layers.dense(embedding, units=self.params['label_size'], activation=None, use_bias=True)
            add_layer_summary('preds', preds)
        return preds

    def __call__(self, features, labels, params, is_training):
        self.params = params
        self.encoder.params = params
        self.features = features
        self.fgm.init(params['epsilon'], params['embedding_name'])
        preds = self.forward(features, is_training)
        return preds, labels

    def init_fn(self):
        return self.encoder.init_fn()

    def optimize(self, loss):
        train_op = self.encoder.optimize(loss)
        return train_op

    def compute_loss(self, predictions, labels):
        supervised_loss = self.params['loss_func'](predictions, labels)
        supervised_loss = tf.reduce_mean(supervised_loss)
        adv_logits = self.fgm.attack(supervised_loss, self.features, self.forward, self.is_training)
        adv_loss = self.params['loss_func'](adv_logits, labels)
        adv_loss = tf.reduce_mean(adv_loss)
        tf.summary.scalar('loss/adv_loss', adv_loss)
        total_loss = adv_loss + supervised_loss
        return total_loss


def get_trainer(model):
    module = importlib.import_module('model.{}.model'.format(model))
    encoder = getattr(module, '{}Encoder'.format(model.capitalize()))
    dataset = getattr(module, 'dataset')

    trainer = Trainer(model_fn=build_model_fn(FgmWrapper(encoder())),
                      dataset_cls=dataset)
    return trainer
