# -*-coding:utf-8 -*-
import tensorflow as tf

from tools.train_utils import add_layer_summary, get_log_hook
from tools.metrics import binary_cls_metrics


def build_graph(features, labels, params, is_training):
    input_ids = features['input_ids']
    seq_len = features['seq_len']
    maxlen = tf.reduce_max(seq_len)

    with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable(shape=[params['vocab_size'], params['embedding_size']],
                                    dtype=tf.float32, name='pretrain_embedding')

    def init_fn(scaffold, sess):
        sess.run(embedding.initializer, {embedding.initial_value: params['embedding']})

    input_emb = tf.nn.embedding_lookup(embedding, input_ids, name='input_emb')
    input_emb = tf.layers.dropout(input_emb, rate=params['input_dropout'], seed=1234, training=is_training)
    add_layer_summary('input_emb', input_emb)

    with tf.variable_scope('fasttext'):
        mask = tf.cast(tf.sequence_mask(seq_len, maxlen=maxlen), tf.float32)
        input_emn = tf
