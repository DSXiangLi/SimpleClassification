# -*-coding:utf-8 -*-
import tensorflow.compat.v1 as tf
from dataset.text_dataset import WordDataset as dataset
from model.train_helper import Trainer, build_model_fn, BaseEncoder
from tools.opt_utils import train_op_clip_decay
from tools.train_utils import add_layer_summary, HpParser

hp_list = [HpParser.hp('embedding_dropout', 0.3),
           HpParser.hp('keep_oov', 0),
           HpParser.hp('lower_clip', -5),
           HpParser.hp('upper_clip', 5),
           HpParser.hp('decay_rate', 0.95),
           HpParser.hp('cell_type', 'lstm'),
           HpParser.hp('cell_size', 1),# 几层RNN
           HpParser.hp('cell_hidden_list', '128', lambda x: [int(i) for i in x.split(',')]),
           HpParser.hp('keep_prob_list', '0.8', lambda x: [float(i) for i in x.split(',')]),
           HpParser.hp('rnn_activation', 'tanh')
           ]
hp_parser = HpParser(hp_list)


def build_rnn_cell(cell_type, activation, hidden_units_list, keep_prob_list, cell_size):
    if cell_type.lower() == 'rnn':
        cell_class = tf.nn.rnn_cell.RNNCell
    elif cell_type.lower() == 'gru':
        cell_class = tf.nn.rnn_cell.GRUCell
    elif cell_type.lower() == 'lstm':
        cell_class = tf.nn.rnn_cell.LSTMCell
    else:
        raise Exception('Only rnn, gru, lstm are supported as cell_type')

    return tf.nn.rnn_cell.MultiRNNCell(
        cells=[tf.nn.rnn_cell.DropoutWrapper(cell=cell_class(num_units=hidden_units_list[i], activation=activation),
                                             output_keep_prob=keep_prob_list[i],
                                             state_keep_prob=keep_prob_list[i]) for i in range(cell_size)])


def bilstm(embedding, cell_type, activation, hidden_units_list, keep_prob_list, cell_size, seq_len, is_training):
    with tf.variable_scope('bilstm_layer'):
        if not is_training:
            keep_prob_list = len(keep_prob_list) * [1.0]
        fw = build_rnn_cell(cell_type, activation, hidden_units_list, keep_prob_list, cell_size)
        bw = build_rnn_cell(cell_type, activation, hidden_units_list, keep_prob_list, cell_size)

        # tuple of 2 : batch_size * max_seq_len * hidden_size
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, embedding, seq_len, dtype=tf.float32)

        # concat forward and backward along embedding axis
        outputs = tf.concat(outputs, axis=-1)  # batch_size * max_seq_len * (hidden_size * 2)
        add_layer_summary('bilstm_concat', outputs)
    return outputs


class TextrcnnEncoder(BaseEncoder):
    def __init__(self):
        super(TextrcnnEncoder, self).__init__()
        self.embedding = None

    def encode(self, features, is_training):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable(shape=[self.params['vocab_size'], self.params['embedding_size']],
                                             dtype=tf.float32, name='pretrain_embedding')
            input_emb = tf.nn.embedding_lookup(self.embedding, features['input_ids'], name='input_emb')
            input_emb = tf.layers.dropout(input_emb, rate=self.params['embedding_dropout'], seed=1234, training=is_training)
            add_layer_summary('input_emb', input_emb)

        with tf.variable_scope('textrcnn', reuse=tf.AUTO_REUSE):
            lstm_output = bilstm(input_emb, self.params['cell_type'], self.params['rnn_activation'],
                                 self.params['cell_hidden_list'], self.params['keep_prob_list'],
                                 self.params['cell_size'], features['seq_len'], is_training)  # batch, max_seq_len, emb*2

            output_emb = tf.reduce_max(lstm_output, axis=1, keep_dims=False)

        return output_emb

    def __call__(self, features, labels, params, is_training):
        self.params = params
        embedding = self.encode(features, is_training)
        with tf.variable_scope('mlp'):
            preds = tf.layers.dense(embedding, units=self.params['label_size'], activation=None, use_bias=True)
            add_layer_summary('preds', preds)
        return preds, labels

    def init_fn(self):
        """
        # load embedding in params using scaffold
        """
        def init_fn(scaffold, sess):
            sess.run(self.embedding.initializer, {self.embedding.initial_value: self.params['embedding']})

        scaffold = tf.train.Scaffold(init_fn=init_fn)
        return scaffold

    def optimize(self, loss):
        """
        Use Adam Optimizer with gradient clip and exponential decay
        """
        train_op = train_op_clip_decay(loss, self.params['lr'],
                                       self.params['num_train_steps'],  self.params['decay_rate'],
                                       self.params['lower_clip'], self.params['upper_clip']
                                       )
        return train_op


trainer = Trainer(model_fn=build_model_fn(TextrcnnEncoder()),
                  dataset_cls=dataset)


