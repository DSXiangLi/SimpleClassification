# -*-coding:utf-8 -*-
import tensorflow as tf
from dataset.text_dataset import WordDataset as dataset
from dataset.tokenizer import get_tokenizer
from model.train_helper import BaseTrainer, build_model_fn, BaseEncoder
from tools.opt_utils import train_op_clip_decay
from tools.train_utils import add_layer_summary, HpParser

hp_list = [HpParser.hp('embedding_dropout', 0.3),
           HpParser.hp('keep_oov', 0),
           HpParser.hp('lower_clip', -5),
           HpParser.hp('upper_clip', 5),
           HpParser.hp('decay_rate', 0.95),
           HpParser.hp('filter_list', '100,100,100', lambda x: [int(i) for i in x.split(',')]),
           HpParser.hp('kernel_size_list', '3,4,5', lambda x: [int(i) for i in x.split(',')]),
           HpParser.hp('cnn_activation', 'relu'),
           HpParser.hp('concat_cnn', True), # if concat use concat, else use stack

           ]
hp_parser = HpParser(hp_list)


def concat_cnn(embedding, filter_list, kernel_size_list, activation):
    outputs = []
    for i in range(len(filter_list)):
        # batch * max_seq_len * filters
        output = tf.layers.conv1d(
            inputs=embedding,
            filters=filter_list[i],
            kernel_size=kernel_size_list[i],
            padding='VALID',
            activation=activation,
            name='concat_cnn_kernel{}'.format(kernel_size_list[i])
        )
        add_layer_summary(output.name, output)
        outputs.append(tf.reduce_max(output, axis=1))  # batch_size * filters
    output = tf.concat(outputs, axis=-1)  # batch_size * max_seq_len * sum(filter_list)
    return output


def stack_cnn(embedding, filter_list, kernel_size_list, activation,):
    for i in range(len(filter_list)):
        # batch * max_seq_len * filters
        embedding = tf.layers.conv1d(
            inputs=embedding,
            filters=filter_list[i],
            kernel_size=kernel_size_list[i],
            padding='VALID',
            activation=activation,
            name='stack_cnn_kernel{}'.format(kernel_size_list[i])
        )
        add_layer_summary(embedding.name, embedding)
    output = tf.reduce_max(embedding, axis=1)
    return output


class TextcnnEncoder(BaseEncoder):
    def __init__(self):
        super(TextcnnEncoder, self).__init__()
        self.params = None
        self.embedding = None

    def encode(self, features, is_training):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable(shape=[self.params['vocab_size'], self.params['embedding_size']],
                                             dtype=tf.float32, name='pretrain_embedding')
            input_emb = tf.nn.embedding_lookup(self.embedding, features['input_ids'], name='input_emb')
            input_emb = tf.layers.dropout(input_emb, rate=self.params['embedding_dropout'], seed=1234, training=is_training)
            add_layer_summary('input_emb', input_emb)

        with tf.variable_scope('textcnn', reuse=tf.AUTO_REUSE):
            mask = tf.cast(self.get_input_mask(features['seq_len']), tf.float32)
            input_emb = tf.multiply(input_emb, tf.expand_dims(mask, axis=-1))
            if self.params['concat_cnn']:
                output_emb = concat_cnn(input_emb, self.params['filter_list'], self.params['kernel_size_list'],
                                         self.params['cnn_activation'])
            else:
                output_emb = stack_cnn(input_emb, self.params['filter_list'], self.params['kernel_size_list'],
                                         self.params['cnn_activation'])
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


class Trainer(BaseTrainer):
    def __init__(self, model_fn, dataset_cls):
        super(Trainer, self).__init__(model_fn, dataset_cls)

    def prepare(self):
        self.logger.info('Prepare dataset')
        self.input_pipe = self.dataset_cls(data_dir=self.train_params['data_dir'],
                                           batch_size=self.train_params['batch_size'],
                                           max_seq_len=self.train_params['max_seq_len'],
                                           tokenizer=get_tokenizer(self.train_params['nlp_pretrain_model'],
                                                                   keep_oov=self.train_params['keep_oov']),
                                           enable_cache=self.train_params['enable_cache'],
                                           clear_cache=self.train_params['clear_cache'])
        self.input_pipe.build_feature('train')
        self.train_params = self.input_pipe.update_params(self.train_params)


trainer = Trainer(model_fn=build_model_fn(TextcnnEncoder()),
                  dataset_cls=dataset)


