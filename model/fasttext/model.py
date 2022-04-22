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
           HpParser.hp('decay_rate', 0.95)]
hp_parser = HpParser(hp_list)


class FasttextEncoder(BaseEncoder):
    def __init__(self):
        super(FasttextEncoder, self).__init__()
        self.embedding = None

    def encode(self, features, is_training):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable(shape=[self.params['vocab_size'], self.params['embedding_size']],
                                             dtype=tf.float32, name='pretrain_embedding')
            input_emb = tf.nn.embedding_lookup(self.embedding, features['input_ids'], name='input_emb')
            input_emb = tf.layers.dropout(input_emb, rate=self.params['embedding_dropout'], seed=1234, training=is_training)
            add_layer_summary('input_emb', input_emb)

        with tf.variable_scope('avg_pool', reuse=tf.AUTO_REUSE):
            mask = tf.cast(self.get_input_mask(features['seq_len']), tf.float32)
            input_emb = tf.multiply(input_emb, tf.expand_dims(mask, axis=-1)) # mask PAD
            output_emb = tf.reduce_sum(input_emb, axis=1)/tf.reduce_sum(mask) # divide by number of tokens
            add_layer_summary('output_emb', output_emb)
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


trainer = Trainer(model_fn=build_model_fn(FasttextEncoder()),
                  dataset_cls=dataset)


