# -*-coding:utf-8 -*-
import os
import tensorflow as tf
from backbone.electra import optimization, modeling
from tools.train_utils import add_layer_summary, get_variables, get_assignment_from_ckpt, HpParser
from dataset.text_dataset import SeqDataset as dataset
from dataset.tokenizer import get_tokenizer
from model.train_helper import BaseTrainer, build_model_fn, BaseEncoder


hp_list = [HpParser.hp('warmup_ratio', 0.1),
           HpParser.hp('embedding_dropout', 0.3)]
hp_parser = HpParser(hp_list)


class ElectraEncoder(BaseEncoder):
    def __init__(self):
        super(ElectraEncoder, self).__init__()

    def encode(self, features, is_training):
        electra_config = modeling.BertConfig.from_json_file(os.path.join(self.params['nlp_pretrain_dir'],
                                                                      'base_discriminator_config.json'))

        electra_model = modeling.BertModel(
            bert_config=electra_config,
            is_training=is_training,
            input_ids=features['input_ids'],
            input_mask=self.get_input_mask(features['seq_len']),
            token_type_ids=features['segment_ids'],
            use_one_hot_embeddings=False,
            scope='electra'
        )

        embedding = electra_model.get_pooled_output()
        embedding = tf.layers.dropout(embedding, rate=self.params['embedding_dropout'], seed=1234, training=is_training)
        add_layer_summary('ouput_emb', embedding)
        return embedding

    def __call__(self, features, labels, params, is_training):
        self.params = params
        embedding = self.encode(features, is_training)
        with tf.variable_scope('transfer'):
            preds = tf.layers.dense(embedding, units=self.params['label_size'], activation=None, use_bias=True)
            add_layer_summary('preds', preds)
        return preds, labels

    def init_fn(self):
        """
        # load vars from ckpt: Default for finetune all bert variables
        """
        tvars = get_variables(trainable_only=True)
        assignment_map = get_assignment_from_ckpt(self.params['nlp_pretrain_ckpt'], tvars)
        tf.train.init_from_checkpoint(self.params['nlp_pretrain_ckpt'], assignment_map)
        return None

    def optimize(self, loss):
        train_op = optimization.create_optimizer(loss, self.params['lr'],
                                                 self.params['num_train_steps'],
                                                 int(self.params['num_train_steps'] * self.params['warmup_ratio']),
                                                 False)
        return train_op


class Trainer(BaseTrainer):
    def __init__(self, model_fn, dataset_cls):
        super(Trainer, self).__init__(model_fn, dataset_cls)

    def prepare(self):
        self.logger.info('Prepare dataset')
        self.input_pipe = self.dataset_cls(data_dir=self.train_params['data_dir'],
                                           batch_size=self.train_params['batch_size'],
                                           max_seq_len=self.train_params['max_seq_len'],
                                           tokenizer=get_tokenizer(self.train_params['nlp_pretrain_model']),
                                           enable_cache=self.train_params['enable_cache'],
                                           clear_cache=self.train_params['clear_cache'])
        self.input_pipe.build_feature('train')

        self.train_params.update({
            'model_dir': self.train_params['ckpt_dir'],
            'sample_size': self.input_pipe.sample_size,
            'steps_per_epoch': self.input_pipe.steps_per_epoch,
            'num_train_steps': int(self.input_pipe.steps_per_epoch * self.train_params['epoch_size'])
        })


trainer = Trainer(model_fn=build_model_fn(ElectraEncoder()),
                  dataset_cls=dataset)


