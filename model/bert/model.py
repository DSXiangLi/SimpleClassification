# -*-coding:utf-8 -*-
import os
import tensorflow as tf
from backbone.bert import optimization, modeling
from tools.train_utils import add_layer_summary, get_variables, get_assignment_from_ckpt, HpParser
from dataset.text_dataset import SeqDataset as dataset
from model.train_helper import Trainer, build_model_fn, BaseEncoder


hp_list = [HpParser.hp('warmup_ratio', 0.1),
           HpParser.hp('embedding_dropout', 0.3)]
hp_parser = HpParser(hp_list)


class BertEncoder(BaseEncoder):
    def __init__(self):
        super(BertEncoder, self).__init__()

    def encode(self, features, is_training):
        bert_config = modeling.BertConfig.from_json_file(os.path.join(self.params['nlp_pretrain_dir'], 'bert_config.json'))

        bert_model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=features['input_ids'],
            input_mask=self.get_input_mask(features['seq_len']),
            token_type_ids=features['segment_ids'],
            use_one_hot_embeddings=False,
            scope='bert'
        )

        embedding = bert_model.get_pooled_output()
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


trainer = Trainer(model_fn=build_model_fn(BertEncoder()),
                  dataset_cls=dataset)


