# -*-coding:utf-8 -*-
import os
import tensorflow.compat.v1 as tf
from backbone.albert import optimization, modeling
from tools.train_utils import add_layer_summary, get_variables, HpParser
from dataset.text_dataset import SeqDataset as dataset
from model.train_helper import Trainer, build_model_fn, BaseEncoder


hp_list = [HpParser.hp('warmup_ratio', 0.1),
           HpParser.hp('embedding_dropout', 0.3),
           HpParser.hp('albert_optimizer', 'adamw')]
hp_parser = HpParser(hp_list)


class AlbertEncoder(BaseEncoder):
    def __init__(self):
        super(AlbertEncoder, self).__init__()

    def encode(self, features, is_training):
        albert_config = modeling.AlbertConfig.from_json_file(os.path.join(self.params['nlp_pretrain_dir'],
                                                                        'albert_config.json'))

        albert_model = modeling.AlbertModel(
            config=albert_config,
            is_training=is_training,
            input_ids=features['input_ids'],
            input_mask=self.get_input_mask(features['seq_len']),
            token_type_ids=features['segment_ids'],
            use_one_hot_embeddings=False,
            scope='bert'
        )

        embedding = albert_model.get_pooled_output()
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
        assignment_map,_ = modeling.get_assignment_map_from_checkpoint(tvars, self.params['nlp_pretrain_ckpt'])
        tf.train.init_from_checkpoint(self.params['nlp_pretrain_ckpt'], assignment_map)
        return None

    def optimize(self, loss):
        train_op = optimization.create_optimizer(loss, self.params['lr'],
                                                 self.params['num_train_steps'],
                                                 int(self.params['num_train_steps'] * self.params['warmup_ratio']),
                                                 False,  self.params['albert_optimizer'])
        return train_op


trainer = Trainer(model_fn=build_model_fn(AlbertEncoder()),
                  dataset_cls=dataset)


