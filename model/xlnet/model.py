# -*-coding:utf-8 -*-
import os
import tensorflow.compat.v1 as tf
from backbone.xlnet.xlnet import XLNetModel, XLNetConfig, RunConfig
from backbone.bert import optimization
from tools.train_utils import add_layer_summary, get_variables, get_assignment_from_ckpt, HpParser
from dataset.text_dataset import SeqDataset as dataset
from model.train_helper import Trainer, build_model_fn, BaseEncoder

hp_list = [HpParser.hp('warmup_ratio', 0.1),
           HpParser.hp('embedding_dropout', 0.3)]
hp_parser = HpParser(hp_list)


class XlnetEncoder(BaseEncoder):
    def __init__(self):
        super(XlnetEncoder, self).__init__()

    def get_input_mask(self, seq_len):
        # 注意 xlnet的input mask和常规是相反的，real token是0，pad是1
        maxlen = tf.reduce_max(seq_len)
        input_mask = 1.0 - tf.cast(tf.sequence_mask(seq_len, maxlen=maxlen), tf.float32)
        return input_mask

    def encode(self, features, is_training):
        xlnet_config = XLNetConfig(json_path=os.path.join(self.params['nlp_pretrain_dir'],
                                                          'xlnet_config.json'))
        # keep Xlnet pretrain config unchanged
        run_config = RunConfig(is_training=is_training,
                               use_tpu=False, use_bfloat16=False,
                               dropout=0.1, dropatt=0.1)
        # Attention!!!! Xlnet input shape is different: [seq_len, batch_size]
        xlnet_model = XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=tf.transpose(features['input_ids'], perm=[1, 0]),
            input_mask=tf.transpose(self.get_input_mask(features['seq_len']), perm=[1, 0]),
            seg_ids=tf.transpose(features['segment_ids'], perm=[1, 0])
        )

        embedding = xlnet_model.get_pooled_out(summary_type='last', use_summ_proj=True)
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
        # 复用Bert的optimization基本差不多，xlnet自己写的需要全局变量不方便引用。。。。
        train_op = optimization.create_optimizer(loss, self.params['lr'],
                                                 self.params['num_train_steps'],
                                                 int(self.params['num_train_steps'] * self.params['warmup_ratio']),
                                                 False)
        return train_op


trainer = Trainer(model_fn=build_model_fn(XlnetEncoder()),
                  dataset_cls=dataset)
