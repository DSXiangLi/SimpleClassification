# -*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from dataset.text_dataset import truncate_seq_pair
from serving.base_infer import BaseInfer
from collections import namedtuple


class ClassifyInfer(BaseInfer):
    prediction = namedtuple('ClassificationPredcition', ['prob', 'pred_class'])

    def decode_prediction(self, resp):
        if not resp:
            return self.prediction([], -1)
        res = resp.result().outputs
        prob = tf.make_ndarray(res['prob'])

        return self.prediction(prob, np.argmax(prob))


class SeqClassifyInfer(ClassifyInfer):
    """
    Infer Class for sequence model like Bert, Xlnet, Albert, Electra
    """

    def __init__(self, server_list, max_seq_len, timeout, nlp_pretrain_model, model_name, model_version):
        super(SeqClassifyInfer, self).__init__(server_list, max_seq_len, timeout, nlp_pretrain_model, model_name,
                                               model_version)
        self.proto = {
            'idx': tf.int32,
            'input_ids': tf.int32,
            'segment_ids': tf.int32,
            'seq_len': tf.int32,
        }

    def make_feature(self, input):
        """
        支持单双输入
        单输入文本：input 为string
        双输入文本：input是tuple or list of string
        """
        if isinstance(input, list) or isinstance(input, tuple):
            text1, text2 = input
        else:
            text1 = input
            text2 = None
        tokens1 = self.tokenizer.tokenize(text1)
        tokens2 = self.tokenizer.tokenize(text2) if text2 else []
        if tokens2:
            tokens1, tokens2 = truncate_seq_pair(tokens1, tokens2, self.max_seq_len - 3)
            tokens1 += ['[SEP]']
            tokens2 += ['[SEP]']
        else:
            tokens1 = tokens1[:(self.max_seq_len - 2)] + ['[SEP]']

        tokens = ['[CLS]']
        segment_ids = [0]
        for i in tokens1:
            tokens.append(i)
            segment_ids.append(0)
        for i in tokens2:
            tokens.append(i)
        return {'idx': [0],
                'input_ids': [self.tokenizer.convert_tokens_to_ids(tokens)],
                'segment_ids': [segment_ids],
                'seq_len': [len(tokens)]}


class WordClassifyInfer(ClassifyInfer):
    """
    Infer Class for word emebdding model like Fasttext, TextCNN, Fasttext
    """
    def __init__(self, server_list, max_seq_len, timeout, nlp_pretrain_model, model_name, model_version):
        super(WordClassifyInfer, self).__init__(server_list, max_seq_len, timeout, nlp_pretrain_model, model_name,
                                               model_version)
        self.proto = {
            'idx': tf.int32,
            'input_ids': tf.int32,
            'seq_len': tf.int32,
        }

    def make_feature(self, input):
        """
        支持单双输入
        单输入文本：input 为string
        双输入文本：input是tuple or list of string
        """
        if isinstance(input, list) or isinstance(input, tuple):
            text1, text2 = input
        else:
            text1 = input
            text2 = None
        tokens1 = self.tokenizer.tokenize(text1)
        tokens2 = self.tokenizer.tokenize(text2) if text2 else []
        if tokens2:
            tokens1, tokens2 = truncate_seq_pair(tokens1, tokens2, self.max_seq_len)
        else:
            tokens1 = tokens1[:self.max_seq_len]

        tokens = []
        for i in tokens1:
            tokens.append(i)
        for i in tokens2:
            tokens.append(i)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return {'idx': [0],
                'input_ids': [input_ids],
                'seq_len': [len(input_ids)]}
