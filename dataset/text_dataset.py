# -*-coding:utf-8 -*-
import tensorflow.compat.v1 as tf
from dataset.base_dataset import GeneratorDataset


def balance_truncate(tokens1, tokens2, max_len):
    while True:
        total_length = len(tokens1) + len(tokens2)
        if total_length <= max_len:
            break
        if len(tokens1) > len(tokens2):
            tokens1.pop()
        else:
            tokens2.pop()
    return tokens1, tokens2


def first_truncate(tokens1, tokens2, max_len):
    len1 = len(tokens1)
    tokens2 = tokens2[:(max_len - len1)]
    return tokens1, tokens2


def truncate_seq_pair(tokens1, tokens2, max_len, method='balance'):
    """
    双输入截断,默认使用balance方案
    method=balance：text1，text2轮流-1，多用于STS
    method='first'：优先截断text2，多用于title+content类型, 默认max_len>max(text1), 如果不满足会报错
    """
    if method == 'first' and len(tokens1) > max_len:
        raise ValueError('[first] method is mainly used for short text1, its length should be < max_len')

    if method == 'balance':
        return balance_truncate(tokens1, tokens2, max_len)
    elif method == 'first':
        return first_truncate(tokens1, tokens2, max_len)
    else:
        raise ValueError('Method must be in [balance, first]')


class SeqDataset(GeneratorDataset):
    def __init__(self, data_dir, batch_size, max_seq_len, tokenizer, enable_cache, clear_cache):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        super(SeqDataset, self).__init__(data_dir, batch_size, enable_cache, clear_cache)

    def build_proto(self):
        self.dtypes.update({
            'input_ids': tf.int32,
            'segment_ids': tf.int32,
            'seq_len': tf.int32,
            'label': tf.int32
        })
        self.shapes.update({
            'input_ids': [None],
            'segment_ids': [None],
            'seq_len': [],
            'label': []
        })
        self.pads.update({
            'input_ids': self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0],
            'segment_ids': 1,
            'seq_len': 0,
            'label': 0
        })

        self.label_names.extend(['label'])
        self.feature_names.extend(['input_ids', 'segment_ids', 'seq_len'])

    def build_single_feature(self, data):
        tokens1 = self.tokenizer.tokenize(data['text1'])
        tokens2 = self.tokenizer.tokenize(data['text2']) if 'text2' in data else []
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
            segment_ids.append(1)

        seq_len = len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'seq_len': seq_len,
            'idx': int(data['idx']),
            'label': int(data['label'])
        }

    def update_params(self, train_params):
        train_params.update({
            'sample_size': self.sample_size,
            'steps_per_epoch': self.steps_per_epoch,
            'num_train_steps': int(self.steps_per_epoch * train_params['epoch_size'])
        })
        return train_params


class WordDataset(GeneratorDataset):
    def __init__(self, data_dir, batch_size, max_seq_len, tokenizer, enable_cache, clear_cache):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        super(WordDataset, self).__init__(data_dir, batch_size, enable_cache, clear_cache)

    def build_proto(self):
        self.dtypes.update({
            'input_ids': tf.int32,
            'seq_len': tf.int32,
            'label': tf.int32
        })
        self.shapes.update({
            'input_ids': [None],
            'seq_len': [],
            'label': []
        })
        self.pads.update({
            'input_ids': self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0],
            'seq_len': 0,
            'label': 0
        })

        self.label_names.extend(['label'])
        self.feature_names.extend({'input_ids', 'seq_len'})

    def build_single_feature(self, data):
        tokens1 = self.tokenizer.tokenize(data['text1'])
        tokens2 = self.tokenizer.tokenize(data['text2']) if 'text2' in data else []
        if tokens2:
            tokens1, tokens2 = truncate_seq_pair(tokens1, tokens2, self.max_seq_len)
        else:
            tokens1 = tokens1[:self.max_seq_len]
        tokens = tokens1 + tokens2

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        seq_len = len(input_ids)  # do after ids due to oov removal
        return {
            'input_ids': input_ids,
            'seq_len': seq_len,
            'label': int(data['label']),
            'idx': int(data['idx'])
        }

    def update_params(self, train_params):
        train_params.update({
            'embedding': self.tokenizer.embedding,
            'embedding_size': self.tokenizer.embedding_size,
            'vocab_size': self.tokenizer.vocab_size,
            'sample_size': self.sample_size,
            'steps_per_epoch': self.steps_per_epoch,
            'num_train_steps': int(self.steps_per_epoch * train_params['epoch_size'])
        })
        return train_params


if __name__ == '__main__':
    import os
    from dataset.tokenizer import get_tokenizer

    pipe =SeqDataset('./trainsample/weibo', 5, 50, get_tokenizer('bert_base'), False, False)

    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    pipe.build_feature('test_teacher')
    sess = tf.Session()
    it = tf.data.make_one_shot_iterator(pipe.build_input_fn()())
    f = sess.run(it.get_next())

    # pipe = WordDataset('./trainsample/weibo', 5, 50, get_tokenizer('word2vec_baike'), False, False)
    # pipe.build_feature('test')
