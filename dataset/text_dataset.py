# -*-coding:utf-8 -*-
import tensorflow as tf
from dataset.base_dataset import GeneratorDataset


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

        self.label_names = ['label']
        self.feature_names.extend(['input_ids', 'segment_ids', 'seq_len'])

    def build_single_feature(self, data):
        tokens = self.tokenizer.tokenize(data['text1'][:self.max_seq_len])
        tokens = tokens[:(self.max_seq_len-2)]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        seq_len = len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * seq_len
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
        self.shapes = {
            'input_ids': [None],
            'seq_len': [],
            'label': []
        }
        self.pads = {
            'input_ids': self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0],
            'seq_len': 0,
            'label': 0
        }

        self.label_names = ['label']
        self.feature_names = ['input_ids', 'seq_len']

    def build_single_feature(self, data):
        tokens = self.tokenizer.tokenize(data['text1'][:self.max_seq_len])
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        seq_len = len(input_ids) # do after ids due to oov removal
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


if __name__ =='__main__':
    from dataset.tokenizer import get_tokenizer
    pipe = SeqDataset('./trainsample/weibo',5, 50, get_tokenizer('bert'), True)
    pipe.build_feature('train')

    pipe = WordDataset('./trainsample/weibo',5, 50, get_tokenizer('word2vec_baike'), False)
    pipe.build_feature('test')