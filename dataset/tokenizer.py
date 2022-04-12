# -*-coding:utf-8 -*-
import jieba_fast as jieba
import os
import numpy as np
from functools import partial
from pretrain.config import PRETRAIN_CONFIG
from tools.logger import logger
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from pathlib import Path


class CustomTokenizer(object):
    """
    Word Embedding Tokenizer Adapter
    """
    def __init__(self, model, addon_vocab=('[UNK]', '[PAD]'), keep_oov=True):
        self.model = model
        self.keep_oov = keep_oov
        self.vocab2idx = None
        self.idx2vocab = None
        self.embedding = None
        self.vocab_size = None
        self.embedding_size = None
        self.addon_vocab = addon_vocab

    def init_vocab(self):
        raise NotImplementedError

    def _add_vocab(self, vocab):
        self.vocab2idx.update({vocab: self.vocab_size})
        self.vocab_size +=1

    def _add_embedding(self):
        self.embedding = np.vstack((self.embedding,
                                    np.random.normal(0, 1, size=(1, self.embedding_size))))

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for i in tokens:
            if i in self.vocab2idx:
                ids.append(self.vocab2idx[i])
            elif self.keep_oov:
                ids.append(self.vocab2idx['[UNK]'])
            else:
                pass
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.idx2vocab[i])
        return tokens


class JiebaTokenizer(CustomTokenizer):
    def tokenize(self, input_):
        tokens = []
        if not isinstance(input_, list):
            tokens = [i for i in jieba.cut(input_) if i.strip()]
        return tokens


class GensimJiebaTokenizer(JiebaTokenizer):
    def init_vocab(self):
        self.vocab2idx = dict([(word, idx) for idx, word in enumerate(self.model.wv.index2word)])
        self.idx2vocab = dict([(j,i ) for i,j in self.vocab2idx.items()])
        self.embedding = np.array(self.model.wv.syn0).astype(np.float32)
        self.vocab_size = len(self.vocab2idx)

        for i in self.addon_vocab:
            self._add_vocab(i)
            self._add_embedding()


def glove2wv(model_name):
    abs_path = Path(__file__).absolute().parent

    glove_file = datapath(os.path.join(abs_path, model_name))
    tmp_file = get_tmpfile(os.path.join(abs_path, model_name + 'tmp'))
    if os.path.isfile(os.path.join(abs_path, tmp_file)):
        print('Tmp file already existed, only generate tmp file 1 time')
    else:
        _ = glove2word2vec(glove_file, tmp_file)

    model = KeyedVectors.load_word2vec_format(tmp_file)
    return model


def get_fasttext_tokenizer(vocab_file, **kwargs):
    logger.info('Loading vocab_file {} '.format(vocab_file))
    from gensim.models.wrappers import FastText
    model = FastText.load_fasttext_format(vocab_file)
    tokenizer = GensimJiebaTokenizer(model, **kwargs)
    tokenizer.init_vocab()
    return tokenizer


def get_word2vec_tokenizer(vocab_file, **kwargs):
    from gensim.models import KeyedVectors
    logger.info('Loading vocab_file {} '.format(vocab_file))
    model = KeyedVectors.load_word2vec_format(vocab_file)
    tokenizer = GensimJiebaTokenizer(model, **kwargs)
    tokenizer.init_vocab()
    return tokenizer


def get_glove_tokenizer(vocab_file, **kwargs):
    logger.info('Loading vocab_file {} '.format(vocab_file))
    model = glove2wv(vocab_file)
    tokenizer = GensimJiebaTokenizer(model, **kwargs)
    return tokenizer


def get_bert_tokenizer(vocab_file, **kwargs):
    logger.info('Loading vocab_file {} '.format(vocab_file))
    from bert_base.bert import tokenization
    tokenizer = tokenization.FullTokenizer(vocab_file,  **kwargs)
    return tokenizer


def get_tokenizer(name, **kwargs):
    """
    Lazy tokenizer factory loader
    """
    basepath = os.path.abspath(__file__)
    pkg_path = os.path.dirname(os.path.dirname(basepath))
    tokenizer_factory = {
        'bert_base': partial(get_bert_tokenizer,
                             vocab_file=os.path.join(pkg_path, PRETRAIN_CONFIG['bert_base'].model_dir, 'vocab.txt')),
        'bert_base_wwm': partial(get_bert_tokenizer,
                                 vocab_file=os.path.join(pkg_path, PRETRAIN_CONFIG['bert_base_wwm'].model_dir, 'vocab.txt')),
        'roberta_base': partial(get_bert_tokenizer,
                                 vocab_file=os.path.join(pkg_path, PRETRAIN_CONFIG['roberta_base'].model_dir, 'vocab.txt')),

        'fasttext': partial(get_fasttext_tokenizer,
                            vocab_file=os.path.join(pkg_path, *PRETRAIN_CONFIG['fasttext'])),
        'word2vec_news': partial(get_word2vec_tokenizer,
                                 vocab_file=os.path.join(pkg_path, *PRETRAIN_CONFIG['word2vec_news'])),
        'word2vec_baike': partial(get_word2vec_tokenizer,
                                  vocab_file=os.path.join(pkg_path, *PRETRAIN_CONFIG['word2vec_baike'])),
        'giga': partial(get_glove_tokenizer,
                        vocab_file=os.path.join(pkg_path, *PRETRAIN_CONFIG['giga'].model_dir)),
        'ctb50': partial(get_glove_tokenizer,
                         vocab_file=os.path.join(pkg_path, *PRETRAIN_CONFIG['ctb50'].model_dir)),
    }

    if name not in tokenizer_factory:
        raise ValueError('Only {} are supported'.format(tokenizer_factory.keys()))
    else:
        return tokenizer_factory[name](**kwargs)

