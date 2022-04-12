# -*-coding:utf-8 -*-
"""
    预训练模型和对应模型文件
"""
from collections import namedtuple

PTM = namedtuple('PTM', ['model_dir', 'model_file'])

PRETRAIN_CONFIG = {
    'bert_base': PTM('pretrain/chinese_L-12_H-768_A-12', 'bert_model.ckpt'),
    'roberta_base': PTM('pretrain/roberta_zh_L-24_H-1024_A-16', 'bert_model.ckpt'),
    'bert_base_wwm': PTM('pretrain/chinese_wwm_L-12_H-768_A-12', 'bert_model.ckpt'),

    'fasttext':  PTM('pretrain_model/fasttext', 'cc.zh.300.bin'),
    'word2vec_news': PTM('pretrain/word2vec_news', 'sgns.renmin.bigram-char.bz2'),
    'word2vec_baike': PTM('pretrain/word2vec_baike', 'sgns.merge.word'),

    'ctb50': PTM('pretrain/ctb50', 'ctb.50d.vec'),
    'giga': PTM('pretrain/giga', 'gigaword_chn.all.a2b.uni.ite50.vec')
}