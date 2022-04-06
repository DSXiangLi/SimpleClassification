# -*-coding:utf-8 -*-
"""
    预训练模型和对应模型文件
"""
from collections import namedtuple

PTM = namedtuple('PTM', ['model_dir', 'model_file'])

PRETRAIN_CONFIG = {
    'bert_base': PTM('pretrain/chinese_L-12_H-768_A-12', 'bert_model.ckpt'),
    'word2vec_news': PTM('pretrain/people_daily_news', 'sgns.renmin.bigram-char.bz2'),
    'word2vec_baike': PTM('pretrain/baidu_baike', 'sgns.merge.word'),
    'ctb50': PTM('pretrain/ctb50', 'ctb.50d.vrc'),
    'giga': PTM('pretrain/giga', 'gigaword_chn.all.a2b.uni.ite50.vec')
}