# -*-coding:utf-8 -*-

import pandas as pd
import os
from trainsample.converter import single_text, split_train_test

Label2Idx = {
    '负面': 0,
    '正面': 1
}


def main():
    data_dir = './trainsample/chnsenticorp'
    train = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t')
    test = pd.read_csv(os.path.join(data_dir, 'test.tsv'), sep='\t')
    dev = pd.read_csv(os.path.join(data_dir, 'dev.tsv'), sep='\t')

    single_text(pd.concat([train, test, dev])['text_a'], pd.concat([train, test,dev])['label'], data_dir, output_file='all')
    single_text(train['text_a'], train['label'], data_dir, output_file='train')
    single_text(test['text_a'], test['label'], data_dir, output_file='test')
    single_text(dev['text_a'], dev['label'], data_dir, output_file='valid')


if __name__ == '__main__':
    main()
