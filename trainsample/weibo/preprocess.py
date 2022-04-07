# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text
from sklearn.model_selection import train_test_split


def main():
    data_dir = './trainsample/weibo'
    df = pd.read_csv(os.path.join(data_dir, 'weibo_senti_100k.csv'))

    train, test = train_test_split(df, test_size=0.2)
    train, valid = train_test_split(train, test_size=0.2)

    single_text(train['review'], train['label'], os.path.join(data_dir, 'train.txt'))
    single_text(valid['review'], valid['label'], os.path.join(data_dir, 'valid.txt'))
    single_text(test['review'], test['label'], os.path.join(data_dir, 'test.txt'))


if __name__ == '__main__':
    main()
