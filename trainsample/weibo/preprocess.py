# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text, split_train_test

Label2Idx = {
    '负面': 0,
    '正面': 1
}


def main():
    data_dir = './trainsample/weibo'
    df = pd.read_csv(os.path.join(data_dir, 'weibo_senti_100k.csv'))

    single_text(df['review'], df['label'], data_dir, 'all.txt')
    split_train_test(data_dir, 'all.txt')


if __name__ == '__main__':
    main()
