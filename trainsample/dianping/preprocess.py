# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text, split_train_test

Label2Idx = {
    '负面': 0,
    '正面': 1
}


def main():
    data_dir = './trainsample/dianping'
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None)
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None)

    single_text(pd.concat([train, test])[1], pd.concat([train, test])[0], data_dir, output_file='all')
    split_train_test(data_dir, org_file='all')


if __name__ == '__main__':
    main()
