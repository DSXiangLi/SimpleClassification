# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text
from sklearn.model_selection import train_test_split

Label2Idx = {
    '负面': 0,
    '正面': 1
}


def main():
    data_dir = './trainsample/dianping'
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None)
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None)

    train, valid = train_test_split(train, test_size=0.2)

    single_text(train[1], train[0], os.path.join(data_dir, 'train.txt'))
    single_text(valid[1], valid[0], os.path.join(data_dir, 'valid.txt'))
    single_text(test[1], test[0], os.path.join(data_dir, 'test.txt'))


if __name__ == '__main__':
    main()
