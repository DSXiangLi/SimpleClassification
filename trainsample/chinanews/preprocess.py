# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text, split_train_test

Label2Idx = {
    'mainland China politics': 0,
    'Hong Kong & Taiwan politics': 1,
    'International news': 2,
    'financial news': 3,
    'culture': 4,
    'entertainment': 5,
    'sports': 6
}


def main():
    data_dir = './trainsample/chinanews'

    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None)
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None)

    train.columns = ['label', 'title', 'content']
    test.columns = ['label', 'title', 'content']
    train = train.loc[~train['title'].isnull(), :]
    test = test.loc[~test['title'].isnull(), :]
    train['label'] = train['label'] - 1
    test['label'] = test['label'] - 1

    single_text(pd.concat([train, test])['title'], pd.concat([train, test])['label'], data_dir, output_file='all')
    split_train_test(data_dir, org_file='all')


if __name__ == '__main__':
    main()
