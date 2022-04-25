# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text
from sklearn.model_selection import train_test_split

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

    train, valid = train_test_split(train, test_size=0.2)

    single_text(train['title'], train['label'], os.path.join(data_dir, 'train.txt'))
    single_text(valid['title'], valid['label'], os.path.join(data_dir, 'valid.txt'))
    single_text(test['title'], test['label'], os.path.join(data_dir, 'test.txt'))
    single_text(pd.concat([train,valid,test])['title'],
                pd.concat([train,valid,test])['label'], os.path.join(data_dir, 'all.txt'))  # 全部样本


if __name__ == '__main__':
    main()
