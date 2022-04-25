# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text
from sklearn.model_selection import train_test_split

Label2Idx = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5
}


def main():
    data_dir = './trainsample/douban'
    df = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))

    train, test = train_test_split(df, test_size=0.2)
    train, valid = train_test_split(train, test_size=0.2)

    single_text(train['comment'], train['rating'], os.path.join(data_dir, 'train.txt'))
    single_text(valid['comment'], valid['rating'], os.path.join(data_dir, 'valid.txt'))
    single_text(test['comment'], test['rating'], os.path.join(data_dir, 'test.txt'))
    single_text(pd.concat([train,valid,test])['comment'],
                pd.concat([train,valid,test])['rating'], os.path.join(data_dir, 'all.txt'))


if __name__ == '__main__':
    main()
