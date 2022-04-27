# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text, split_train_test

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

    single_text(df['comment'], df['rating'], data_dir, 'all.txt')
    split_train_test(data_dir, 'all.txt')


if __name__ == '__main__':
    main()
