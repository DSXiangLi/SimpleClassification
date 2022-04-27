# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text, split_train_test

Label2Idx = {
    '负面': 0,
    '正面': 1
}


def main():
    data_dir = './trainsample/waimai'
    df = pd.read_csv(os.path.join(data_dir, 'waimai_10k.csv'))

    single_text(df['review'], df['label'], data_dir, output_file='all')
    split_train_test(data_dir, org_file='all')


if __name__ == '__main__':
    main()
