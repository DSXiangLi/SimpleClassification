# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text, split_train_test

Label2Idx = {
    'news_story': 0,
    'news_culture': 1,
    'news_entertainment': 2,
    'news_sports': 3,
    'news_finance': 4,
    'news_house': 5,
    'news_car': 6,
    'news_edu': 7,
    'news_tech': 8,
    'news_military': 9,
    'news_travel': 10,
    'news_world': 11,
    'stock': 12,
    'news_agriculture': 13,
    'news_game': 14
}


def main():
    data_dir = './trainsample/toutiao'

    df = []
    with open(os.path.join(data_dir, 'toutiao_cat_data.txt'), 'r') as f:
        for line in f.readlines():
            df.append(line.strip().split('_!_'))
    df = pd.DataFrame(df, columns=['gid', 'cat_id', 'cat', 'title', 'keywords'])
    df['label'] = df['cat'].map(lambda x: Label2Idx[x])

    single_text(df['title'], df['label'], data_dir, output_file='all')
    split_train_test(data_dir, org_file='all')


if __name__ == '__main__':
    main()
