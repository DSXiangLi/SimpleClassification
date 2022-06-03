# -*-coding:utf-8 -*-
import pandas as pd
import os
import json
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


def main(add_unlabel):
    data_dir = './trainsample/toutiao'

    df = []
    with open(os.path.join(data_dir, 'toutiao_cat_data.txt'), 'r') as f:
        for line in f.readlines():
            df.append(line.strip().split('_!_'))
    df = pd.DataFrame(df, columns=['gid', 'cat_id', 'cat', 'title', 'keywords'])
    df['label'] = df['cat'].map(lambda x: Label2Idx[x])

    single_text(df['title'], df['label'], data_dir, output_file='all')
    split_train_test(data_dir, org_file='all')

    if add_unlabel:
        # 是否加入ChinaNews的未标注数据做半监督
        train = []
        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            for line in f.readlines():
                train.append(json.loads(line.strip()))
        # for unlabel dataset: label <0
        chinanews = pd.read_csv('./trainsample/chinanews/train.csv')
        chinanews = chinanews.loc[~chinanews.iloc[:,1].isnull(),:]
        chinanews = chinanews.sample(len(train)) # 需要控制下未标注样本的数据量, 这里和标注样本保持1比1

        for text in chinanews.iloc[:, 1]:
            train.append({'text1': text, 'label': -1})

        with open(os.path.join(data_dir, 'train_unlabel.txt'), 'w') as f:
            for i in train:
                f.writelines(json.dumps(i, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main(add_unlabel=False)
