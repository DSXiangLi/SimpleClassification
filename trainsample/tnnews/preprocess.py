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
    'news_stock': 12,
    'news_agriculture': 13,
    'news_game': 14
}

Idx2Label = dict([(j,i) for i,j in Label2Idx.items()])
# 从0-14的预测转化为CLUE的label
Mapping = {
    0: 100,
    1: 101,
    2: 102,
    3: 103,
    4: 104,
    5: 106,
    6: 107,
    7: 108,
    8: 109,
    9: 110,
    10: 112,
    11: 113,
    12: 114,
    13: 115,
    14: 116
}


def load_data(file):
    lines = []
    with open(file, 'r') as f:
        for i in f.readlines():
            lines.append(json.loads(i))
    df = pd.DataFrame(lines)
    return df


def main():
    data_dir = './trainsample/tnnews'

    train = load_data(os.path.join(data_dir, 'train.json'))
    train['label'] = train['label_desc'].map(lambda x: Label2Idx[x])

    eval = load_data(os.path.join(data_dir, 'dev.json'))
    eval['label'] = eval['label_desc'].map(lambda x: Label2Idx[x])

    test = load_data(os.path.join(data_dir, 'test1.0.json'))
    test['label'] = 1 # for CLUE, you need to submit the file to CLUE dataset for eval

    single_text(train['sentence'], train['label'], data_dir, output_file='train')
    single_text(eval['sentence'], eval['label'], data_dir, output_file='valid')
    single_text(test['sentence'], test['label'], data_dir, output_file='test')


def clue_submit(data_dir, output, input):
    """
    convert test.csv to predict.json for CLUE submision
    {"id": 2, "label": "107", "label_desc": "news_car"}
    """
    df = load_data(os.path.join(data_dir, input))
    df['label_desc'] = df['label'].map(lambda x: Idx2Label[x])
    df['label'] = df['label'].map(lambda x: Mapping[x])

    with open(os.path.join(data_dir, output), 'w') as f:
        for idx, l, desc in zip(df['idx'], df['label'], df['label_desc']):
            f.write(json.dumps({'id': idx, 'label':l, 'label_desc': desc}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
