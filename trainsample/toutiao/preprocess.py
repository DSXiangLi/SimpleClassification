# -*-coding:utf-8 -*-
import pandas as pd
import os
from trainsample.converter import single_text
from sklearn.model_selection import train_test_split

CategoryMapping = {
    'news_story':0,
    'news_culture':1,
    'news_entertainment':2,
    'news_sports':3,
    'news_finance':4,
    'news_house':5,
    'news_car':6,
    'news_edu':7,
    'news_tech':8,
    'news_military':9,
    'news_travel':10,
    'news_world':11,
    'stock':12,
    'news_agriculture':13,
    'news_game':14
}

def main():
    data_dir = './trainsample/toutiao'

    df = []
    with open(os.path.join(data_dir, 'toutiao_cat_data.txt'), 'r') as f:
        for line in f.readlines():
            df.append(line.strip().split('_!_'))
    df = pd.DataFrame(df, columns = ['gid', 'cat_id', 'cat', 'title', 'keywords'])
    df['label'] = df['cat'].map(lambda x: CategoryMapping[x])

    train, test = train_test_split(df, test_size=0.2)
    train, valid = train_test_split(train, test_size=0.2)

    single_text(train['title'], train['label'], os.path.join(data_dir, 'train.txt'))
    single_text(valid['title'], valid['label'], os.path.join(data_dir, 'valid.txt'))
    single_text(test['title'], test['label'], os.path.join(data_dir, 'test.txt'))


if __name__ == '__main__':
    main()

