# -*-coding:utf-8 -*-

"""
    Convert Trainsample into Default format
"""
import os
import numpy as np
import pandas as pd
import json
from collections import namedtuple
from sklearn.model_selection import train_test_split


def split_train_test(data_dir, org_file, surfix=None):
    """
    split org file into train/test/valid_{output_surfix}.txt with ratio 6:2:2
    """
    with open(os.path.join(data_dir, org_file + '.txt'), 'r') as f:
        lines = f.readlines()
    # 因为设置了种子，所以train/test/valid不会改变
    train, test = train_test_split(lines, test_size=0.2, random_state=1234)
    train, valid = train_test_split(train, test_size=0.25, random_state=1234)

    with open(os.path.join(data_dir, '_'.join(filter(None, ['train', surfix])) + '.txt'), 'w') as f1, \
            open(os.path.join(data_dir, '_'.join(filter(None, ['test', surfix])) + '.txt'), 'w') as f2, \
            open(os.path.join(data_dir, '_'.join(filter(None, ['valid', surfix])) + '.txt'), 'w') as f3:
        f1.writelines(train)
        f2.writelines(test)
        f3.writelines(valid)


def single_text(text_list, label_list, data_dir, output_file):
    Fmt = namedtuple('SingleText', ['text1', 'label'])

    with open(os.path.join(data_dir, output_file + '.txt'), 'w') as f:
        for t, l in zip(text_list, label_list):
            f.write(json.dumps(Fmt(t, l)._asdict(), ensure_ascii=False) + '\n')


def double_text(text_list1, text_list2, label_list, data_dir, output_file):
    """
    双文本输入Iterable
    生成{'text1':'', 'text2':'', 'label':''}
    """
    Fmt = namedtuple('SingleText', ['text1', 'text2', 'label'])

    with open(os.path.join(data_dir, output_file + '.txt'), 'w') as f:
        for t1, t2, l in zip(text_list1, text_list2, label_list):
            f.write(json.dumps(Fmt(t1, t2, l)._asdict(), ensure_ascii=False) + '\n')


def json2df(file):
    lines = []
    with open(file, 'r') as f:
        for i in f.readlines():
            lines.append(json.loads(i))
    df = pd.DataFrame(lines)
    return df


def clue_submit(data_dir, input, output, Idx2Label, Mapping=None):
    """
    convert test.csv to predict.json for CLUE submision
    {"id": 2, "label": "107", "label_desc": "news_car"}
    """
    df = json2df(os.path.join(data_dir, input))
    df['label'] = df['prob'].map(lambda x: np.argmax(x))
    df['label_desc'] = df['label'].map(lambda x: Idx2Label[x])

    if Mapping:
        #Remapping label id to original CLUE id
        df['label'] = df['label'].map(lambda x: Mapping[x])
    # return df

    with open(os.path.join(data_dir, output), 'w') as f:
        for idx, l, desc in zip(df['idx'], df['label'], df['label_desc']):
            f.write(json.dumps({'id': idx, 'label':l, 'label_desc': desc}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # Only Used in Distill Mode, to split teacher prediction
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_surfix', type=str)
    args = parser.parse_args()
    split_train_test(args.data_dir, args.input_file, args.output_surfix)
