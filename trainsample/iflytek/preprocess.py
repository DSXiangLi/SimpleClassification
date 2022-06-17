# -*-coding:utf-8 -*-
import pandas as pd
import os
import json
from trainsample.converter import single_text, json2df, clue_submit


def load_data(file):
    lines = []
    with open(file, 'r') as f:
        for i in f.readlines():
            lines.append(json.loads(i))
    df = pd.DataFrame(lines)
    return df

labels  = load_data('./trainsample/iflytek/labels.json')

Idx2Label = labels.to_dict()['label_des']
Label2Idx = dict([(j,i) for i,j in Idx2Label.items()])


def main():
    data_dir = './trainsample/iflytek'

    train = load_data(os.path.join(data_dir, 'train.json'))
    train['label'] = train['label_des'].map(lambda x: Label2Idx[x])

    eval = load_data(os.path.join(data_dir, 'dev.json'))
    eval['label'] = eval['label_des'].map(lambda x: Label2Idx[x])

    test = load_data(os.path.join(data_dir, 'test.json'))
    test['label'] = 1 # for CLUE, you need to submit the file to CLUE dataset for eval

    single_text(train['sentence'], train['label'], data_dir, output_file='train')
    single_text(eval['sentence'], eval['label'], data_dir, output_file='valid')
    single_text(test['sentence'], test['label'], data_dir, output_file='test')


if __name__ == '__main__':
    #main()
    clue_submit('./trainsample/iflytek', 'iflytek_bert_test.txt',
                'iflytek_predict.json', Idx2Label, None)
