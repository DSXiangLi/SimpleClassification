from trainsample.converter import json2df
import numpy as np
import pandas as pd
from trainsample.medical_i.preprocess import Idx2Label as Label_i
from trainsample.medical_j.preprocess import Idx2Label as Label_j


def merge_label(file1, file2):
    """
    合并i和jlabel生成提交文件
    """
    df1 = json2df(file1)
    df2 = json2df(file2)

    df1['label_i'] = df1['prob'].map(lambda x: np.argmax(x))
    df2['label_j'] = df2['prob'].map(lambda x: np.argmax(x))

    df = pd.concat([df1.loc[:,['text1', 'label_i']], df2.loc[:,['label_j']]], axis=1)

    df['label_i_name'] = df['label_i'].map(lambda x: Label_i[x])
    df['label_j_name'] = df['label_j'].map(lambda x: Label_j[x])
    df.index.names = ['id']
    return df


if __name__== '__main__':
    file1 ='./trainsample/medical_i/medical_i_mc_bert_test.txt'
    file2 = './trainsample/medical_j/medical_j_mc_bert_test.txt'

    df = merge_label(file1, file2)
    df.loc[:,['label_i','label_j']].to_csv('submit_bert_v3.csv')
    #df.loc[: ['text1','label_i_name']].head(100)