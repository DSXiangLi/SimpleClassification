# -*-coding:utf-8 -*-

"""
    Convert Trainsample into Default format
"""

import json
from collections import namedtuple


def single_text(text_list, label_list, file_name):
    Fmt = namedtuple('SingleText', ['text', 'label'])

    with open(file_name, 'w') as f:
        for t, l in zip(text_list, label_list):
            f.write(json.dumps(Fmt(t, l)._asdict(), ensure_ascii=False) + '\n')


def double_text(text_list1, text_list2, label_list, file_name):
    """
    双文本输入Iterable
    生成{'text1':'', 'text2':'', 'label':''}
    """
    Fmt = namedtuple('SingleText', ['text1', 'text2','label'])

    with open(file_name, 'w') as f:
        for t1, t2, l in zip(text_list1, text_list2,  label_list):
            f.write(json.dumps(Fmt(t1, t2, l)._asdict(), ensure_ascii=False) + '\n')


