# -*-coding:utf-8 -*-

import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor


class Action:
    substitute = 'substitute'
    shuffle = 'shuffle'
    delete = 'delete'


class Augmenter(object):
    """
    Action: Delete, Swap, Substitute
    Granularity: char, word, entity, sentence
    """

    def __init__(self, min_sample, max_sample, prob, action=None, granularity=None):
        self.action = action
        self.granularity = granularity
        self.min_sample = min_sample
        self.max_sample = max_sample
        self.prob = prob

    def action(self, textt):
        """
        Core Augment action
        """
        raise NotImplementedError

    @staticmethod
    def _data_check(data):
        if not data or len(data) == 0:
            return False
        else:
            return True

    def _param_check(self):
        pass

    def augment(self, data, n_thread=4):
        """

        :param augment:
        :param data:
        :param n:
        :param n_thead:
        :return:
        """
        min_output = len(data) * self.min_sample
        max_output = len(data) * self.max_sample  # 默认增强样本<=原始样本
        max_retry = 3
        result = set()  # only keep non-dupulicate
        for _ in range(max_retry):
            with ThreadPoolExecutor(n_thread) as executor:
                for aug_data in executor.map(self.action, data):
                    if self._data_check(aug_data):
                        result.add(aug_data)
            if len(result) > min_output:
                break
        if len(result) > max_output:
            return random.sample(result, max_output)
        else:
            return result