# -*-coding:utf-8 -*-
from preprocess.base_augment import Augmenter
from dataset.tokenizer import get_tokenizer
from preprocess.str_utils import *
from preprocess.base_augment import Action
import jieba
import random


class WordAugmenter(Augmenter):
    def __init__(self, min_sample, max_sample, prob, action, granularity='word'):
        super(WordAugmenter, self).__init__(action, granularity, min_sample, max_sample, prob)
        self.filters = [stop_word_handler, punctuation_handler, emoji_handler]

    def tokenize(self, text):
        return jieba.cut(text)

    def get_aug_index(self, tokens):
        index = set()
        for i, t in enumerate(tokens):
            if any(f.check(t) for f in self.filters):
                continue
            index.add(i)
        return index


class W2vSynomous(WordAugmenter):
    def __init__(self, min_sample, max_sample, prob, action=Action.substitute, model_name='word2vec_baike', topn=10):
        super(W2vSynomous, self).__init__(min_sample, max_sample, prob, action)
        self.tokenizer = get_tokenizer(model_name)
        self.model = self.tokenizer.model
        self.topn = topn

    def gen_synom(self, word):
        if random.random() < self.prob:
            try:
                nn = self.model.most_similar(word, topn=self.topn)
                return random.choice(nn)[0]
            except:
                return None
        else:
            return None

    def action(self, text):
        new_sample = []
        words = self.tokenize(text)
        flag = False
        for i, t in enumerate(words):
            if i in self.get_aug_index(words):
                self.gen_synom(t)
                flag = True
            else:
                new_sample.append(t)
        if flag:
            return new_sample
        else:
            return None


class WordnetSynomous(WordAugmenter):
    def __init__(self, min_sample, max_sample, prob, action=Action.substitute,
                 model_file='preprocess/data/word_net.txt'):
        super(WordnetSynomous, self).__init__(min_sample, max_sample, prob, action)
        self.wordnet = self.load(model_file)

    def load(self, file):
        wordnet = {}
        with open(file, 'r') as f:
            for line in f:
                line = line.strip().split(" ")
                if not line[0].endswith('='):
                    continue
                for i in range(1, len(line)):
                    wordnet[line[i]] = line[1:i] + line[(i + 1):]
        return wordnet

    def gen_synom(self, word):
        if word in self.wordnet and random.random() < self.prob:
            return random.choice(self.wordnet[word])
        else:
            return word

    def action(self, text):
        new_sample = []
        tokens = self.tokenize(text)
        flag = False
        for i, t in enumerate(tokens):
            if i in self.get_aug_index(tokens):
                self.gen_synom(t)
                flag = True
            else:
                new_sample.append(t)
        if flag:
            return tokens
        else:
            return None


class WordShuffle(WordAugmenter):
    def __init__(self, min_sample, max_sample, prob, action=Action.shuffle):
        super(WordShuffle, self).__init__(min_sample, max_sample, prob, action)

    def get_swap_pos(self, left, right):
        if random.random() < self.prob:
            return random.randint(left, right)
        else:
            return left - 1

    def action(self, text):
        new_sample = []
        tokens = self.tokenize(text)
        l = len(text)
        for i, t in enumerate(tokens):
            if i in self.get_aug_index(tokens):
                pos = self.get_swap_pos(i + 1, l - 1)
                tokens[i], tokens[pos] = tokens[pos], tokens[i]
            new_sample.append(tokens[i])
        return new_sample


class WordDelete(WordAugmenter):
    def __init__(self, min_sample, max_sample, prob, action=Action.delete):
        super(WordDelete, self).__init__(min_sample, max_sample, prob, action)

    def action(self, text):
        new_sample = []
        tokens = self.tokenize(text)
        for i, t in enumerate(tokens):
            if i in self.get_aug_index(tokens):
                if random.random()< self.prob:
                    continue
            new_sample.append(t)
        return new_sample


if __name__ == '__main__':
    word_delete = WordDelete(1, 3, 0.2)
    word_shuffle = WordShuffle(1, 3, 0.2)
    wordnet_syn = WordnetSynomous(1, 3, 0.2)
    w2v_syn = W2vSynomous(1, 3, 0.2)

