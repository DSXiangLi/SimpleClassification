# -*-coding:utf-8 -*-
"""
    Str Preprocess Handler Family

"""
import re
import string


class StrHandler(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.re_pattern = self.init()

    def init(self):
        raise NotImplementedError

    def preprocess(self, text):
        text = text.strip()
        text = self.full2half(text)
        text = self.rm_dup_space(text)
        return text

    def remove(self, text, replace=" "):
        return self.re_pattern.sub(replace, self.preprocess(text))

    def check(self, text):
        if self.re_pattern.search(text):
            return True
        else:
            return False

    def readline(self):
        lines = []
        with open(self.file_path, encoding='UTF-8') as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                lines.append(line.strip())
        return lines

    @staticmethod
    def rm_dup_space(text):
        re_dup_spaces = re.compile(r'(\s)(\1+)')
        return re_dup_spaces.sub(" ", text)

    @staticmethod
    def full2half(text):
        # 全角转半角需要优先做，然后再做sub
        s = ''
        for c in text:
            num = ord(c)
            if num == 0x3000:
                num = 0x20
            elif 0xFF01 <= num <= 0xFF5E:
                num = num - 0xFEE0
            s += chr(num)
        return s


class EmojiHandler(StrHandler):
    def __init__(self, file_path='preprocess/data/emojis.txt'):
        super(EmojiHandler, self).__init__(file_path)

    def init(self):
        emoji = self.readline()
        emoji = [i.replace('[', '\[').replace(']', '\]') for i in emoji]
        re_emoji = re.compile(r'%s' % '|'.join(emoji))
        return re_emoji


class PunctuationHandler(StrHandler):
    def __init__(self, file_path='preprocess/data/puncts.txt'):
        super(PunctuationHandler, self).__init__(file_path)

    def init(self):
        puncts = self.readline()
        for p in list(string.punctuation):
            if p != '.':
                puncts.append('\\' + p)
        puncts.extend(['\r', '\n', '\t'])
        re_puncts = re.compile(r'%s' % "|".join(puncts))
        return re_puncts


class StopWordHandler(StrHandler):
    def __init__(self, file_path='preprocess/data/stop_words.txt'):
        super(StopWordHandler, self).__init__(file_path)

    def init(self):
        stop_words = self.readline()
        re_stop_words = re.compile(r'({})'.format('|'.join(stop_words)))
        return re_stop_words


def str_preprocess_pipeline(func_list):
    def helper(text):
        for func in func_list:
            text = func(text)
        return text.strip()

    return helper


if __name__ == '__main__':
    stop_word_handler = StopWordHandler()
    emoji_handler = EmojiHandler()
    punctuation_handler = PunctuationHandler()
    print(emoji_handler.remove('记者都怒了[惊呆]'))
    print(punctuation_handler.remove('今天天气特别好！我们出去浪吧'))
    print(stop_word_handler.remove('具体说来，今天的事情'))
    print(emoji_handler.check('[惊呆]'))
    print(emoji_handler.check('[惊]'))
    print(stop_word_handler.check('我们'))
    print(stop_word_handler.check('天气'))
    print(punctuation_handler.check('\n'))
    print(punctuation_handler.check('a'))