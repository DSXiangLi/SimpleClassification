# -*-coding:utf-8 -*-
"""
    Str Preprocess Handler Family

"""
import re
import string

__all__ = ['stop_word_handler', 'emoji_handler', 'punctuation_handler', 'text_emoji_handler','mention_handler']


class StrHandler(object):
    def __init__(self, file_path=None, **kwargs):
        self.file_path = file_path
        self.re_pattern = self.init(**kwargs)

    def init(self, **kwargs):
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


class TextEmojiHandler(StrHandler):
    def __init__(self, file_path='preprocess/data/emojis.txt', strict=False):
        super(TextEmojiHandler, self).__init__(file_path, strict=strict)

    def init(self, strict=False):
        if strict:
            emoji = self.readline()
            emoji = [i.replace('[', '\[').replace(']', '\]') for i in emoji]
            re_emoji = re.compile(r'%s' % '|'.join(emoji))
        else:
            re_emoji = re.compile(r'\[[\w\W\u4e00-\u9fff]{1,6}\]')
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


class EmojiHandler(StrHandler):
    def __init__(self):
        super(EmojiHandler, self).__init__()

    def init(self):
        re_emoji = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        return re_emoji


class MentionHandler(StrHandler):
    def __init__(self):
        super(MentionHandler, self).__init__()

    def init(self):
        re_mention = re.compile(r'@[\w\W\u4e00-\u9fff]+\s')
        return re_mention


stop_word_handler = StopWordHandler()
text_emoji_handler = TextEmojiHandler()
punctuation_handler = PunctuationHandler()
emoji_handler = EmojiHandler()
mention_handler = MentionHandler()

if __name__ == '__main__':
    print(emoji_handler.remove('记者都怒了[惊呆]'))
    print(punctuation_handler.remove('今天天气特别好！我们出去浪吧'))
    print(stop_word_handler.remove('具体说来，今天的事情'))
    print(emoji_handler.remove("How is your 🙈 and 😌. Have a nice weekend 💕👭👙"))
    print(mention_handler.remove('@小黑 你好么'))
    print(text_emoji_handler.check('[惊呆]'))
    print(text_emoji_handler.check('[惊]'))
    print(stop_word_handler.check('我们'))
    print(stop_word_handler.check('天气'))
    print(punctuation_handler.check('\n'))
    print(punctuation_handler.check('a'))


