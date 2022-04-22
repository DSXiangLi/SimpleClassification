# -*-coding:utf-8 -*-
"""
    Tokenizer Adapter for Xlnet
"""

import sentencepiece as spm
from backbone.xlnet.prepro_utils import encode_pieces, preprocess_text


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.lower = do_lower_case
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(vocab_file)
        self.special_mapping = {
            '[CLS]': '<cls>',
            '[PAD]': '<pad>',
            '[SEP]': '<sep>',
            '[UNK]': '<unk>'
        }  # map xlnet special token to bert special token

    def tokenize(self, text):
        text = preprocess_text(text, lower=self.lower)
        tokens = encode_pieces(self.sp_model, text, return_unicode=False, sample=False)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.sp_model.PieceToId(self.special_mapping.get(i, i)) for i in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.sp_model.IdToPiece(id_) for id_ in ids]


if __name__ == '__main__':
    tokenizer = FullTokenizer(vocab_file='./pretrain/chinese_xlnet_base_L-12_H-768_A-12/spiece.model')
    text = '//@全球奇事趣闻:重口味[衰]'
    tokens = tokenizer.tokenize(text)
    tokenizer.convert_tokens_to_ids(tokens)
