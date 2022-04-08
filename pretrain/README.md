## 预训练模型

下载预训练模型到当前Folder，受文件大小限制这里移除了checkpoint文件，请自行去以下链接下载完整模型
哈工大wwm bert的vocab文件和google bert一致，所以tokenizer加载哪个模型结果都是一样的。

1. bert_base: chinese_L-12_H-768_A-12
- 项目链接：https://github.com/google-research/bert
- 模型链接：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

2. bert_base_wwm: chinese_wwm_L-12_H-768_A-12
- 项目链接： https://github.com/ymcui/Chinese-BERT-wwm
- 模型链接：https://drive.google.com/open?id=1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi

3. roberta_base: roeberta_zh_L-24_H-1024_A-16
- 项目链接： https://github.com/brightmart/roberta_zh
- 模型链接：https://drive.google.com/file/d/1W3WgPJWGVKlU9wpUYsdZuurAIFKvrl_Y/view

4. word2vec_news/word2vec_baike: People's Daily News/百度百科 300d word+Ngram 预训练词向量
- 项目链接：https://github.com/Embedding/Chinese-Word-Vectors
- 模型链接：https://pan.baidu.com/s/1upPkA8KJnxTZBfjuNDtaeQ

5. fasttext: 中文fasttext
- 模型链接：https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz
- 项目链接: https://fasttext.cc/docs/en/crawl-vectors.html

6. Giga: Character embeddings
- 模型链接：https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing
- 模型链接2： https://pan.baidu.com/s/1pLO6T9D#list/path=%2F
- 项目链接：https://github.com/jiesutd/LatticeLSTM 复用LatticeLSTM

7. ctb50: Word(Lattice) embeddings (ctb.50d.vec) 包含字，词，ngram
- 模型链接：https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view
- 项目链接：https://github.com/jiesutd/LatticeLSTM 复用LatticeLSTM
