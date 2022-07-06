# 文本分类：支持多种预训练，半监督，领域迁移，降噪，对抗，蒸馏

## 支持任务

- [x] 兼容TF1 & TF2

### 输出
- [x] 二分类任务
- [x] 多分类任务 
- [ ] 有序多分类任务

### 输入
- [x] 单输入
- [x] 双输入

## 支持模型
### 文本预训练模型
- [x] Bert
- [x] Bert-wwm
- [x] Roberta
- [x] Albert
- [x] Xlnet
- [x] Electra

### 词袋模型
- [x] Fasttext
- [x] TextCNN
- [x] TextRCNN
- [ ] DPCNN


### 半监督 & 领域迁移框架
- [x] Multitask / Domain Transfer
- [x] Adversarial 
- [x] Mixup
- [x] Temporal Ensemble
- [ ] MixMatch

### 模型蒸馏
- [x] Knowledge distill
- [ ] distill bert 
- [ ] tiny bert 

### 文本增强
- [ ] EDA: Random Delete, Random Swap, Random Insert, Random Substitute
- [ ] Entity Replacement 
- [ ] Mask MLM 
- [ ] Vote

### Loss
- [x] Focal Loss
- [x] General Cross Entropy
- [x] Symmetric Cross Entropy
- [x] Peer cross Entropy
- [x] Bootstrapping

### Others
- [x] 分层learning rate
- [x] 模型推理


## 相关Blogs
- [小样本利器1.半监督一致性正则 Temporal Ensemble & Mean Teacher代码实现 ](https://www.cnblogs.com/gogoSandy/p/16340973.html)
- [小样本利器2.文本对抗+半监督 FGSM & VAT & FGM代码实现 ](https://www.cnblogs.com/gogoSandy/p/16419026.html)
- [Bert不完全手册1. Bert推理太慢？模型蒸馏 ](https://www.cnblogs.com/gogoSandy/p/15978982.html)
- [Bert不完全手册2. Bert不能做NLG？MASS/UNILM/BART ](https://www.cnblogs.com/gogoSandy/p/15996974.html)
- [Bert不完全手册3. Bert训练策略优化！RoBERTa & SpanBERT](https://www.cnblogs.com/gogoSandy/p/16038057.html)
- [Bert不完全手册4. 绕开BERT的MASK策略？XLNET & ELECTRA](https://www.cnblogs.com/gogoSandy/p/16065757.html)
- [Bert不完全手册5. BERT推理提速？训练提速!内存压缩！Albert ](https://www.cnblogs.com/gogoSandy/p/16265469.html)
