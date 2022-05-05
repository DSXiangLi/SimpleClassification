# -*-coding:utf-8 -*-
from serving.inference import SeqClassifyInfer, WordClassifyInfer


def test_chinanews():
    seq_client = SeqClassifyInfer(server_list=['localhost:8500'],
                                  max_seq_len=150,
                                  nlp_pretrain_model='albert_base',
                                  model_name='chinanews_albert',
                                  model_version=1,
                                  timeout=10)

    samples = [{"text1": "中国网协提高奖金分配比例 金花单飞后不愁资金", "label": 6},
               {"text1": "香港食物安全中心称内地方便面未发现塑化剂超标", "label": 1},
               {"text1": "日本各政党党首演讲拉票 “安倍经济学”再遭攻击", "label": 2},
               {"text1": "评好莱坞大片遭遇滑铁卢：中国人的挑剔，开始了", "label": 5},
               {"text1": "诚品书店进驻香港 别让实体书店成\"免费样品店\"", "label": 4},
               {"text1": "未来三天华南强降雨将趋减弱 西南降水将增多", "label": 0},
               {"text1": "两岸关帝庙信众共聚福建东山祖庙祭关帝(组图)", "label": 1},
               {"text1": "中国有条件有能力实现全年经济发展预期目标", "label": 3}]

    for i in samples:
        print('text={}, label={}, pred={}'.format(i['text1'], i['label'], seq_client.infer(i['text1'])))


def test_toutiao():
    seq_client = WordClassifyInfer(server_list=['localhost:8500'],
                                   max_seq_len=1000,
                                   nlp_pretrain_model='word2vec_baike',
                                   model_name='toutiao_fasttext_temporal',
                                   model_version=1,
                                   timeout=10)

    samples = [{"text1": "沃尔沃推出新款电动垃圾车，居然是为了避免噪音悄悄搬走垃圾", "label": 6},
                {"text1": "以色列警告称如果战机被击落将会轰炸俄军事基地，你怎么看？", "label": 9},
                {"text1": "“以木筑梦”黄杨木雕在浙博展出", "label": 1},
                {"text1": "女孩子如何在网约车环境里更好地保护自己？", "label": 6},
                {"text1": "黄金有什么价值？", "label": 4},
                {"text1": "Faker大魔王22岁生日，曾经统治世界的大魔王还会回来吗？", "label": 14},
                {"text1": "破净股数量超80只：市场见底信号？", "label": 12},
                {"text1": "三名美国人获释 特朗普感谢金正恩：美朝关系处于新起点", "label": 11},
                {"text1": "《复仇者联盟3》预售破亿，最终能超越《速度与激情8》吗？", "label": 2},
                {"text1": "中国光棍越来越多，国家对这方面有没有什么新政策？", "label": 13}]

    for i in samples:
        print('text={}, label={}, pred={}'.format(i['text1'], i['label'], seq_client.infer(i['text1'])))


if __name__ == '__main__':
    test_chinanews()
    #test_toutiao()