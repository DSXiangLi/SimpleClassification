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


if __name__ == '__main__':
    test_chinanews()