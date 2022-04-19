# -*-coding:utf-8 -*-
import os
import importlib
import tensorflow as tf
from argparse import ArgumentParser
from tools.loss import LossHP, LossFunc
from tools.train_utils import clear_model, RUN_CONFIG
from dataset.tokenizer import PRETRAIN_CONFIG


def main():
    parser = ArgumentParser()
    # 确认训练模型和Loss Function
    parser.add_argument("--model", default='bert', type=str)
    parser.add_argument("--loss", default='ce', type=str)

    #Semi-Supervised Method
    parser.add_argument('--use_mixup', action='store_true', default=False)  # 使用mixup

    #多任务和对抗训练相关
    parser.add_argument('--use_multitask', action='store_true', default=False)  # 使用share private multitask
    parser.add_argument('--use_adversarial', action='store_true', default=False)  # 使用share private adversarial

    # 导入模型特有HP
    model_name = parser.parse_known_args()[0].model
    model_hp_parser = getattr(importlib.import_module('model.{}.model'.format(model_name)), 'hp_parser')
    parser = model_hp_parser.append(parser)

    # 导入Loss特有HP
    loss_name = parser.parse_known_args()[0].loss
    loss_hp_parser = LossHP[loss_name]
    parser = loss_hp_parser.append(parser)

    # 导入半监督所需HP
    mixup_hp_parser = getattr(importlib.import_module('model.mixup'), 'hp_parser')
    if parser.parse_known_args()[0].use_mixup:
        parser = mixup_hp_parser.append(parser)

    # 所有模型通用HP
    parser.add_argument('--nlp_pretrain_model', default='chinese_L-12_H-768_A-12', type=str)

    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--data_dir", type=str) # 数据目录默认包含train/test/valid.txt,如果多输入用，分割

    parser.add_argument("--max_seq_len", default=150, type=int)  # 文本最大长度
    parser.add_argument("--label_size", default=2, type=int)  # 文本最大长度
    parser.add_argument("--lr", default=2e-5, type=float)

    parser.add_argument("--epoch_size", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--early_stop_ratio", default=1, type=float)  # 遍历先x%的Eval就early stop

    parser.add_argument("--log_steps", default=100, type=float)
    parser.add_argument("--save_steps", default=1000, type=float)

    # GPU
    parser.add_argument("--use_gpu", action='store_true', default=False)
    parser.add_argument("--device", default='0', type=str)

    # train/predict/export
    parser.add_argument("--clear_model", action='store_true', default=False)
    parser.add_argument("--do_train", action='store_true', default=False)  # 训练
    parser.add_argument("--do_eval", action='store_true', default=False)  # 测试集预测 & 评估
    parser.add_argument("--do_export", action='store_true', default=False)  # 导出模型

    #其他
    parser.add_argument("--enable_cache", action='store_true', default=False)  # 使用之前tokenizer cache的特征
    parser.add_argument("--clear_cache", action='store_true', default=False)  # 清楚之前tokenizer cache的特征
    parser.add_argument("--thresholds", default='0.6,0.7,0.8,0.9')  # 评估F1的阈值

    args = parser.parse_args()

    CKPT_DIR = './checkpoint'
    EXPORT_DIR = './serving'
    DATA_DIR = './trainsample'

    TP = {
        'model': args.model,
        'ckpt_dir': os.path.join(CKPT_DIR, args.ckpt_dir),
        'export_dir': os.path.join(EXPORT_DIR, args.ckpt_dir),# 这里导出模型和checkpoint默认保持同名
        'data_dir': os.path.join(DATA_DIR, args.data_dir),
        'predict_file': args.ckpt_dir + '.txt', # 默认预测文件名和ckpt相同

        'nlp_pretrain_model': args.nlp_pretrain_model,
        'nlp_pretrain_dir': PRETRAIN_CONFIG[args.nlp_pretrain_model].model_dir,
        'nlp_pretrain_ckpt': os.path.join(*PRETRAIN_CONFIG[args.nlp_pretrain_model]),

        'max_seq_len': args.max_seq_len,
        'label_size': args.label_size,
        'lr': args.lr,
        'enable_cache': args.enable_cache,
        'clear_cache': args.clear_cache,

        'epoch_size': args.epoch_size,
        'batch_size': args.batch_size,
        'early_stop_ratio': args.early_stop_ratio,

        'log_steps': args.log_steps,
        'save_steps': args.save_steps,
        'thresholds': [float(i) for i in args.thresholds.split(',')] # threshold list to evaluate F1/precision/recall
    }

    TP = model_hp_parser.update(TP, args)
    if parser.parse_known_args()[0].use_mixup:
        TP = mixup_hp_parser.update(TP, args)

    # get loss function
    loss_hp = loss_hp_parser.parse(args)
    TP['loss_func'] = LossFunc[loss_name](**loss_hp)

    # 多分类问题：加入labelid到分类名称的映射
    if TP['label_size']>2:
        TP['label2idx'] = getattr(importlib.import_module('trainsample.{}.preprocess'.format(args.data_dir)), 'Label2Idx')
        TP['idx2label'] = dict([(j,i) for i,j in TP['label2idx'].items()])

    # 多任务问题：得到任务列表和任务数
    if ',' in TP['data_dir']:
        TP['data_dir'] = TP['data_dir'].split(',')
        TP['task_size'] = len(TP['data_dir'])
        if not args.use_multitask and not args.use_adversarial:
            raise ValueError('For multi data source, you must enable either mutlitask or adversarial')

    # 删除checkpoint，summary cache
    if args.clear_model:
        clear_model(TP['ckpt_dir'])
        tf.summary.FileWriterCache.clear()

    # 如果ckpt为空创建目录
    if not os.path.isdir(TP['ckpt_dir']):
        os.mkdir(TP['ckpt_dir'])

    RUN_CONFIG.update({
        'use_gpu': args.use_gpu,
        'log_steps': args.log_steps,
        'save_steps': args.save_steps,
        'summary_steps': args.save_steps
    })

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.use_mixup:
        from model.mixup import get_trainer
        trainer = get_trainer(args.model)
    elif args.use_multitask:
        from model.multitask import get_trainer
        trainer = get_trainer(args.model)
    elif args.use_adversarial:
        from model.adversarial import get_trainer
        trainer = get_trainer(args.model)
    else:
        trainer = getattr(importlib.import_module('model.{}.model'.format(args.model)), 'trainer')

    trainer.train(TP, RUN_CONFIG, args.do_train, args.do_eval, args.do_export)


if __name__ == '__main__':
    main()
