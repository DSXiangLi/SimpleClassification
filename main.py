# -*-coding:utf-8 -*-
import os
import importlib
import tensorflow as tf
from argparse import ArgumentParser
from tools.loss import LossHP, LossFunc
from tools.train_utils import clear_model, RUN_CONFIG
from pretrain.config import PRETRAIN_CONFIG


def main():
    parser = ArgumentParser()
    # 确认训练模型
    parser.add_argument("--model", default='bert', type=str)
    parser.add_argument("--loss", default='ce', type=str)

    # 导入模型特有HP
    model_name = parser.parse_known_args()[0].model
    model_hp_parser = getattr(importlib.import_module('model.{}.model'.format(model_name)), 'hp_parser')
    parser = model_hp_parser.append(parser)

    # 导入Loss特有HP
    loss_name = parser.parse_known_args()[0].loss
    loss_hp_parser = LossHP[loss_name]
    parser = loss_hp_parser.append(parser)

    # 所有模型通用HP
    parser.add_argument('--nlp_pretrain_model', default='chinese_L-12_H-768_A-12', type=str)

    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--data_dir", type=str) # 数据目录，默认包含train.txt/valid.txt/test.txt

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
    parser.add_argument("--do_train", action='store_true', default=False) # 训练
    parser.add_argument("--do_eval", action='store_true', default=False) # 测试集预测 & 评估
    parser.add_argument("--do_export", action='store_true', default=False) # 导出模型

    #其他
    parser.add_argument("--enable_cache", action='store_true', default=False) # 使用之前cache的特征
    parser.add_argument("--thresholds", default='0.6,0.7,0.8,0.9') # 评估F1的阈值

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

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

        'epoch_size': args.epoch_size,
        'batch_size': args.batch_size,
        'early_stop_ratio': args.early_stop_ratio,

        'log_steps': args.log_steps,
        'save_steps': args.save_steps,
        'thresholds': [float(i) for i in args.thresholds.split(',')] # threshold list to evaluate F1/precision/recall
    }

    TP = model_hp_parser.update(TP, args)

    # get loss function
    loss_hp = loss_hp_parser.parse(args)
    TP['loss_func'] = LossFunc[loss_name](**loss_hp)

    if args.clear_model:
        #删除checkpoint，summary cache，创建新的ckpt
        clear_model(TP['ckpt_dir'])
        os.mkdir(TP['ckpt_dir'])
        tf.summary.FileWriterCache.clear()

    RUN_CONFIG.update({
        'use_gpu': args.use_gpu,
        'log_steps': args.log_steps,
        'save_steps': args.save_steps,
        'summary_steps': args.save_steps
    })

    trainer = getattr(importlib.import_module('model.{}.model'.format(args.model)), 'trainer')
    trainer.train(TP, RUN_CONFIG, args.do_train, args.do_eval, args.do_export)


if __name__ == '__main__':
    main()
