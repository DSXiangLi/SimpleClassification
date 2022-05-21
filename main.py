# -*-coding:utf-8 -*-

import importlib
import tensorflow.compat.v1 as tf
from argparse import ArgumentParser
from tools.loss import LossHP, LossFunc
from tools.train_utils import clear_model, RUN_CONFIG
from dataset.tokenizer import PRETRAIN_CONFIG
from model.mixup import hp_parser as mixup_hp_parser
from model.temporal import hp_parser as temporal_hp_parser
from model.multisource import hp_parser as multisource_hp_parser
from model.adversarial import hp_parser as adversarial_hp_parser
from model.knowledge_distill import hp_parser as knowledge_distill_hp_parser


def main():
    parser = ArgumentParser()
    # 确认训练模型和Loss Function
    parser.add_argument("--model", default='bert', type=str)
    parser.add_argument("--loss", default='ce', type=str)

    # Semi-Supervised Method
    parser.add_argument('--use_mixup', action='store_true', default=False)  # 使用mixup
    parser.add_argument('--use_temporal', action='store_true', default=False)  # 使用Temporal

    # 领域迁移和对抗训练相关
    parser.add_argument('--use_multisource', action='store_true', default=False)  # 使用share private multisouce
    parser.add_argument('--use_adversarial', action='store_true', default=False)  # 使用share private adversarial

    # 模型蒸馏
    parser.add_argument('--knowledge_distill', action='store_true', default=False)  # 使用Knowledge Distill进行模型蒸馏

    # 导入模型特有HP
    model_name = parser.parse_known_args()[0].model
    model_hp_parser = getattr(importlib.import_module('model.{}.model'.format(model_name)), 'hp_parser')
    parser = model_hp_parser.append(parser)

    # 导入Loss特有HP
    loss_name = parser.parse_known_args()[0].loss
    loss_hp_parser = LossHP[loss_name]
    parser = loss_hp_parser.append(parser)

    # 导入半监督所需HP
    if parser.parse_known_args()[0].use_mixup:
        parser = mixup_hp_parser.append(parser)

    if parser.parse_known_args()[0].use_temporal:
        parser = temporal_hp_parser.append(parser)

    # 导入领域迁移相关HP
    if parser.parse_known_args()[0].use_multisource:
        parser = multisource_hp_parser.append(parser)

    if parser.parse_known_args()[0].use_adversarial:
        parser = adversarial_hp_parser.append(parser)

    # 导入模型蒸馏相关HP
    if parser.parse_known_args()[0].knowledge_distill:
        parser = knowledge_distill_hp_parser.append(parser)

    # 所有模型通用HP
    parser.add_argument('--nlp_pretrain_model', default='chinese_L-12_H-768_A-12', type=str)

    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--data_dir", type=str)  # 数据目录默认包含train/test/valid.txt,如果多输入用，分割

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

    # train/predict/export/predict
    parser.add_argument("--clear_model", action='store_true', default=False)
    parser.add_argument("--do_train", action='store_true', default=False)  # 训练
    parser.add_argument("--do_eval", action='store_true', default=False)  # 测试集预测 & 评估
    parser.add_argument("--do_export", action='store_true', default=False)  # 导出模型
    parser.add_argument("--do_predict", action='store_true', default=False)  # 对离线样本进行预测

    # 以下文件名常规任务不需要改动，对于增强任务，蒸馏任务需要修改为对应的训练，评估文件
    parser.add_argument('--train_file', default='train', type=str)  # 训练文件名，默认指代训练集
    parser.add_argument('--valid_file', default='valid', type=str)  # 验证文件名用于early stop，默认指代验证集
    parser.add_argument('--eval_file', default='test', type=str)  # 评估文件名，默认指代测试集
    parser.add_argument('--predict_file', default='all', type=str)  # 预测文件名，默认指代全样本

    # 其他
    parser.add_argument("--enable_cache", action='store_true', default=False)  # 使用之前tokenizer cache的特征
    parser.add_argument("--clear_cache", action='store_true', default=False)  # 清楚之前tokenizer cache的特征
    parser.add_argument("--thresholds", default='0.6,0.7,0.8,0.9')  # 评估F1的阈值

    args = parser.parse_args()

    CKPT_DIR = './checkpoint'
    EXPORT_DIR = './serving'
    DATA_DIR = './trainsample'

    TP = {
        'model': args.model,
        'ckpt_name': args.ckpt_dir,  # checkpoint 名称，用于指代当前模型版本，和为输出文件命名
        'ckpt_dir': os.path.join(CKPT_DIR, args.ckpt_dir),
        'export_dir': os.path.join(EXPORT_DIR, args.ckpt_dir),  # 这里导出模型和checkpoint默认保持同名
        # 默认预测文件为eval文件，生成文件名和ckpt相同，在distill中需要制定预测文件

        'train_file': args.train_file,
        'valid_file': args.valid_file,
        'eval_file': args.eval_file,
        'predict_file': args.predict_file,

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
        'thresholds': [float(i) for i in args.thresholds.split(',')]  # threshold list to evaluate F1/precision/recall
    }

    # Update TP
    TP = model_hp_parser.update(TP, args)
    if parser.parse_known_args()[0].use_mixup:
        TP = mixup_hp_parser.update(TP, args)

    if parser.parse_known_args()[0].use_temporal:
        TP = temporal_hp_parser.update(TP, args)

    if parser.parse_known_args()[0].use_multisource:
        TP = multisource_hp_parser.update(TP, args)

    if parser.parse_known_args()[0].use_adversarial:
        TP = adversarial_hp_parser.update(TP, args)

    if parser.parse_known_args()[0].knowledge_distill:
        TP = knowledge_distill_hp_parser.update(TP, args)

    # get loss function
    loss_hp = loss_hp_parser.parse(args)
    TP['loss_func'] = LossFunc[loss_name](**loss_hp)

    # 多数据源：得到任务列表和任务数以及label映射
    if args.use_multisource or args.use_adversarial:
        data_list = args.data_dir.split(',')
        TP['data_dir_list'] = [os.path.join(DATA_DIR, i) for i in data_list]

        idx2label = {}
        for data_dir in TP['data_dir_list']:
            label2idx = getattr(importlib.import_module('{}.preprocess'.format(data_dir[2:].replace('/', '.'))),
                                'Label2Idx')
            idx2label[data_dir] = dict([(j, i) for i, j in label2idx.items()])
        TP['idx2label'] = idx2label
    else:
        data_dir = os.path.join(DATA_DIR, args.data_dir)
        TP['data_dir'] = data_dir
        TP['data_dir_list'] = [data_dir]  # 兼容多任务TP
        label2idx = getattr(importlib.import_module('{}.preprocess'.format(data_dir[2:].replace('/', '.'))),
                            'Label2Idx')
        TP['idx2label'] = {data_dir: dict([(j, i) for i, j in label2idx.items()])}  # 兼容多任务

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
    elif args.use_temporal:
        from model.temporal import get_trainer
        trainer = get_trainer(args.model)
    elif args.use_multisource:
        from model.multisource import get_trainer
        trainer = get_trainer(args.model)
    elif args.use_adversarial:
        from model.adversarial import get_trainer
        trainer = get_trainer(args.model)
    elif args.knowledge_distill:
        from model.knowledge_distill import get_trainer
        trainer = get_trainer(args.model)
    else:
        trainer = getattr(importlib.import_module('model.{}.model'.format(args.model)), 'trainer')

    trainer.train(TP, RUN_CONFIG, args.do_train, args.do_eval, args.do_predict, args.do_export)


if __name__ == '__main__':
    import os

    # set logging level to WARN
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.WARN)
    main()
