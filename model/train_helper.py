# -*-coding:utf-8 -*-
import os
import json
import tensorflow.compat.v1 as tf
from tools.train_utils import build_estimator, get_log_hook
from tools.logger import get_logger
from tools.metrics import get_eval_report, get_metric_ops
from dataset.tokenizer import get_tokenizer


class BaseEncoder(object):
    def __init__(self):
        self.params = None

    def get_input_mask(self, seq_len):
        maxlen = tf.reduce_max(seq_len)
        input_mask = tf.sequence_mask(seq_len, maxlen=maxlen)
        return input_mask

    def encode(self, features, is_training):
        """
        Raw data encoding logic
        """
        raise NotImplementedError

    def __call__(self, features, labels, params, is_training):
        """
        core computation goes here
        """
        raise NotImplementedError

    def init_fn(self):
        """
        checkpoint or variable initialization
        """
        raise NotImplementedError

    def optimize(self, loss):
        """
        :param loss:
        :return: train op
        """
        raise NotImplementedError

    def compute_loss(self, predictions, labels):
        loss_func = self.params['loss_func']
        loss = loss_func(predictions, labels)
        total_loss = tf.reduce_mean(loss)
        return total_loss


def build_model_fn(encoder):
    def model_fn(features, labels, params, mode):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if labels and not 'logit' in labels:
            # Distill 模型需要保留完整labels
            labels = labels['label']

        predictions, labels = encoder(features, labels, params, is_training)
        probs = tf.nn.softmax(predictions, axis=-1)

        # For prediction label is not used
        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode, predictions={'prob': probs, 'logit': predictions})
            return spec

        # Custom Loss function
        total_loss = encoder.compute_loss(predictions, labels)

        # None for pretrain model, init_fn for word embedding
        scaffold = encoder.init_fn()

        if is_training:
            train_op = encoder.optimize(total_loss)
            spec = tf.estimator.EstimatorSpec(mode, loss=total_loss,
                                              train_op=train_op,
                                              scaffold=scaffold,
                                              training_hooks=[get_log_hook(total_loss, params['log_steps'])])

        else:
            if params.get('task_size', 1) == 1:
                # 单任务metrics
                metric_ops = get_metric_ops(probs, labels, params['idx2label'][params['data_dir']])
            else:
                # 多任务metrics
                metric_ops = {}
                for task_id, (task_name, idx2label) in enumerate(params['idx2label'].items()):
                    weights = tf.cast(tf.equal(features['task_ids'], task_id), tf.float32)
                    task_ops = get_metric_ops(probs, labels, idx2label, weights)
                    metric_ops.update(dict([('task{}'.format(task_id) + i, j) for i, j in task_ops.items()]))

            spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                              scaffold=scaffold,
                                              eval_metric_ops=metric_ops)
        return spec

    return model_fn


class Trainer(object):
    def __init__(self, model_fn, dataset_cls):
        self.model_fn = model_fn
        self.dataset_cls = dataset_cls
        self.estimator = None
        self.logger = None
        self.input_pipe = None
        self.train_params = None

    def prepare(self):
        self.logger.info('Prepare dataset')
        self.input_pipe = self.dataset_cls(data_dir=self.train_params['data_dir'],
                                           batch_size=self.train_params['batch_size'],
                                           max_seq_len=self.train_params['max_seq_len'],
                                           tokenizer=get_tokenizer(self.train_params['nlp_pretrain_model']),
                                           enable_cache=self.train_params['enable_cache'],
                                           clear_cache=self.train_params['clear_cache'])
        self.input_pipe.build_feature(self.train_params['train_file'])
        self.train_params = self.input_pipe.update_params(self.train_params)

    def _train(self):
        self.logger.info('=' * 10 + 'Training {} '.format(self.train_params['model']) + '=' * 10)
        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
            self.estimator, metric_name='loss',
            max_steps_without_decrease=int(self.train_params['steps_per_epoch'] * self.train_params['early_stop_ratio'])
        )
        train_spec = tf.estimator.TrainSpec(self.input_pipe.build_input_fn(),
                                            max_steps=self.train_params['num_train_steps'],
                                            hooks=[early_stopping_hook])
        self.input_pipe.build_feature(self.train_params['valid_file'])
        eval_spec = tf.estimator.EvalSpec(self.input_pipe.build_input_fn(is_predict=True),
                                          steps=self.train_params['steps_per_epoch'],
                                          throttle_secs=60)
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    def _export(self):
        self.logger.info('Exporting Model for serving_model at {}'.format(self.train_params['export_dir']))
        self.estimator._export_to_train_params = False
        self.estimator.export_saved_model(self.train_params['export_dir'],
                                          lambda: self.input_pipe.build_serving_proto())

    def _infer(self, predict_only=False):
        # predict only: 只输出预测文件不做评估, 用于对全样本进行预估多用于模型蒸馏
        file = 'predict_file' if predict_only else 'eval_file'
        self.input_pipe.build_feature(self.train_params[file])

        for data_dir, idx2label in self.train_params['idx2label'].items():
            # 拼接ckpt & 输入文件名 得到预测输出文件名
            output_file = os.path.join(data_dir,
                                       '_'.join([self.train_params['ckpt_name'], self.train_params[file]]) + '.txt')
            self.logger.info('Dumping Prediction at {}'.format(output_file))
            if self.train_params.get('task_size', 1) > 1:
                predictions = self.estimator.predict(
                    self.input_pipe.build_input_fn(is_predict=True, task_name=data_dir))
            else:
                predictions = self.estimator.predict(self.input_pipe.build_input_fn(is_predict=True))

            preds = [{'prob': i['prob'].tolist(), 'logit': i['logit'].tolist()} for i in predictions]
            labels = [i['label'] for i in self.input_pipe.samples]

            with open(output_file, 'w') as f:
                for pred, data in zip(preds, self.input_pipe.raw_data):
                    # combine raw input and model prediction
                    f.write(json.dumps({**data, **pred}, ensure_ascii=False) + '\n')

            if not predict_only:
                self.logger.info('=' * 10 + 'Evaluation Report of {} '.format(self.train_params[file]) + '=' * 10)
                eval_report = get_eval_report([i['prob'] for i in preds],
                                              labels, idx2label, self.train_params['thresholds'])
                self.logger.info('\n' + eval_report + '\n')

    def train(self, train_params, run_config, do_train, do_eval, do_predict, do_export):
        self.train_params = train_params
        self.logger = get_logger(name=train_params['model'], log_dir=train_params['ckpt_dir'])

        self.prepare()
        self.estimator = build_estimator(self.model_fn, self.train_params, run_config)

        self.logger.info('=' * 10 + 'Train Params' + '=' * 10)
        self.logger.info(self.train_params)
        self.logger.info('=' * 10 + 'Run Config' + '=' * 10)
        self.logger.info(run_config)

        if do_train:
            self._train()
        if do_eval:
            self._infer()
        if do_predict:
            self._infer(predict_only=True)
        if do_export:
            self._export()


class MultiTrainer(Trainer):
    def __init__(self, model_fn, dataset_cls):
        super(MultiTrainer, self).__init__(model_fn, dataset_cls)

    def prepare(self):
        self.logger.info('Prepare dataset')
        self.input_pipe = self.dataset_cls(data_dir_list=self.train_params['data_dir_list'],
                                           batch_size=self.train_params['batch_size'],
                                           max_seq_len=self.train_params['max_seq_len'],
                                           tokenizer=get_tokenizer(self.train_params['nlp_pretrain_model']),
                                           enable_cache=self.train_params['enable_cache'],
                                           clear_cache=self.train_params['clear_cache'])
        self.input_pipe.build_feature('train')
        self.train_params = self.input_pipe.update_params(self.train_params)
