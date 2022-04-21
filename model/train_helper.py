# -*-coding:utf-8 -*-
import os
import json
import tensorflow as tf
from tools.train_utils import build_estimator, get_log_hook
from tools.logger import get_logger
from tools.metrics import get_eval_report, get_metric_ops


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
        loss = loss_func(labels, predictions)
        total_loss = tf.reduce_mean(loss)
        return total_loss


def build_model_fn(encoder):
    def model_fn(features, labels, params, mode):
        if labels is not None:
            labels = labels['label']
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        predictions, labels = encoder(features, labels, params, is_training)
        probs = tf.nn.softmax(predictions, axis=-1)

        # For prediction label is not used
        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode, predictions={'prob': probs})
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
            if params.get('task_size',1)==1:
                # 单任务metrics
                metric_ops = get_metric_ops(probs, labels, params['idx2lable'])
            else:
                # 多任务metrics
                metric_ops = {}
                for task_id, (task_name, idx2label) in enumerate(params['idx2label'].items()):
                    task_idx = tf.where(tf.equal(features['task_ids'], task_id))
                    task_probs = tf.gather_nd(probs, task_idx)
                    task_labels = tf.gather_nd(labels, task_idx)
                    task_ops = get_metric_ops(task_probs, task_labels, idx2label)
                    metric_ops.update(dict([('task{}'.format(task_id) + i, j) for i,j in task_ops.items()]))

            spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                              scaffold=scaffold,
                                              eval_metric_ops=metric_ops)
        return spec
    return model_fn


class BaseTrainer(object):
    def __init__(self, model_fn, dataset_cls):
        self.model_fn = model_fn
        self.dataset_cls = dataset_cls
        self.estimator = None
        self.logger = None
        self.input_pipe = None
        self.train_params = None

    def prepare(self):
        """
        Init Input Pipe and update train_params with input specific parameters
        """
        raise NotImplementedError

    def _train(self):
        self.logger.info('=' * 10 + 'Training {} '.format(self.train_params['model']) + '=' * 10)
        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
            self.estimator, metric_name='loss',
            max_steps_without_decrease=int(self.train_params['steps_per_epoch'] * self.train_params['early_stop_ratio'])
        )
        train_spec = tf.estimator.TrainSpec(self.input_pipe.build_input_fn(),
                                            max_steps=self.train_params['num_train_steps'],
                                            hooks=[early_stopping_hook])
        self.input_pipe.build_feature('valid')
        eval_spec = tf.estimator.EvalSpec(self.input_pipe.build_input_fn(is_predict=True),
                                          steps=self.train_params['steps_per_epoch'],
                                          throttle_secs=60)
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    def _export(self):
        self.logger.info('Exporting Model for serving_model at {}'.format(self.train_params['export_dir']))
        self.estimator._export_to_train_params = False
        self.estimator.export_saved_model(self.train_params['export_dir'],
                                          lambda: self.input_pipe.build_serving_proto())

    def _eval(self):
        self.input_pipe.build_feature('test')
        for data_dir, idx2label in self.train_params['idx2label'].items():
            self.logger.info('Dumping Prediction at {}'.format(os.path.join(data_dir, self.train_params['predict_file'])))
            if self.train_params.get('task_size', 1)>1:
                predictions = self.estimator.predict(self.input_pipe.build_input_fn(is_predict=True, task_name=data_dir))
            else:
                predictions = self.estimator.predict(self.input_pipe.build_input_fn(is_predict=True))

            probs = [i['prob'] for i in predictions]
            labels = [i['label'] for i in self.input_pipe.samples]

            with open(os.path.join(data_dir, self.train_params['predict_file']), 'w') as f:
                for prob, label in zip(probs, labels):
                    f.write(json.dumps({'prob': prob.tolist(), 'label': label}, ensure_ascii=False) + '\n')

            self.logger.info('='*10 + 'Evaluation Report' + '='*10)

            eval_report = get_eval_report(probs, labels, idx2label, self.train_params['thresholds'])
            self.logger.info('\n' + eval_report + '\n')

    def train(self, train_params, run_config, do_train, do_eval, do_export):
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
            self._eval()
        if do_export:
            self._export()
