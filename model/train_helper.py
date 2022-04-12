# -*-coding:utf-8 -*-
import os
import json
import tensorflow as tf
from tools.train_utils import build_estimator, get_log_hook
from tools.logger import get_logger
from tools.metrics import binary_cls_report, binary_cls_metrics, pr_summary_hook


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

    def infer(self, file):
        self.logger.info('Running Prediction for test.text')
        self.input_pipe.build_feature(file)
        predictions = self.estimator.predict(self.input_pipe.build_input_fn(is_predict=True))
        predictions = [i['pred'] for i in predictions]
        return predictions

    def _eval(self):
        predictions = self.infer('test')
        self.logger.info('Dumping Prediction at {}'.format(os.path.join(self.train_params['data_dir'],
                                                                        self.train_params['predict_file'])))
        labels = [i['label'] for i in self.input_pipe.data_list]

        ##TODO: 区分多分类任务和二分类任务
        with open(os.path.join(self.train_params['data_dir'], self.train_params['predict_file']), 'w') as f:
            for probs, label in zip(predictions, labels):
                f.write(json.dumps({'pred': probs.tolist(), 'label': label}, ensure_ascii=False) + '\n')

        self.logger.info('='*10 + 'Evaluation Report' + '='*10)
        eval_report = binary_cls_report(predictions, labels, self.train_params['threshold'])
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


def build_model_fn(encoder):
    def model_fn(features, labels, params, mode):
        if labels is not None:
            labels = labels['label']
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        predictions = encoder(features, params, is_training)
        probs = tf.nn.softmax(predictions, axis=-1)

        # For prediction label is not used
        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode, predictions={'pred': predictions, 'probs': probs})
            return spec

        # Custom Loss function
        loss_func = params['loss_func']
        loss = loss_func(labels, predictions)
        total_loss = tf.reduce_mean(loss)

        # None for pretrain model, init_fn for word embedding
        scaffold = encoder.init_fn()

        if is_training:
            train_op = encoder.optimize(total_loss)

            spec = tf.estimator.EstimatorSpec(mode, loss=total_loss,
                                              train_op=train_op,
                                              scaffold = scaffold,
                                              training_hooks=[get_log_hook(total_loss, params['log_steps'])])

        else:
            metric_ops = binary_cls_metrics(probs, labels)
            summary_hook = pr_summary_hook(probs, labels, num_threshold=20,
                                           output_dir=params['model_dir'], save_steps=params['save_steps'])
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                              scaffold=scaffold,
                                              eval_metric_ops=metric_ops,
                                              evaluation_hooks=[summary_hook])
        return spec
    return model_fn